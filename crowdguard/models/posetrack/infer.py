"""YOLO-pose + ByteTrack inference: one pass, two results.

A single pose pass over a frame yields everything both downstream consumers
need: box centres for the crowd-motion aggregate, and 17 keypoints per person
for the anomaly features. Running one model instead of two is what makes this
affordable on a 6 GB GPU alongside CSRNet.

A separate YOLO instance is held per camera, because Ultralytics keeps
ByteTrack state inside the model object when ``persist=True``. Sharing one
instance across cameras would let tracks from different scenes contaminate each
other.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from crowdguard.config.schemas import AnomalyConfig, PoseTrackModelConfig
from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.models.contracts import (
    AnomalyLabel,
    AnomalyResult,
    MotionResult,
    PersonObservation,
)
from crowdguard.models.posetrack.classifier import AnomalyClassifier
from crowdguard.models.posetrack.features import build_feature_vector
from crowdguard.models.posetrack.tracker import CameraAnomalyTracker
from crowdguard.utils.calibration import GroundCalibration
from crowdguard.utils.geometry import box_center

logger = logging.getLogger(__name__)


@dataclass
class _TrackCenter:
    cx: float
    cy: float


class PoseTrackInference:
    """Pose detection, tracking, crowd motion, and anomaly labelling."""

    def __init__(
        self,
        pose_config: PoseTrackModelConfig,
        anomaly_config: AnomalyConfig,
        classifier: AnomalyClassifier | None = None,
        device: str = "cpu",
    ) -> None:
        self._pose_config = pose_config
        self._anomaly_config = anomaly_config
        self._classifier = classifier
        self._device = device

        self._models: dict[str, Any] = {}
        self._trackers: dict[str, CameraAnomalyTracker] = {}
        self._previous_centers: dict[str, dict[int, _TrackCenter]] = {}
        self._last_frame_id: dict[str, int] = {}
        self._last_loop_index: dict[str, int] = {}
        # Ground calibration per camera, and each track's last foot position on
        # the ground (metres), for walking speed in m/s.
        self._ground: dict[str, GroundCalibration] = {}
        self._previous_ground: dict[str, dict[int, tuple[float, float]]] = {}

        logger.info(
            "PoseTrackInference ready: model=%s tracker=%s device=%s conf=%.2f imgsz=%d "
            "classifier=%s",
            pose_config.model_path, pose_config.tracker_config, device,
            pose_config.confidence_threshold, pose_config.imgsz,
            classifier.model_name if classifier else "rule-based fallback",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def has_classifier(self) -> bool:
        return self._classifier is not None

    def tracker_for(self, camera_id: str, stride: int = 1) -> CameraAnomalyTracker:
        """Per-camera tracking state, configured for the stride frames arrive at.

        The anomaly constants are defined per frame at training rate. When the
        pose branch sees one frame in ``stride``, those constants must be rescaled
        or every motion feature is measured over the wrong span of time - the
        heading window alone would cover ``stride`` times too long. The stride is
        taken from each packet, so the tracker always matches what it is fed.
        """
        stride = max(1, int(stride))
        tracker = self._trackers.get(camera_id)
        if tracker is None:
            tracker = CameraAnomalyTracker(camera_id, self._anomaly_config.scaled_for_stride(stride))
            tracker.stride = stride
            self._trackers[camera_id] = tracker
        elif getattr(tracker, "stride", 1) != stride:
            tracker.config = self._anomaly_config.scaled_for_stride(stride)
            tracker.stride = stride
        return tracker

    def set_ground_calibration(self, camera_id: str, calibration: GroundCalibration | None) -> None:
        """Attach (or with None, remove) a camera's ground-plane calibration."""
        if calibration is None:
            self._ground.pop(camera_id, None)
        else:
            self._ground[camera_id] = calibration
        self._previous_ground.pop(camera_id, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, packet: FramePacket) -> tuple[MotionResult, AnomalyResult]:
        """Run one pose pass and derive both motion and anomaly results."""
        started = time.perf_counter()

        detections = self._detect(packet)
        inference_ms = (time.perf_counter() - started) * 1000.0

        continuity_ok, frames_skipped = self._check_continuity(packet)

        motion = self._aggregate_motion(packet, detections, inference_ms)
        anomaly = self._label_anomalies(
            packet, detections, inference_ms, continuity_ok, frames_skipped
        )
        return motion, anomaly

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _get_model(self, camera_id: str) -> Any:
        """Return this camera's YOLO instance, loading weights on first use."""
        if camera_id not in self._models:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for pose tracking. Install it with: "
                    "pip install ultralytics"
                ) from exc

            logger.info(
                "Loading pose model for camera %s (%s)", camera_id, self._pose_config.model_path
            )
            self._models[camera_id] = YOLO(self._pose_config.model_path)

        return self._models[camera_id]

    def _detect(
        self, packet: FramePacket
    ) -> list[tuple[tuple[float, float, float, float], int, np.ndarray | None]]:
        """Detect and track people, returning ``(bbox, track_id, keypoints)``."""
        cfg = self._pose_config
        model = self._get_model(packet.camera_id)

        precision = {"quantize": "fp16"} if (cfg.fp16 and self._device != "cpu") else {}

        try:
            results = model.track(
                packet.frame,
                persist=True,
                tracker=cfg.tracker_config,
                classes=[0],  # person
                conf=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                imgsz=cfg.imgsz,
                max_det=cfg.max_det,
                device=self._device,
                verbose=False,
                **precision,
            )
        except Exception as exc:
            logger.warning(
                "Pose inference failed on %s frame %s: %s",
                packet.camera_id, packet.frame_id, exc,
            )
            return []

        return self._extract(results)

    @staticmethod
    def _extract(
        results: list,
    ) -> list[tuple[tuple[float, float, float, float], int, np.ndarray | None]]:
        """Pull boxes, track ids and keypoints out of an Ultralytics result."""
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        keypoints = None
        if getattr(result, "keypoints", None) is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.cpu().numpy()

        extracted = []
        for index, (box, track_id) in enumerate(zip(boxes, ids)):
            person_kp = (
                keypoints[index] if keypoints is not None and index < len(keypoints) else None
            )
            extracted.append((
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                int(track_id),
                person_kp,
            ))
        return extracted

    # ------------------------------------------------------------------
    # Continuity
    # ------------------------------------------------------------------

    def _check_continuity(self, packet: FramePacket) -> tuple[bool, int]:
        """Detect frames dropped since this camera's previous processed frame.

        Every trajectory feature assumes evenly spaced observations. When the
        pose branch falls behind and frames are dropped, displacement and
        straightness are computed over a longer interval than the classifier was
        trained on. That cannot be fixed here, but it can be reported, so the
        result is marked and the operator sees it.
        """
        camera_id = packet.camera_id
        expected_gap = packet.anomaly_stride

        previous_loop = self._last_loop_index.get(camera_id)
        self._last_loop_index[camera_id] = packet.loop_index

        previous_id = self._last_frame_id.get(camera_id)
        self._last_frame_id[camera_id] = packet.frame_id

        # A source that just looped legitimately discontinues; reset instead of
        # reporting thousands of skipped frames.
        if previous_loop is not None and packet.loop_index != previous_loop:
            self.tracker_for(camera_id, packet.anomaly_stride).reset()
            self._previous_centers.pop(camera_id, None)
            self._previous_ground.pop(camera_id, None)
            self._reset_bytetrack(camera_id)
            return True, 0

        if previous_id is None:
            return True, 0

        actual_gap = packet.frame_id - previous_id
        if actual_gap <= expected_gap:
            return True, 0

        skipped = int((actual_gap - expected_gap) / max(1, expected_gap))
        return False, skipped

    # ------------------------------------------------------------------
    # Crowd motion
    # ------------------------------------------------------------------

    def _aggregate_motion(
        self,
        packet: FramePacket,
        detections: list[tuple[tuple[float, float, float, float], int, np.ndarray | None]],
        inference_ms: float,
    ) -> MotionResult:
        """Aggregate per-track displacement into crowd-level motion metrics.

        Velocity is reported in px/s, computed as per-frame displacement times
        the effective frame rate of this path. Expressing it in seconds rather
        than frames keeps the figure comparable regardless of stride, which is
        what lets the congestion thresholds stay fixed.
        """
        camera_id = packet.camera_id
        fps = packet.anomaly_fps if packet.anomaly_fps > 0 else 1.0

        previous = self._previous_centers.get(camera_id, {})
        current: dict[int, _TrackCenter] = {}

        velocities: list[float] = []
        displacements: list[tuple[float, float]] = []

        for bbox, track_id, _kp in detections:
            cx, cy = box_center(bbox)
            current[track_id] = _TrackCenter(cx=cx, cy=cy)

            if track_id in previous:
                dx = cx - previous[track_id].cx
                dy = cy - previous[track_id].cy
                distance = math.hypot(dx, dy)
                velocities.append(distance * fps)
                if distance > 0:
                    displacements.append((dx, dy))

        self._previous_centers[camera_id] = current

        result = self._build_motion_result(
            packet=packet,
            velocities=velocities,
            displacements=displacements,
            track_count=len(current),
            inference_ms=inference_ms,
        )
        calibration = self._ground.get(camera_id)
        if calibration is None or packet.frame is None:
            return result
        speed, in_zone = self._ground_speed(camera_id, calibration, packet, detections, fps)
        return replace(result, median_speed_ms=speed, zone_track_count=in_zone)

    MIN_ZONE_TRACKS = 3

    def _ground_speed(self, camera_id, calibration, packet, detections, fps) -> tuple[float | None, int]:
        """Median walking speed (m/s) of people whose feet are in the zone.

        A person's ground position is the bottom-centre of their box mapped
        through the homography. Only people inside the zone count, so speed and
        density describe the same patch of ground.
        """
        height, width = packet.frame.shape[:2]
        previous = self._previous_ground.get(camera_id, {})
        current: dict[int, tuple[float, float]] = {}
        if detections:
            feet = np.array([((b[0] + b[2]) / 2.0, b[3]) for b, _tid, _kp in detections])
            ground = calibration.to_ground(feet, width, height)
            inside = calibration.inside_ground(ground)
            speeds = []
            for (bbox, track_id, _kp), g, ok in zip(detections, ground, inside):
                if not ok:
                    continue
                current[track_id] = (float(g[0]), float(g[1]))
                if track_id in previous:
                    px, py = previous[track_id]
                    speeds.append(math.hypot(g[0] - px, g[1] - py) * fps)
        else:
            speeds = []
        self._previous_ground[camera_id] = current
        if len(speeds) < self.MIN_ZONE_TRACKS:
            return None, len(current)
        return float(np.median(speeds)), len(current)

    def _build_motion_result(
        self,
        packet: FramePacket,
        velocities: list[float],
        displacements: list[tuple[float, float]],
        track_count: int,
        inference_ms: float,
    ) -> MotionResult:
        if not velocities:
            return MotionResult(
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                source_frame_index=packet.source_frame_index,
                video_timestamp=packet.video_timestamp,
                wall_clock_timestamp=packet.wall_clock_timestamp,
                track_count=track_count,
                median_velocity=0.0,
                average_velocity=0.0,
                dominant_direction_deg=0.0,
                flow_consistency=0.0,
                active_motion_ratio=0.0,
                motion_confidence=0.0,
                inference_time_ms=inference_ms,
                model_name=self._pose_config.model_path,
                device=self._device,
                result_timestamp=time.time(),
            )

        array = np.asarray(velocities, dtype=np.float32)
        median_velocity = float(np.median(array))
        average_velocity = float(np.mean(array))

        flow_consistency = 0.0
        dominant_direction = 0.0
        if displacements:
            sum_dx = sum(d[0] for d in displacements)
            sum_dy = sum(d[1] for d in displacements)
            sum_magnitude = sum(math.hypot(d[0], d[1]) for d in displacements)
            vector_magnitude = math.hypot(sum_dx, sum_dy)
            if sum_magnitude > 0:
                flow_consistency = float(min(1.0, vector_magnitude / sum_magnitude))
            dominant_direction = float(math.degrees(math.atan2(sum_dy, sum_dx)) % 360.0)

        threshold = self._pose_config.motion_threshold_px
        moving = sum(1 for v in velocities if v > threshold)
        active_ratio = float(moving / len(velocities))

        # Confidence rises with coherent motion, with the share of people
        # actually moving, and with having enough people to average over.
        track_norm = min(1.0, track_count / 20.0)
        motion_confidence = float(min(1.0, max(0.0,
            flow_consistency * 0.5 + active_ratio * 0.3 + track_norm * 0.2
        )))

        return MotionResult(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            source_frame_index=packet.source_frame_index,
            video_timestamp=packet.video_timestamp,
            wall_clock_timestamp=packet.wall_clock_timestamp,
            track_count=track_count,
            median_velocity=median_velocity,
            average_velocity=average_velocity,
            dominant_direction_deg=dominant_direction,
            flow_consistency=flow_consistency,
            active_motion_ratio=active_ratio,
            motion_confidence=motion_confidence,
            inference_time_ms=inference_ms,
            model_name=self._pose_config.model_path,
            device=self._device,
            result_timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Anomaly labelling
    # ------------------------------------------------------------------

    def _label_anomalies(
        self,
        packet: FramePacket,
        detections: list[tuple[tuple[float, float, float, float], int, np.ndarray | None]],
        inference_ms: float,
        continuity_ok: bool,
        frames_skipped: int,
    ) -> AnomalyResult:
        """Update tracks, establish crowd flow, and classify each person."""
        tracker = self.tracker_for(packet.camera_id, packet.anomaly_stride)

        tracker.update_tracks(detections)
        crowd_direction, crowd_valid = tracker.update_crowd_direction()

        classifier_active = self._classifier is not None and self._anomaly_config.enabled

        mature = tracker.classifiable_tracks() if classifier_active else []
        mature_ids = {track_id for track_id, _ in mature}

        # Young tracks, and every track when no model is loaded, fall back to
        # the geometric counter-flow rule.
        for track_id, track in tracker.tracks.items():
            if track_id not in mature_ids:
                tracker.apply_rule_fallback(track)

        if mature:
            features = np.stack([
                build_feature_vector(
                    positions=list(track.positions),
                    step_speeds=list(track.step_speeds),
                    direction_rad=track.direction,
                    bbox=track.bbox,
                    keypoints=track.keypoints,
                    crowd_direction_rad=crowd_direction,
                    crowd_direction_valid=crowd_valid,
                    frame_height=packet.height,
                )
                for _track_id, track in mature
            ])

            try:
                predictions = self._classifier.predict(features)
            except Exception as exc:
                logger.warning(
                    "Anomaly classification failed on %s frame %s: %s",
                    packet.camera_id, packet.frame_id, exc,
                )
                predictions = [AnomalyLabel.NORMAL] * len(mature)

            for (_track_id, track), prediction in zip(mature, predictions):
                tracker.apply_prediction(track, prediction)

        people = self._build_observations(tracker, crowd_direction, crowd_valid)
        sitting_count, opposite_count = tracker.label_counts()

        return AnomalyResult(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            source_frame_index=packet.source_frame_index,
            video_timestamp=packet.video_timestamp,
            wall_clock_timestamp=packet.wall_clock_timestamp,
            people=people,
            sitting_count=sitting_count,
            opposite_count=opposite_count,
            tracked_count=len(tracker.tracks),
            crowd_direction_rad=crowd_direction,
            crowd_direction_valid=crowd_valid,
            classifier_active=classifier_active,
            continuity_ok=continuity_ok,
            frames_skipped=frames_skipped,
            inference_time_ms=inference_ms,
            model_name=self._pose_config.model_path,
            device=self._device,
            result_timestamp=time.time(),
            loop_index=packet.loop_index,
        )

    @staticmethod
    def _build_observations(
        tracker: CameraAnomalyTracker,
        crowd_direction: float,
        crowd_valid: bool,
    ) -> tuple[PersonObservation, ...]:
        """Snapshot the visible tracks as immutable observations."""
        observations: list[PersonObservation] = []

        for track_id, track in tracker.tracks.items():
            if track.bbox is None or track.frames_seen < 3:
                continue

            cos_sim = (
                float(np.cos(track.direction - crowd_direction)) if crowd_valid else 0.0
            )

            observations.append(PersonObservation(
                track_id=track_id,
                bbox=track.bbox,
                label=track.label,
                frames_seen=track.frames_seen,
                speed_px_per_frame=track.speed,
                direction_rad=track.direction,
                cos_sim_to_crowd=cos_sim,
                confirmed=track.label.is_anomalous,
                was_vetoed=track.was_vetoed,
                trail=track.trail(),
            ))

        return tuple(observations)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _reset_bytetrack(self, camera_id: str) -> None:
        """Clear the tracker state Ultralytics keeps inside the model object.

        With ``persist=True`` ByteTrack's tracks live on the predictor. Left in
        place across a clip loop or a new session, they hold positions from the
        end of the previous run and mis-associate the first detections of the
        next one. Resetting is cheaper than reloading the weights.
        """
        model = self._models.get(camera_id)
        predictor = getattr(model, "predictor", None) if model is not None else None
        for tracker in getattr(predictor, "trackers", None) or []:
            try:
                tracker.reset()
            except Exception:
                # Unknown tracker API: fall back to dropping the instance so the
                # next frame starts from a clean model.
                self._models.pop(camera_id, None)
                return

    def reset_tracking(self, camera_id: str) -> None:
        """Forget all tracking history for a camera, keeping its loaded model.

        Called at the start of each session so tracks, crowd direction and the
        continuity check all begin from nothing.
        """
        tracker = self._trackers.get(camera_id)
        if tracker is not None:
            tracker.reset()
        self._previous_centers.pop(camera_id, None)
        self._last_frame_id.pop(camera_id, None)
        self._last_loop_index.pop(camera_id, None)
        # the calibration describes the camera and survives a new session;
        # only the per-track ground positions are history
        self._previous_ground.pop(camera_id, None)
        self._reset_bytetrack(camera_id)

    def release_camera(self, camera_id: str) -> None:
        """Drop all per-camera state, including the model instance."""
        self._models.pop(camera_id, None)
        self._trackers.pop(camera_id, None)
        self._previous_centers.pop(camera_id, None)
        self._last_frame_id.pop(camera_id, None)
        self._last_loop_index.pop(camera_id, None)
        self._ground.pop(camera_id, None)
        self._previous_ground.pop(camera_id, None)

    def stats(self) -> dict:
        return {
            "device": self._device,
            "model": self._pose_config.model_path,
            "cameras_loaded": list(self._models.keys()),
            "classifier": self._classifier.model_name if self._classifier else None,
            "trackers": {cam: t.stats() for cam, t in self._trackers.items()},
        }
