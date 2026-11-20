"""Per-camera person tracking state for the anomaly branch.

Each camera owns one :class:`CameraAnomalyTracker`. It holds the trajectory
history that the 10-D features are computed from, establishes the dominant
crowd direction, and applies the two safeguards that make per-person labels
usable in a dense crowd:

**Occlusion re-stitching.** When a box disappears the track is retained for a
short buffer rather than destroyed, so a person who is briefly occluded keeps
their history and identity instead of becoming a new track.

**Sitting-track protection.** ID theft is the dominant failure mode in dense
crowds: a person walking in front of a seated person can inherit the seated
track, carrying the SITTING label across the scene as a false alarm. Two
independent guards prevent this - a mostly-stationary lost track will only
re-stitch to a detection within a few pixels, and any track labelled SITTING
while physically moving fast is vetoed outright.

This module does no inference. It is pure state, so it can be unit-tested
without a GPU or model weights.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from crowdguard.config.schemas import AnomalyConfig
from crowdguard.models.contracts import AnomalyLabel
from crowdguard.utils.geometry import (
    box_center,
    euclidean,
    exponential_circular_smooth,
    iou,
    weighted_circular_mean,
)

# How many crowd-direction samples are smoothed together.
_CROWD_DIRECTION_HISTORY = 30


class PersonTrack:
    """Trajectory and label state for one tracked person."""

    __slots__ = (
        "positions", "step_speeds", "direction", "speed", "frames_seen",
        "bbox", "keypoints", "label", "stationary_count",
        "rule_consec_opposite", "rule_consec_normal",
        "ml_consec_label", "ml_consec_count", "was_vetoed",
    )

    def __init__(self, history_length: int) -> None:
        self.positions: deque[tuple[float, float]] = deque(maxlen=history_length)
        self.step_speeds: deque[float] = deque(maxlen=history_length)
        self.direction: float = 0.0
        self.speed: float = 0.0
        self.frames_seen: int = 0
        self.bbox: tuple[float, float, float, float] | None = None
        self.keypoints: np.ndarray | None = None
        self.label: AnomalyLabel = AnomalyLabel.NORMAL
        self.stationary_count: int = 0

        # Rule-based fallback confirmation counters.
        self.rule_consec_opposite: int = 0
        self.rule_consec_normal: int = 0

        # ML temporal-smoothing counters.
        self.ml_consec_label: AnomalyLabel = AnomalyLabel.NORMAL
        self.ml_consec_count: int = 0

        self.was_vetoed: bool = False

    @property
    def is_sitting(self) -> bool:
        return self.label is AnomalyLabel.SITTING

    @property
    def is_opposite(self) -> bool:
        return self.label is AnomalyLabel.OPPOSITE

    @property
    def stationary_fraction(self) -> float:
        """Share of observed frames in which this track barely moved."""
        if self.frames_seen <= 0:
            return 0.0
        return self.stationary_count / self.frames_seen

    @property
    def recent_step_speed(self) -> float:
        """Mean of the last two per-frame steps, or 0 when too short.

        Deliberately short: the point is to catch a track that is moving *now*,
        not one whose long-run average is low.
        """
        if len(self.step_speeds) < 2:
            return 0.0
        return (self.step_speeds[-1] + self.step_speeds[-2]) / 2.0

    def trail(self, length: int = 30) -> tuple[tuple[float, float], ...]:
        """Most recent positions, for drawing a motion trail."""
        return tuple(list(self.positions)[-length:])


class CameraAnomalyTracker:
    """Tracking, crowd-flow estimation, and label smoothing for one camera."""

    def __init__(self, camera_id: str, config: AnomalyConfig) -> None:
        self.camera_id = camera_id
        # Stride the config was scaled for; kept in step by PoseTrackInference.
        self.stride = 1
        self.config = config

        self.tracks: dict[int, PersonTrack] = {}
        # track_id -> [track, frames_since_lost]
        self._lost_tracks: dict[int, list] = {}
        self._crowd_direction_history: deque[float] = deque(maxlen=_CROWD_DIRECTION_HISTORY)

        self.crowd_direction: float = 0.0
        self.crowd_direction_valid: bool = False

        self.restitch_count = 0
        self.veto_count = 0
        self.new_track_count = 0

    # ------------------------------------------------------------------
    # Phase 1: tracking
    # ------------------------------------------------------------------

    def update_tracks(
        self,
        detections: list[tuple[tuple[float, float, float, float], int, np.ndarray | None]],
    ) -> None:
        """Fold this frame's detections into the track set.

        Args:
            detections: ``(bbox, track_id, keypoints)`` triples for the frame.
        """
        cfg = self.config
        active_ids: set[int] = set()

        for bbox, track_id, keypoints in detections:
            active_ids.add(track_id)

            if track_id not in self.tracks:
                self.tracks[track_id] = self._adopt_or_create(bbox)

            track = self.tracks[track_id]
            track.keypoints = keypoints

            center = box_center(bbox)

            if track.positions:
                step = euclidean(center, track.positions[-1])
                track.step_speeds.append(step)
                if step < cfg.min_speed_px:
                    track.stationary_count += 1

            track.positions.append(center)
            track.bbox = tuple(float(v) for v in bbox)
            track.frames_seen += 1
            track.direction, track.speed = self._direction_and_speed(track.positions)

        self._retire_missing(active_ids)

    def _adopt_or_create(self, bbox: tuple[float, float, float, float]) -> PersonTrack:
        """Re-stitch a recently lost track onto this detection, or start fresh.

        The stationary guard is the important part. A lost track that was mostly
        stationary - a candidate seated person - may only be re-adopted by a
        detection whose centre is within a few pixels. Without it, a pedestrian
        walking across a seated person's last known box can overlap it enough to
        satisfy the IoU test and inherit the SITTING label.
        """
        cfg = self.config
        new_center = box_center(bbox)

        best_id: int | None = None
        best_iou = 0.0

        for lost_id, (lost_track, _age) in self._lost_tracks.items():
            if lost_track.bbox is None:
                continue

            if lost_track.stationary_fraction > 0.5 and lost_track.positions:
                distance = euclidean(new_center, lost_track.positions[-1])
                if distance > cfg.stationary_restitch_max_dist_px:
                    continue

            overlap = iou(bbox, lost_track.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_id = lost_id

        if best_id is not None and best_iou >= cfg.restitch_min_iou:
            track = self._lost_tracks.pop(best_id)[0]
            self.restitch_count += 1
            return track

        self.new_track_count += 1
        return PersonTrack(history_length=cfg.history_length)

    def _retire_missing(self, active_ids: set[int]) -> None:
        """Move vanished tracks into the lost buffer and age that buffer out."""
        for track_id in list(self.tracks.keys()):
            if track_id not in active_ids:
                self._lost_tracks[track_id] = [self.tracks.pop(track_id), 0]

        for lost_id in list(self._lost_tracks.keys()):
            self._lost_tracks[lost_id][1] += 1
            if self._lost_tracks[lost_id][1] > self.config.lost_track_buffer_frames:
                del self._lost_tracks[lost_id]

    def _direction_and_speed(
        self, positions: deque[tuple[float, float]]
    ) -> tuple[float, float]:
        """Heading and speed over the configured look-back window.

        Measuring across a window rather than between adjacent frames is what
        makes the heading robust to detector jitter.
        """
        if len(positions) < 2:
            return 0.0, 0.0

        window = min(self.config.direction_window, len(positions))
        old = positions[-window]
        new = positions[-1]
        dx = new[0] - old[0]
        dy = new[1] - old[1]

        speed = float(np.hypot(dx, dy) / window)
        direction = float(np.arctan2(dy, dx))
        return direction, speed

    # ------------------------------------------------------------------
    # Phase 2: crowd flow
    # ------------------------------------------------------------------

    def update_crowd_direction(self) -> tuple[float, bool]:
        """Estimate the dominant crowd heading for this frame.

        Faster people weigh more, since they carry the clearer signal of where
        the flow is going. The result is smoothed over recent frames so a
        transient cluster cannot swing the reference direction that every
        ``cos_sim`` is measured against.
        """
        cfg = self.config
        angles: list[float] = []
        weights: list[float] = []

        for track in self.tracks.values():
            if track.speed >= cfg.min_speed_px and track.frames_seen >= 3:
                angles.append(track.direction)
                weights.append(track.speed)

        if len(angles) < cfg.min_crowd_people:
            self.crowd_direction_valid = False
            return self.crowd_direction, False

        instantaneous = weighted_circular_mean(angles, weights)
        self._crowd_direction_history.append(instantaneous)

        self.crowd_direction = exponential_circular_smooth(list(self._crowd_direction_history))
        self.crowd_direction_valid = True
        return self.crowd_direction, True

    # ------------------------------------------------------------------
    # Phase 3: labelling
    # ------------------------------------------------------------------

    def apply_rule_fallback(self, track: PersonTrack) -> None:
        """Label a track by the cosine-similarity rule.

        Used for tracks too young for the classifier, and whenever no model is
        loaded. Requires several consecutive readings to flip state, which stops
        the label flickering on noise or a brief pause.
        """
        cfg = self.config

        if track.frames_seen < 3 or track.speed < cfg.min_speed_px or not self.crowd_direction_valid:
            is_opposite = False
        else:
            cos_sim = float(np.cos(track.direction - self.crowd_direction))
            is_opposite = cos_sim < cfg.opposite_threshold

        if is_opposite:
            track.rule_consec_opposite += 1
            track.rule_consec_normal = 0
        else:
            track.rule_consec_normal += 1
            track.rule_consec_opposite = 0

        if track.rule_consec_opposite >= cfg.confirm_frames:
            track.label = AnomalyLabel.OPPOSITE
        elif track.rule_consec_normal >= cfg.confirm_frames:
            track.label = AnomalyLabel.NORMAL

    def apply_prediction(self, track: PersonTrack, prediction: AnomalyLabel) -> bool:
        """Fold a classifier prediction into a track, with smoothing and veto.

        An anomaly must repeat for ``confirm_frames`` consecutive predictions
        before it is accepted, while a return to NORMAL takes effect at once:
        clearing a stale alarm should never be delayed.

        Returns:
            True if the track currently holds a confirmed anomaly label.
        """
        cfg = self.config

        if prediction == track.ml_consec_label:
            track.ml_consec_count += 1
        else:
            track.ml_consec_label = prediction
            track.ml_consec_count = 1

        track.was_vetoed = False

        if prediction is AnomalyLabel.NORMAL:
            track.label = AnomalyLabel.NORMAL
        elif track.ml_consec_count >= cfg.confirm_frames:
            track.label = prediction

        # Physics veto: a person who is moving is not sitting, whatever the
        # model says. This is the backstop against an inherited sitting track.
        if track.is_sitting and track.recent_step_speed > cfg.sitting_veto_speed_px:
            track.label = AnomalyLabel.NORMAL
            track.ml_consec_label = AnomalyLabel.NORMAL
            track.ml_consec_count = 0
            track.was_vetoed = True
            self.veto_count += 1

        return track.label.is_anomalous

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def classifiable_tracks(self) -> list[tuple[int, PersonTrack]]:
        """Tracks mature enough for the classifier, with a current box."""
        minimum = self.config.min_track_frames
        return [
            (track_id, track)
            for track_id, track in self.tracks.items()
            if track.frames_seen >= minimum and track.bbox is not None
        ]

    def count_nearby(self, track: PersonTrack) -> int:
        """Tracked people within the configured radius of this one."""
        if track.bbox is None:
            return 0
        center = box_center(track.bbox)
        radius = self.config.nearby_radius_px
        count = 0
        for other in self.tracks.values():
            if other is track or other.bbox is None:
                continue
            if euclidean(center, box_center(other.bbox)) < radius:
                count += 1
        return count

    def label_counts(self) -> tuple[int, int]:
        """``(sitting_count, opposite_count)`` over currently visible tracks."""
        sitting = sum(1 for t in self.tracks.values() if t.is_sitting and t.frames_seen >= 3)
        opposite = sum(1 for t in self.tracks.values() if t.is_opposite and t.frames_seen >= 3)
        return sitting, opposite

    def reset(self) -> None:
        """Drop all tracking state, e.g. when a source loops or is replaced."""
        self.tracks.clear()
        self._lost_tracks.clear()
        self._crowd_direction_history.clear()
        self.crowd_direction = 0.0
        self.crowd_direction_valid = False

    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "active_tracks": len(self.tracks),
            "lost_tracks": len(self._lost_tracks),
            "new_tracks": self.new_track_count,
            "restitched": self.restitch_count,
            "vetoed": self.veto_count,
            "crowd_direction_deg": round(float(np.degrees(self.crowd_direction) % 360.0), 1),
            "crowd_direction_valid": self.crowd_direction_valid,
        }
