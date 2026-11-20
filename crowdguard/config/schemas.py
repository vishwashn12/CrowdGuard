"""Pydantic schemas for the application configuration.

No torch / cv2 imports - these are plain validated data structures.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    """Optional dataset descriptor, used for display and relative path resolution."""

    name: str = Field(default="uploads", min_length=1)
    root: str = Field(default="uploads", min_length=1)


class CameraConfig(BaseModel):
    """A single video feed bound to a venue gate."""

    camera_id: str = Field(..., min_length=1)
    name: str | None = None
    source_type: Literal["video", "rtsp", "webcam"] = "video"
    source: str = Field(..., min_length=1)
    enabled: bool = True
    gate_id: str | None = Field(
        default=None,
        description="Gate this camera observes. If omitted, the venue config "
                    "camera_id -> gate mapping is used instead.",
    )


class IngestionConfig(BaseModel):
    """Dual-rate ingestion settings.

    A single read of each video feeds two sampling paths:

    * the density path is decimated to ``density_fps`` for CSRNet, and
    * the anomaly path keeps every ``anomaly_stride``-th frame for pose
      tracking, because the trajectory features are per-frame quantities.

    ``anomaly_stride = 1`` reproduces the frame rate the anomaly classifier was
    trained at. Raise it only when the GPU cannot keep up - see
    ``scripts/benchmark_gpu.py`` - and understand that doing so makes the
    classifier inputs an approximation (the stride is compensated for in
    :class:`AnomalyConfig`, but not exactly).
    """

    density_fps: float = Field(default=5.0, gt=0, description="Target FPS for the CSRNet path")
    anomaly_stride: int = Field(
        default=1, ge=1,
        description="Keep every Nth source frame for the pose/anomaly path (1 = every frame)",
    )
    real_time_mode: bool = Field(
        default=True,
        description="Pace reads to the source FPS. Disable to process a file as fast as possible.",
    )
    buffer_size: int = Field(default=30, gt=0)
    drop_policy: Literal["oldest", "newest", "drop_latest"] = "oldest"
    loop_sources: bool = Field(
        default=False,
        description="Restart video files on EOF - useful for demos with short clips.",
    )


class CSRNetModelConfig(BaseModel):
    enabled: bool = True
    checkpoint_path: str = "checkpoints/PartAmodel_best.pth"
    inference_resolution: tuple[int, int] | None = (720, 540)
    strict_checkpoint: bool = Field(
        default=True,
        description="Fail fast when the checkpoint is missing instead of emitting "
                    "meaningless counts from untrained regression weights.",
    )
    scales: tuple[float, ...] = Field(
        default=(0.7, 1.0, 1.3),
        description="Multi-scale inference factors. The three-scale default is the "
                    "paper pipeline the shipped checkpoint was calibrated against "
                    "and gives the most accurate count. A single scale (1.0) costs "
                    "about a third as much GPU time and under-counts by roughly "
                    "10%; see the calibration note below.",
    )

    @model_validator(mode="after")
    def validate_scales(self) -> "CSRNetModelConfig":
        if not self.scales:
            raise ValueError("scales must contain at least one factor")
        if any(s <= 0 for s in self.scales):
            raise ValueError(f"scales must all be positive, got {self.scales}")
        if not any(abs(s - 1.0) < 1e-6 for s in self.scales):
            # Scale 1.0 produces the density map used for the heat-map overlay,
            # and is the reference the per-scale area correction is relative to.
            raise ValueError(
                f"scales must include 1.0, got {self.scales}"
            )
        return self


class PoseTrackModelConfig(BaseModel):
    """YOLO-pose + ByteTrack configuration.

    One model instance is created per camera so that each stream ByteTrack
    state (``persist=True``) stays isolated.
    """

    model_config = {"protected_namespaces": ()}

    enabled: bool = True
    model_path: str = Field(
        default="assets/yolov8m-pose.pt",
        description="Pose weights. Must be a *-pose model: the anomaly features "
                    "need the 17 COCO keypoints. Downloaded on first run if absent.",
    )
    tracker_config: str = Field(default="bytetrack.yaml")
    confidence_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    imgsz: int = Field(
        default=1280, ge=320, le=1920,
        description="Inference resolution. Measured on labelled Hajj test footage, "
                    "yolov8m-pose found 16% of seated people at 960 but 26% at 1280 - "
                    "resolution moved recall more than model size did. Boxes are "
                    "returned in original-frame pixels at any setting, so this does "
                    "not shift the classifier's feature scale.",
    )
    fp16: bool = Field(
        default=True,
        description="Run inference in half precision on CUDA. Measured ~1.7x faster "
                    "with identical detections. Ignored on CPU.",
    )
    max_det: int = Field(default=300, gt=0)
    motion_threshold_px: float = Field(
        default=2.0, gt=0,
        description="Minimum per-track velocity (px/s) to count as moving "
                    "for the active-motion ratio.",
    )

    @model_validator(mode="after")
    def validate_pose_weights(self) -> "PoseTrackModelConfig":
        if self.enabled and "pose" not in self.model_path.lower():
            raise ValueError(
                f"model_path must be a YOLO pose checkpoint (got {self.model_path!r}). "
                f"The anomaly classifier requires keypoints; a detection-only model "
                f"cannot produce knee_bend_angle / head_to_hip_ratio / torso_height_ratio."
            )
        return self


class AnomalyConfig(BaseModel):
    """Per-person anomaly classifier settings.

    Defaults reproduce the constants the CatBoost model in ``assets/best_model.pkl``
    was trained with (see ``realtime/extract_features.py``). They are expressed
    per *processed* frame, so when ``anomaly_stride > 1`` the motion-related
    values are scaled by :meth:`scaled_for_stride`.
    """

    enabled: bool = True
    classifier_path: str = Field(default="assets/best_model.pkl")

    direction_window: int = Field(default=15, ge=2, description="Frames to look back for heading")
    min_speed_px: float = Field(default=2.0, gt=0, description="Below this a track counts as stationary")
    history_length: int = Field(default=60, ge=10, description="Positions remembered per track")

    min_track_frames: int = Field(default=10, ge=1, description="Frames before the classifier engages")
    confirm_frames: int = Field(default=5, ge=1, description="Consecutive predictions needed to flag")
    min_crowd_people: int = Field(default=3, ge=1, description="Moving tracks needed to define crowd flow")
    opposite_threshold: float = Field(default=-0.5, ge=-1.0, le=1.0, description="cos_sim rule-based fallback")

    sitting_veto_speed_px: float = Field(
        default=3.0, gt=0,
        description="A track labelled SITTING but moving faster than this is vetoed. "
                    "Defeats ID theft through occlusion.",
    )
    lost_track_buffer_frames: int = Field(
        default=30, ge=0,
        description="Frames a vanished track is retained for re-stitching",
    )
    restitch_min_iou: float = Field(default=0.3, ge=0.0, le=1.0)
    stationary_restitch_max_dist_px: float = Field(
        default=30.0, gt=0,
        description="A mostly-stationary lost track may only be re-stitched to a "
                    "detection within this distance, so walkers cannot inherit "
                    "a sitting track history.",
    )
    nearby_radius_px: float = Field(default=150.0, gt=0)

    def scaled_for_stride(self, stride: int) -> "AnomalyConfig":
        """Return a copy with per-frame motion quantities rescaled for a stride.

        At ``stride = k`` the pose branch sees every k-th frame, so a track moves
        k times further between observations and a fixed frame window spans k
        times more wall-clock time. Distances are multiplied by k and windows are
        divided by k.

        Every window measured in frames is shrunk so it still spans the same
        stretch of time as at training: the heading window, the position history
        that ``displacement`` and ``straightness`` are taken over, the minimum
        track age before classification, the confirmation run, and the lost-track
        buffer.

        This is a linear approximation. ``displacement`` and ``min_speed`` rescale
        exactly; ``straightness`` and ``cos_sim`` do not, because path curvature is
        lost when frames are skipped. Prefer ``stride = 1``.
        """
        if stride <= 1:
            return self
        direction_window = max(2, round(self.direction_window / stride))
        return self.model_copy(update={
            "direction_window": direction_window,
            "history_length": max(direction_window + 2, round(self.history_length / stride)),
            "min_track_frames": max(3, round(self.min_track_frames / stride)),
            "confirm_frames": max(2, round(self.confirm_frames / stride)),
            "min_speed_px": self.min_speed_px * stride,
            "sitting_veto_speed_px": self.sitting_veto_speed_px * stride,
            "stationary_restitch_max_dist_px": self.stationary_restitch_max_dist_px * stride,
            "lost_track_buffer_frames": max(1, round(self.lost_track_buffer_frames / stride)),
        })


class ModelsConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    csrnet: CSRNetModelConfig = Field(default_factory=CSRNetModelConfig)
    posetrack: PoseTrackModelConfig = Field(default_factory=PoseTrackModelConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)


class DisplayConfig(BaseModel):
    """Operator video playback, kept separate from analysis.

    The video panes play every source frame at the source frame rate, paced by
    the session clock, regardless of how fast or slow the models run. Analysis
    only decides how often the overlay boxes refresh.
    """

    delay_s: float = Field(
        default=0.5, ge=0.0, le=5.0,
        description="Playback runs this far behind live, so the analysis for each "
                    "displayed frame is already finished and boxes sit exactly on the "
                    "people rather than trailing them. Must exceed the pose stride "
                    "interval plus inference latency to fully hide it.",
    )
    width: int = Field(
        default=960, ge=320, le=1920,
        description="Display frames are scaled to this width before overlay and JPEG "
                    "encoding. Every frame of every camera is encoded, so this is the "
                    "main lever on CPU cost and browser bandwidth.",
    )
    jpeg_quality: int = Field(default=75, ge=30, le=95)
    buffer_s: float = Field(
        default=1.5, gt=0.0, le=10.0,
        description="Seconds of display frames retained per camera. Must exceed delay_s.",
    )

    @model_validator(mode="after")
    def validate_buffer(self) -> "DisplayConfig":
        if self.buffer_s <= self.delay_s:
            raise ValueError(
                f"buffer_s ({self.buffer_s}) must exceed delay_s ({self.delay_s}), "
                f"or the frame due for display will already have been discarded"
            )
        return self


class UploadConfig(BaseModel):
    """Settings for operator-uploaded footage."""

    directory: str = Field(default="uploads")
    max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0, description="Per-file upload cap")
    allowed_suffixes: tuple[str, ...] = Field(default=(".mp4", ".avi", ".mov", ".mkv", ".m4v"))


class AppConfig(BaseModel):
    """Root application configuration."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    uploads: UploadConfig = Field(default_factory=UploadConfig)
    venue_config: str = Field(default="config/venue/circular_venue.yaml")

    @model_validator(mode="after")
    def validate_unique_camera_ids(self) -> "AppConfig":
        ids = [c.camera_id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate camera_ids in configuration: {ids}")
        return self

    @property
    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]
