"""Immutable result contracts for the vision models, with full provenance.

Three result types leave this layer:

* :class:`CSRNetResult`  - crowd density and count for one frame,
* :class:`MotionResult`  - crowd-level motion aggregated over tracked people,
* :class:`AnomalyResult` - per-person behaviour labels for one frame.

``MotionResult`` and ``AnomalyResult`` are produced by the *same* YOLO-pose pass
over a frame. Running one model and deriving both is what keeps a 6 GB GPU
viable: the pose pass yields box centres for the motion aggregate and keypoints
for the anomaly features at the same time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class AnomalyLabel(IntEnum):
    """Per-person behaviour class.

    The integer values are the class indices the trained classifier emits and
    must not be renumbered: they are baked into ``assets/best_model.pkl``.
    """

    NORMAL = 0
    SITTING = 1
    OPPOSITE = 2

    @property
    def display(self) -> str:
        return {0: "NORMAL", 1: "SITTING", 2: "OPPOSITE"}[int(self)]

    @property
    def is_anomalous(self) -> bool:
        return self is not AnomalyLabel.NORMAL


# ===================================================================
# CSRNet
# ===================================================================


@dataclass(frozen=True)
class CSRNetResult:
    """Density-estimation result for a single frame."""

    camera_id: str
    frame_id: int
    source_frame_index: int
    video_timestamp: float
    wall_clock_timestamp: float
    crowd_count: float
    density_map: np.ndarray
    density_map_width: int
    density_map_height: int
    input_width: int
    input_height: int
    inference_time_ms: float
    model_name: str
    device: str
    checkpoint_identifier: str = "CSRNet-ShanghaiTech-A"
    result_timestamp: float = 0.0
    frame: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.density_map, np.ndarray):
            raise TypeError("density_map must be a numpy.ndarray")

        density = np.asarray(self.density_map, dtype=np.float32)
        if density.ndim != 2:
            raise ValueError(f"density_map must be 2D, got shape {density.shape}")
        if density.shape != (self.density_map_height, self.density_map_width):
            raise ValueError(
                f"density_map shape {density.shape} does not match declared "
                f"{(self.density_map_height, self.density_map_width)}"
            )
        if not np.isfinite(self.crowd_count):
            raise ValueError(f"crowd_count must be finite, got {self.crowd_count}")
        if self.crowd_count < 0:
            raise ValueError(f"crowd_count cannot be negative, got {self.crowd_count}")

        object.__setattr__(self, "density_map", density.copy())
        if self.frame is not None:
            object.__setattr__(self, "frame", np.asarray(self.frame, dtype=np.uint8).copy())

        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "crowd_count", float(self.crowd_count))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "source_frame_index", int(self.source_frame_index))
        object.__setattr__(self, "video_timestamp", float(self.video_timestamp))
        object.__setattr__(self, "wall_clock_timestamp", float(self.wall_clock_timestamp))
        object.__setattr__(self, "inference_time_ms", float(self.inference_time_ms))
        object.__setattr__(self, "input_width", int(self.input_width))
        object.__setattr__(self, "input_height", int(self.input_height))
        object.__setattr__(
            self, "result_timestamp",
            time.time() if self.result_timestamp <= 0.0 else float(self.result_timestamp),
        )


# ===================================================================
# Crowd motion
# ===================================================================


@dataclass(frozen=True)
class MotionResult:
    """Crowd-level motion aggregated from per-person track displacement.

    Derived from the pose pass rather than dense optical flow: each tracked
    person contributes one displacement vector between consecutive observations,
    and those vectors are summed.

    Attributes:
        track_count: Active tracked people in this frame.
        median_velocity: 50th-percentile track velocity, px/s.
        average_velocity: Mean track velocity, px/s.
        dominant_direction_deg: Crowd heading in [0, 360), from the vector sum.
        flow_consistency: ``norm(sum v) / sum(norm v)`` in [0, 1]; 1 is uniform.
        active_motion_ratio: Fraction of tracks above the motion threshold.
        motion_confidence: Composite confidence in [0, 1].
        median_speed_ms: Median walking speed in m/s of people inside the
            calibrated ground zone; None when the camera is not calibrated or
            too few people are in the zone to measure.
        zone_track_count: Tracked people whose feet are inside the zone.
    """

    camera_id: str
    frame_id: int
    source_frame_index: int
    video_timestamp: float
    wall_clock_timestamp: float
    track_count: int
    median_velocity: float
    average_velocity: float
    dominant_direction_deg: float
    flow_consistency: float
    active_motion_ratio: float
    motion_confidence: float
    inference_time_ms: float
    model_name: str
    device: str
    result_timestamp: float = 0.0
    median_speed_ms: float | None = None
    zone_track_count: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.dominant_direction_deg < 360.0):
            raise ValueError(
                f"dominant_direction_deg must be in [0, 360), got {self.dominant_direction_deg}"
            )
        for name in ("flow_consistency", "active_motion_ratio", "motion_confidence"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.median_velocity < 0 or self.average_velocity < 0:
            raise ValueError("velocities must be non-negative")

        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "source_frame_index", int(self.source_frame_index))
        object.__setattr__(self, "video_timestamp", float(self.video_timestamp))
        object.__setattr__(self, "wall_clock_timestamp", float(self.wall_clock_timestamp))
        object.__setattr__(self, "track_count", int(self.track_count))
        object.__setattr__(self, "median_velocity", float(self.median_velocity))
        object.__setattr__(self, "average_velocity", float(self.average_velocity))
        object.__setattr__(self, "dominant_direction_deg", float(self.dominant_direction_deg))
        object.__setattr__(self, "flow_consistency", float(self.flow_consistency))
        object.__setattr__(self, "active_motion_ratio", float(self.active_motion_ratio))
        object.__setattr__(self, "motion_confidence", float(self.motion_confidence))
        object.__setattr__(self, "inference_time_ms", float(self.inference_time_ms))
        object.__setattr__(
            self, "result_timestamp",
            time.time() if self.result_timestamp <= 0.0 else float(self.result_timestamp),
        )


# ===================================================================
# Per-person anomaly
# ===================================================================


@dataclass(frozen=True)
class PersonObservation:
    """One tracked person in one frame, with their current behaviour label."""

    track_id: int
    bbox: tuple[float, float, float, float]
    label: AnomalyLabel
    frames_seen: int
    speed_px_per_frame: float
    direction_rad: float
    cos_sim_to_crowd: float
    confirmed: bool
    was_vetoed: bool = False
    trail: tuple[tuple[float, float], ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": [round(v, 1) for v in self.bbox],
            "label": self.label.display,
            "frames_seen": self.frames_seen,
            "speed": round(self.speed_px_per_frame, 2),
            "cos_sim": round(self.cos_sim_to_crowd, 3),
            "confirmed": self.confirmed,
            "was_vetoed": self.was_vetoed,
        }


@dataclass(frozen=True)
class AnomalyResult:
    """Per-person behaviour classification for one frame of one camera.

    Attributes:
        people: Every tracked person considered in this frame.
        sitting_count: Confirmed people sitting or collapsed in the flow.
        opposite_count: Confirmed people moving against the dominant flow.
        tracked_count: Total tracks held this frame.
        crowd_direction_rad: Smoothed dominant crowd heading, radians.
        crowd_direction_valid: Whether enough moving people were present to
            establish a flow direction at all.
        classifier_active: Whether the ML classifier ran, as opposed to the
            cosine-similarity fallback rule.
        continuity_ok: False when frames were skipped since the last processed
            frame for this camera, which degrades every trajectory feature.
        frames_skipped: How many frames were missed immediately before this one.
    """

    camera_id: str
    frame_id: int
    source_frame_index: int
    video_timestamp: float
    wall_clock_timestamp: float

    people: tuple[PersonObservation, ...] = ()
    sitting_count: int = 0
    opposite_count: int = 0
    tracked_count: int = 0

    crowd_direction_rad: float = 0.0
    crowd_direction_valid: bool = False

    classifier_active: bool = True
    continuity_ok: bool = True
    frames_skipped: int = 0

    inference_time_ms: float = 0.0
    model_name: str = ""
    device: str = "cpu"
    result_timestamp: float = field(default_factory=time.time)

    # Which pass over a looping clip this frame came from. Tracking restarts at
    # a loop, so the display must never interpolate a track id across one.
    loop_index: int = 0

    @property
    def anomaly_count(self) -> int:
        """Confirmed anomalies of any kind."""
        return self.sitting_count + self.opposite_count

    @property
    def has_anomaly(self) -> bool:
        return self.anomaly_count > 0

    @property
    def anomaly_ratio(self) -> float:
        """Share of tracked people currently flagged, in [0, 1]."""
        if self.tracked_count <= 0:
            return 0.0
        return min(1.0, self.anomaly_count / self.tracked_count)

    @property
    def crowd_direction_deg(self) -> float:
        return float(np.degrees(self.crowd_direction_rad) % 360.0)

    def anomalous_people(self) -> tuple[PersonObservation, ...]:
        return tuple(p for p in self.people if p.label.is_anomalous and p.confirmed)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "sitting_count": self.sitting_count,
            "opposite_count": self.opposite_count,
            "anomaly_count": self.anomaly_count,
            "tracked_count": self.tracked_count,
            "anomaly_ratio": round(self.anomaly_ratio, 3),
            "crowd_direction_deg": round(self.crowd_direction_deg, 1),
            "crowd_direction_valid": self.crowd_direction_valid,
            "classifier_active": self.classifier_active,
            "continuity_ok": self.continuity_ok,
            "frames_skipped": self.frames_skipped,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "people": [p.to_dict() for p in self.people if p.label.is_anomalous and p.confirmed],
        }
