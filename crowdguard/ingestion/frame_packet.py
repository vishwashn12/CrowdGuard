"""Immutable frame packet passed through the ingestion stream."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """One sampled frame plus its provenance.

    Frame format is a BGR ``uint8`` ndarray, matching OpenCV.

    A single packet instance is handed to both the density and the anomaly path,
    which is safe because the packet is frozen and the array is copied once on
    construction. Consumers must treat ``frame`` as read-only.

    ``frame_id`` is a per-camera monotonic counter over frames *read*, not
    frames emitted. Both paths therefore share one clock, which is what lets the
    temporal synchroniser compare a density frame against a motion frame. It
    also survives looping a short demo clip, where ``source_frame_index`` resets
    to zero.

    Attributes:
        camera_id: Source camera identifier.
        frame_id: Monotonic per-camera frame counter (shared clock).
        source_frame_index: Index within the current pass over the source.
        video_timestamp: Position within the source video, in seconds.
        wall_clock_timestamp: Capture time, seconds since epoch.
        frame: BGR uint8 image.
        source_fps: Native frame rate of the source.
        anomaly_stride: Decimation factor applied on the anomaly path.
        density_stride: Decimation factor applied on the density path.
        width: Frame width in pixels.
        height: Frame height in pixels.
        loop_index: How many times the source has restarted (0 on first pass).
    """

    camera_id: str
    frame_id: int
    source_frame_index: int
    video_timestamp: float
    wall_clock_timestamp: float
    frame: np.ndarray
    source_fps: float
    anomaly_stride: int = 1
    density_stride: int = 1
    width: int = 0
    height: int = 0
    loop_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        frame = np.asarray(self.frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be a 3-channel BGR image, got shape {frame.shape}")

        object.__setattr__(self, "frame", frame.copy())
        object.__setattr__(self, "camera_id", str(self.camera_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "source_frame_index", int(self.source_frame_index))
        object.__setattr__(self, "video_timestamp", float(self.video_timestamp))
        object.__setattr__(self, "wall_clock_timestamp", float(self.wall_clock_timestamp))
        object.__setattr__(self, "source_fps", float(self.source_fps))
        object.__setattr__(self, "anomaly_stride", max(1, int(self.anomaly_stride)))
        object.__setattr__(self, "density_stride", max(1, int(self.density_stride)))
        object.__setattr__(self, "loop_index", int(self.loop_index))

        height, width = frame.shape[:2]
        object.__setattr__(self, "width", int(width))
        object.__setattr__(self, "height", int(height))

    @property
    def anomaly_fps(self) -> float:
        """Effective frame rate seen by the pose/anomaly path.

        Used to convert per-frame pixel displacement into px/s.
        """
        return self.source_fps / self.anomaly_stride if self.source_fps > 0 else 0.0

    @property
    def density_fps(self) -> float:
        """Effective frame rate seen by the CSRNet path."""
        return self.source_fps / self.density_stride if self.source_fps > 0 else 0.0

    def describe(self) -> str:
        """Compact provenance string for logs."""
        return (
            f"{self.camera_id}#{self.frame_id} "
            f"(src {self.source_frame_index}, t={self.video_timestamp:.2f}s, "
            f"{self.width}x{self.height})"
        )
