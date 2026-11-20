"""Regression tests for two live-system bugs.

1. ``scaled_for_stride`` existed but was never applied, so at stride 7 the
   tracker ran with constants meant for every-frame analysis.
2. Pose analysis had no latency bound, so under GPU contention it drifted
   seconds behind live and every box was withdrawn as stale.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from conftest import ROOT  # noqa: F401  (ensures sys.path setup)

from crowdguard.config.schemas import AnomalyConfig, PoseTrackModelConfig
from crowdguard.ingestion.frame_buffer import FrameBuffer
from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.models.posetrack.infer import PoseTrackInference
from crowdguard.pipeline.posetrack_worker import PoseTrackWorker


def packet(frame_id: int, age_s: float = 0.0, stride: int = 1) -> FramePacket:
    return FramePacket(
        camera_id="cam",
        frame_id=frame_id,
        source_frame_index=frame_id,
        video_timestamp=frame_id / 27.0,
        wall_clock_timestamp=time.time() - age_s,
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        source_fps=27.0,
        anomaly_stride=stride,
    )


class TestStrideReachesTheTracker:
    def _engine(self) -> PoseTrackInference:
        # No weights are loaded until the first frame, so this needs no GPU.
        return PoseTrackInference(
            pose_config=PoseTrackModelConfig(),
            anomaly_config=AnomalyConfig(),
            classifier=None,
            device="cpu",
        )

    def test_tracker_gets_stride_scaled_constants(self):
        """The bug: at stride 7 the heading window must span ~15 source frames, not 105."""
        tracker = self._engine().tracker_for("cam", stride=7)
        assert tracker.config.direction_window == 2
        assert tracker.config.min_speed_px == 14.0
        assert tracker.config.sitting_veto_speed_px == 21.0

    def test_stride_one_keeps_training_constants(self):
        tracker = self._engine().tracker_for("cam", stride=1)
        assert tracker.config.direction_window == 15
        assert tracker.config.min_speed_px == 2.0

    def test_tracker_follows_a_stride_change(self):
        engine = self._engine()
        engine.tracker_for("cam", stride=1)
        tracker = engine.tracker_for("cam", stride=4)
        assert tracker.stride == 4
        assert tracker.config.direction_window == 4

    def test_scaling_covers_every_frame_counted_window(self):
        scaled = AnomalyConfig().scaled_for_stride(7)
        # Each still spans roughly the training-time duration.
        assert scaled.history_length == 9          # 60 / 7
        assert scaled.min_track_frames == 3        # 10 / 7, floored at 3
        assert scaled.confirm_frames == 2          # 5 / 7, floored at 2
        assert scaled.lost_track_buffer_frames == 4

    def test_history_always_longer_than_heading_window(self):
        for stride in range(1, 30):
            scaled = AnomalyConfig().scaled_for_stride(stride)
            assert scaled.history_length > scaled.direction_window


class _StubEngine:
    """Stands in for the pose model: records which frames were analysed."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def infer(self, pkt):
        self.seen.append(pkt.frame_id)
        motion = SimpleNamespace(inference_time_ms=1.0)
        anomaly = SimpleNamespace(continuity_ok=True)
        return motion, anomaly


class TestLatencyBound:
    def _run(self, packets, max_latency_s):
        queue = FrameBuffer(capacity=64, drop_policy="newest")
        engine = _StubEngine()
        worker = PoseTrackWorker(input_queue=queue, engine=engine, max_latency_s=max_latency_s)
        for pkt in packets:
            queue.put(pkt)
        worker.start()
        time.sleep(0.4)
        worker.stop()
        return engine, worker

    def test_late_frames_are_skipped(self):
        """Analysing a frame that is already stale only produces stale boxes."""
        engine, worker = self._run(
            [packet(1, age_s=2.0), packet(2, age_s=1.5), packet(3, age_s=0.0)],
            max_latency_s=0.4,
        )
        assert engine.seen == [3]
        assert worker.frames_skipped_late == 2

    def test_fresh_frames_are_all_analysed_in_order(self):
        engine, _ = self._run([packet(i) for i in range(1, 6)], max_latency_s=0.4)
        assert engine.seen == [1, 2, 3, 4, 5]

    def test_no_bound_means_nothing_skipped(self):
        engine, worker = self._run([packet(1, age_s=5.0)], max_latency_s=None)
        assert engine.seen == [1]
        assert worker.frames_skipped_late == 0

    def test_skips_count_as_degraded_tracking(self):
        """Skipping must be visible: it feeds the TRACKING_DEGRADED advisory."""
        _, worker = self._run(
            [packet(1, age_s=3.0), packet(2, age_s=3.0), packet(3)],
            max_latency_s=0.4,
        )
        stats = worker.stats()
        assert stats["late_skip_ratio"] > 0.5
        assert stats["continuity_break_ratio"] > 0.5

    def test_reset_clears_the_health_window(self):
        _, worker = self._run([packet(1, age_s=3.0)], max_latency_s=0.4)
        worker.reset_counters()
        stats = worker.stats()
        assert stats["continuity_break_ratio"] == 0.0
        assert stats["frames_skipped_late"] == 0
