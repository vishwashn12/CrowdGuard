"""Temporal synchronisation and the model-to-decision fusion boundary."""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import ROOT  # noqa: F401  (ensures sys.path setup)

from crowdguard.adapters.fusion_adapter import FusionAdapter
from crowdguard.adapters.synchronizer import MotionFreshness, TemporalSynchronizer
from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import MotionStatus, Trend
from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.models.contracts import CSRNetResult, MotionResult


def make_density(
    camera_id: str = "cam_1",
    frame_id: int = 100,
    crowd_count: float = 250.0,
    age_s: float = 0.0,
) -> CSRNetResult:
    density_map = np.full((10, 10), crowd_count / 100.0, dtype=np.float32)
    return CSRNetResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source_frame_index=frame_id,
        video_timestamp=frame_id / 25.0,
        wall_clock_timestamp=time.time(),
        crowd_count=crowd_count,
        density_map=density_map,
        density_map_width=10,
        density_map_height=10,
        input_width=1280,
        input_height=720,
        inference_time_ms=40.0,
        model_name="CSRNet",
        device="cpu",
        result_timestamp=time.time() - age_s,
    )


def make_motion(
    camera_id: str = "cam_1",
    frame_id: int = 100,
    velocity: float = 12.0,
    confidence: float = 0.8,
) -> MotionResult:
    return MotionResult(
        camera_id=camera_id,
        frame_id=frame_id,
        source_frame_index=frame_id,
        video_timestamp=frame_id / 25.0,
        wall_clock_timestamp=time.time(),
        track_count=18,
        median_velocity=velocity,
        average_velocity=velocity,
        dominant_direction_deg=90.0,
        flow_consistency=0.7,
        active_motion_ratio=0.6,
        motion_confidence=confidence,
        inference_time_ms=30.0,
        model_name="yolov8m-pose.pt",
        device="cpu",
    )


class TestFramePacket:
    def test_derives_dimensions_from_the_array(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(
            camera_id="c", frame_id=1, source_frame_index=1,
            video_timestamp=0.0, wall_clock_timestamp=time.time(),
            frame=frame, source_fps=25.0,
        )
        assert (packet.width, packet.height) == (640, 480)

    def test_rejects_non_image_arrays(self):
        with pytest.raises(ValueError, match="3-channel"):
            FramePacket(
                camera_id="c", frame_id=1, source_frame_index=1,
                video_timestamp=0.0, wall_clock_timestamp=time.time(),
                frame=np.zeros((480, 640), dtype=np.uint8), source_fps=25.0,
            )

    def test_copies_the_frame_so_mutation_cannot_leak(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = FramePacket(
            camera_id="c", frame_id=1, source_frame_index=1,
            video_timestamp=0.0, wall_clock_timestamp=time.time(),
            frame=frame, source_fps=25.0,
        )
        frame[0, 0] = 255
        assert packet.frame[0, 0, 0] == 0

    def test_effective_rates_follow_the_strides(self):
        packet = FramePacket(
            camera_id="c", frame_id=1, source_frame_index=1,
            video_timestamp=0.0, wall_clock_timestamp=time.time(),
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            source_fps=25.0, anomaly_stride=1, density_stride=5,
        )
        assert packet.anomaly_fps == pytest.approx(25.0)
        assert packet.density_fps == pytest.approx(5.0)


class TestSynchronizerCausality:
    def test_exact_match_is_new(self):
        sync = TemporalSynchronizer()
        sync.add_motion_result(make_motion(frame_id=100))
        result = sync.synchronize(make_density(frame_id=100))

        assert result.status is MotionFreshness.NEW
        assert result.is_valid_motion
        assert result.age_frames == 0

    def test_recent_past_is_reused(self):
        sync = TemporalSynchronizer(stale_frame_gap=10)
        sync.add_motion_result(make_motion(frame_id=95))
        result = sync.synchronize(make_density(frame_id=100))

        assert result.status is MotionFreshness.LAST_KNOWN
        assert result.is_valid_motion
        assert result.age_frames == 5

    def test_future_motion_is_refused(self):
        """A later observation must never justify an earlier decision."""
        sync = TemporalSynchronizer()
        sync.add_motion_result(make_motion(frame_id=120))
        result = sync.synchronize(make_density(frame_id=100))

        assert result.status is MotionFreshness.UNAVAILABLE
        assert not result.is_valid_motion
        assert "postdates" in result.reason

    def test_too_old_is_stale_and_excluded(self):
        sync = TemporalSynchronizer(stale_frame_gap=5)
        sync.add_motion_result(make_motion(frame_id=50))
        result = sync.synchronize(make_density(frame_id=100))

        assert result.status is MotionFreshness.STALE
        assert not result.is_valid_motion
        assert result.motion_result is not None  # retained for telemetry

    def test_empty_history_is_unavailable(self):
        sync = TemporalSynchronizer()
        result = sync.synchronize(make_density())
        assert result.status is MotionFreshness.UNAVAILABLE

    def test_newest_admissible_result_wins(self):
        sync = TemporalSynchronizer(stale_frame_gap=20)
        for frame_id in (80, 90, 95, 110):
            sync.add_motion_result(make_motion(frame_id=frame_id))

        result = sync.synchronize(make_density(frame_id=100))
        assert result.motion_result.frame_id == 95

    def test_cameras_never_cross(self):
        sync = TemporalSynchronizer()
        sync.add_motion_result(make_motion(camera_id="cam_1", frame_id=100))
        result = sync.synchronize(make_density(camera_id="cam_2", frame_id=100))
        assert result.status is MotionFreshness.UNAVAILABLE

    def test_forget_camera_drops_history(self):
        sync = TemporalSynchronizer()
        sync.add_motion_result(make_motion(camera_id="cam_1"))
        sync.forget_camera("cam_1")
        assert sync.synchronize(make_density(camera_id="cam_1")).status is MotionFreshness.UNAVAILABLE

    def test_stats_track_outcomes(self):
        sync = TemporalSynchronizer(stale_frame_gap=5)
        sync.add_motion_result(make_motion(frame_id=100))
        sync.synchronize(make_density(frame_id=100))
        sync.synchronize(make_density(frame_id=200))

        stats = sync.stats()
        assert stats["new"] == 1
        assert stats["stale"] == 1
        assert stats["total_queries"] == 2


class TestValidateSinglePair:
    def test_mismatched_cameras_rejected(self):
        result = TemporalSynchronizer.validate_single_pair(
            make_density(camera_id="cam_1"), make_motion(camera_id="cam_2")
        )
        assert not result.is_valid_motion
        assert "mismatch" in result.reason.lower()

    def test_missing_motion_is_unavailable(self):
        result = TemporalSynchronizer.validate_single_pair(make_density(), None)
        assert result.status is MotionFreshness.UNAVAILABLE

    def test_motion_without_density_is_accepted(self):
        result = TemporalSynchronizer.validate_single_pair(None, make_motion())
        assert result.is_valid_motion


class TestFusionAdapter:
    def test_both_halves_produce_a_confident_state(self):
        adapter = FusionAdapter(DecisionConfig())
        state = adapter.fuse(
            csrnet_result=make_density(crowd_count=250.0),
            motion_result=make_motion(velocity=12.0),
            gate_id="G1",
            gate_capacity=500,
        )
        assert state.gate_id == "G1"
        assert state.crowd_count == 250.0
        assert state.density_score == pytest.approx(0.5)
        assert state.has_density and state.has_motion
        assert state.motion_status is MotionStatus.OK

    def test_camera_mismatch_raises(self):
        adapter = FusionAdapter()
        with pytest.raises(ValueError, match="different cameras"):
            adapter.fuse(
                csrnet_result=make_density(camera_id="cam_1"),
                motion_result=make_motion(camera_id="cam_2"),
                gate_id="G1",
                gate_capacity=500,
            )

    def test_density_only_marks_motion_unavailable(self):
        adapter = FusionAdapter()
        state = adapter.fuse(
            csrnet_result=make_density(),
            motion_result=None,
            gate_id="G1",
            gate_capacity=500,
        )
        assert state.has_density
        assert not state.has_motion
        assert state.motion_status is MotionStatus.UNAVAILABLE
        assert state.velocity == 0.0

    def test_motion_only_marks_density_missing(self):
        adapter = FusionAdapter()
        state = adapter.fuse(
            csrnet_result=None,
            motion_result=make_motion(),
            gate_id="G1",
            gate_capacity=500,
        )
        assert not state.has_density
        assert state.has_motion
        assert state.crowd_count == 0.0

    def test_stale_motion_is_excluded_from_the_state(self):
        adapter = FusionAdapter()
        state = adapter.fuse(
            csrnet_result=make_density(frame_id=200),
            motion_result=make_motion(frame_id=100, velocity=30.0),
            gate_id="G1",
            gate_capacity=500,
            stale_frame_gap=10,
        )
        assert state.motion_status is MotionStatus.STALE
        assert not state.has_motion
        assert state.velocity == 0.0

    def test_observation_age_is_carried_through(self):
        adapter = FusionAdapter()
        state = adapter.fuse(
            csrnet_result=make_density(age_s=3.0),
            motion_result=make_motion(),
            gate_id="G1",
            gate_capacity=500,
        )
        assert state.result_age_s >= 2.9

    def test_confidence_blends_both_halves(self):
        adapter = FusionAdapter()
        both = adapter.fuse(
            csrnet_result=make_density(), motion_result=make_motion(confidence=0.6),
            gate_id="G1", gate_capacity=500,
        )
        motion_only = adapter.fuse(
            csrnet_result=None, motion_result=make_motion(confidence=0.6),
            gate_id="G1", gate_capacity=500,
        )
        # Density is treated as reliable, so pairing lifts a weak motion score.
        assert both.confidence > motion_only.confidence

    def test_trend_is_passed_through(self):
        adapter = FusionAdapter()
        state = adapter.fuse(
            csrnet_result=make_density(), motion_result=make_motion(),
            gate_id="G1", gate_capacity=500, trend=Trend.INCREASING,
        )
        assert state.trend is Trend.INCREASING

    def test_unmonitored_state_helper(self):
        adapter = FusionAdapter()
        state = adapter.unmonitored_state("G9", 400)
        assert state.gate_id == "G9"
        assert not state.has_density
        assert not state.has_motion
        assert state.crowd_count > 0  # neutral assumption, not empty

    def test_gate_id_is_never_a_camera_id(self):
        """The decision half must only ever see gates."""
        adapter = FusionAdapter()
        state = adapter.fuse(
            csrnet_result=make_density(camera_id="cam_99"),
            motion_result=make_motion(camera_id="cam_99"),
            gate_id="G3",
            gate_capacity=500,
        )
        assert state.gate_id == "G3"
        assert "cam" not in state.gate_id


class TestBoundaryDiscipline:
    def test_decision_modules_do_not_import_model_code(self):
        """The architectural invariant, enforced rather than documented.

        Only the adapter may see both halves. If a decision module grows an
        import of models or ingestion, swapping a model stops being a local
        change and this test is the early warning.
        """
        import ast
        from pathlib import Path

        decision_dir = Path(ROOT) / "crowdguard" / "decision"
        forbidden = ("crowdguard.models", "crowdguard.ingestion", "torch", "cv2", "ultralytics")

        offenders: list[str] = []
        for source_file in decision_dir.glob("*.py"):
            tree = ast.parse(source_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == bad or name.startswith(bad + ".") for bad in forbidden):
                        offenders.append(f"{source_file.name} imports {name}")

        assert offenders == [], "decision layer must stay model-agnostic: " + "; ".join(offenders)

    def test_venue_modules_do_not_import_model_code(self):
        import ast
        from pathlib import Path

        venue_dir = Path(ROOT) / "crowdguard" / "venue"
        forbidden = ("crowdguard.models", "torch", "cv2", "ultralytics")

        offenders: list[str] = []
        for source_file in venue_dir.glob("*.py"):
            tree = ast.parse(source_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == bad or name.startswith(bad + ".") for bad in forbidden):
                        offenders.append(f"{source_file.name} imports {name}")

        assert offenders == []
