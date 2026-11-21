"""Ground-plane calibration: geometry, zone counting, physical scoring."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from crowdguard.adapters.fusion_adapter import FusionAdapter
from crowdguard.alerts.rules import GateView, generate_alerts
from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import CongestionLevel
from crowdguard.decision.fusion import build_crowd_state, calibrated_capacity
from crowdguard.models.contracts import CSRNetResult, MotionResult
from crowdguard.pipeline.calibration_store import CalibrationStore
from crowdguard.utils.calibration import CalibrationError, GroundCalibration, apply_homography

# a trapezoid as a floor rectangle looks in perspective: wide near, narrow far
TRAPEZOID = ((0.1, 0.9), (0.9, 0.9), (0.7, 0.4), (0.3, 0.4))


def calib(**kw) -> GroundCalibration:
    return GroundCalibration(points=kw.get("points", TRAPEZOID),
                             width_m=kw.get("width_m", 10.0), length_m=kw.get("length_m", 8.0))


class TestGeometry:
    def test_corners_map_to_the_rectangle(self):
        c = calib()
        h = c.homography(1000, 800)
        corners = np.array(TRAPEZOID) * [1000, 800]
        ground = apply_homography(h, corners)
        assert np.allclose(ground, [[0, 0], [10, 0], [10, 8], [0, 8]], atol=1e-6)

    def test_area(self):
        assert calib().area_m2 == 80.0

    def test_far_pixels_cover_more_ground_than_near_ones(self):
        """Perspective: one pixel step near the top is more metres than near the bottom."""
        c = calib()
        near = c.to_ground(np.array([[500, 700], [510, 700]]), 1000, 800)
        far = c.to_ground(np.array([[500, 340], [510, 340]]), 1000, 800)
        assert np.linalg.norm(far[1] - far[0]) > np.linalg.norm(near[1] - near[0])

    @pytest.mark.parametrize("points,message", [
        (((0.1, 0.9), (0.7, 0.4), (0.9, 0.9), (0.3, 0.4)), "in order"),     # crossing sides
        (((0.1, 0.9), (0.9, 0.9), (0.7, 0.4)), "four"),
        (((0.1, 0.9), (1.2, 0.9), (0.7, 0.4), (0.3, 0.4)), "inside the frame"),
        (((0.50, 0.50), (0.52, 0.50), (0.52, 0.52), (0.50, 0.52)), "too small"),
    ])
    def test_rejects_bad_points(self, points, message):
        with pytest.raises(CalibrationError, match=message):
            calib(points=points)

    def test_rejects_silly_sizes(self):
        with pytest.raises(CalibrationError):
            calib(width_m=0.1)
        with pytest.raises(CalibrationError):
            calib(length_m=2000)

    def test_round_trip(self):
        c = calib()
        assert GroundCalibration.from_dict(c.to_dict()) == c


class TestZoneCounting:
    def test_coverage_matches_the_quadrilateral_area(self):
        c = calib()
        weights = c.coverage(160, 120)
        assert weights.mean() == pytest.approx(c.image_area, rel=0.02)

    def test_counts_only_people_inside_the_zone(self):
        c = calib()
        density = np.zeros((120, 160))
        density[100, 80] = 7.0      # inside (bottom centre)
        density[10, 10] = 5.0       # outside (top left)
        assert c.count_in_zone(density) == pytest.approx(7.0)

    def test_uniform_density_scales_with_zone_share(self):
        c = calib()
        density = np.full((120, 160), 0.01)
        assert c.count_in_zone(density) == pytest.approx(density.sum() * c.image_area, rel=0.03)


class TestCalibratedScoring:
    def test_capacity_follows_the_zone(self):
        assert calibrated_capacity(80.0) == 400   # 80 m^2 at 5 people/m^2

    def test_physical_units_drive_the_score(self):
        cfg = DecisionConfig()
        # 3.4 people/m^2 shuffling at 0.3 m/s: HIGH on the physical scale
        s = build_crowd_state("G1", gate_capacity=999, crowd_count=272, velocity=0.3,
                              area_m2=80.0, config=cfg)
        assert s.velocity_unit == "m/s"
        assert s.density_per_m2 == pytest.approx(3.4)
        assert s.density_score == pytest.approx(272 / 400)
        assert s.congestion_level is CongestionLevel.HIGH

    def test_free_walking_counts_as_flowing(self):
        cfg = DecisionConfig()
        slow = build_crowd_state("G1", 0, crowd_count=200, velocity=0.1, area_m2=80.0, config=cfg)
        free = build_crowd_state("G1", 0, crowd_count=200, velocity=1.34, area_m2=80.0, config=cfg)
        assert free.congestion_score < slow.congestion_score

    def test_uncalibrated_gate_is_unchanged(self):
        s = build_crowd_state("G1", 200, crowd_count=100, velocity=30.0)
        assert s.velocity_unit == "px/s" and s.area_m2 is None and s.density_per_m2 is None
        assert s.density_score == 0.5


def _density(camera="cam", count_map=None) -> CSRNetResult:
    m = count_map if count_map is not None else np.zeros((120, 160))
    now = time.time()
    return CSRNetResult(camera_id=camera, frame_id=10, source_frame_index=10, video_timestamp=1.0,
                        wall_clock_timestamp=now, crowd_count=float(m.sum()), density_map=m,
                        density_map_width=160, density_map_height=120, input_width=1280, input_height=960,
                        inference_time_ms=50, model_name="csrnet", device="cpu", result_timestamp=now)


def _motion(camera="cam", speed=None) -> MotionResult:
    now = time.time()
    return MotionResult(camera_id=camera, frame_id=10, source_frame_index=10, video_timestamp=1.0,
                        wall_clock_timestamp=now, track_count=30, median_velocity=40.0, average_velocity=40.0,
                        dominant_direction_deg=90.0, flow_consistency=0.9, active_motion_ratio=0.8,
                        motion_confidence=0.8, inference_time_ms=30, model_name="pose", device="cpu",
                        result_timestamp=now, median_speed_ms=speed, zone_track_count=12)


class TestAdapter:
    def test_calibrated_fusion_uses_zone_count_and_metres(self):
        density = np.zeros((120, 160))
        density[100, 80] = 250.0
        density[10, 10] = 1000.0        # a crowd outside the zone must not count
        state = FusionAdapter().fuse(_density(count_map=density), _motion(speed=0.4), "G1", 200,
                                     calibration=calib())
        assert state.crowd_count == pytest.approx(250.0)
        assert state.velocity == pytest.approx(0.4)
        assert state.velocity_unit == "m/s"
        assert state.area_m2 == 80.0

    def test_zone_count_never_below_people_tracked_in_the_zone(self):
        """CSRNet under-counts large people near the camera; the detector does not."""
        density = np.zeros((120, 160))
        density[100, 80] = 5.0
        state = FusionAdapter().fuse(_density(count_map=density), _motion(speed=0.4), "G1", 200,
                                     calibration=calib())
        assert state.crowd_count == pytest.approx(12.0)   # zone_track_count

    def test_no_zone_speed_falls_back_to_density_only(self):
        state = FusionAdapter().fuse(_density(), _motion(speed=None), "G1", 200, calibration=calib())
        assert not state.has_motion
        assert "zone" in (state.motion_provenance or "")

    def test_uncalibrated_fusion_is_unchanged(self):
        state = FusionAdapter().fuse(_density(), _motion(speed=0.4), "G1", 200)
        assert state.velocity == pytest.approx(40.0) and state.velocity_unit == "px/s"


class TestRulesUseTheRightUnits:
    def _view(self, velocity_ms: float) -> GateView:
        state = build_crowd_state("G1", 0, crowd_count=380, velocity=velocity_ms, area_m2=80.0)
        return GateView(gate_id="G1", capacity=400, state=state)

    def test_nearly_stopped_dense_crowd_is_crush_risk(self):
        ids = {a.rule_id for a in generate_alerts({"G1": self._view(0.05)})}
        assert "CRUSH_RISK" in ids

    def test_walking_dense_crowd_is_not_crush_risk(self):
        ids = {a.rule_id for a in generate_alerts({"G1": self._view(1.0)})}
        assert "CRUSH_RISK" not in ids

    def test_calibrated_camera_never_asks_for_px_calibration(self):
        ids = {a.rule_id for a in generate_alerts({"G1": self._view(1.34)})}
        assert "VELOCITY_UNCALIBRATED" not in ids


class TestGroundSpeed:
    def test_walking_speed_in_metres_per_second(self):
        from crowdguard.models.posetrack.infer import PoseTrackInference
        inf = PoseTrackInference.__new__(PoseTrackInference)
        inf._ground, inf._previous_ground = {}, {}
        c = calib()
        inf.set_ground_calibration("cam", c)
        h_inv = np.linalg.inv(c.homography(1000, 800))

        def foot_box(gx, gy):
            x, y = apply_homography(h_inv, [[gx, gy]])[0]
            return (x - 10, y - 60, x + 10, y)

        packet = SimpleNamespace(frame=np.zeros((800, 1000, 3), np.uint8))
        people = [(2.0, 1.0), (5.0, 2.0), (8.0, 3.0), (4.0, 6.0)]
        first = [(foot_box(x, y), i, None) for i, (x, y) in enumerate(people)]
        # each person walks 0.5 m along the length between frames, at 2 fps -> 1.0 m/s
        second = [(foot_box(x, y + 0.5), i, None) for i, (x, y) in enumerate(people)]
        inf._ground_speed("cam", c, packet, first, 2.0)
        speed, in_zone = inf._ground_speed("cam", c, packet, second, 2.0)
        assert in_zone == 4
        assert speed == pytest.approx(1.0, rel=1e-3)


def test_new_session_keeps_the_calibration():
    """Start resets tracking history; it must not forget how the camera is calibrated."""
    from crowdguard.models.posetrack.infer import PoseTrackInference
    inf = PoseTrackInference.__new__(PoseTrackInference)
    inf._ground, inf._previous_ground = {}, {"cam": {1: (0.0, 0.0)}}
    inf._trackers, inf._previous_centers, inf._models = {}, {}, {}
    inf._last_frame_id, inf._last_loop_index = {}, {}
    inf._ground["cam"] = calib()
    inf.reset_tracking("cam")
    assert inf._ground.get("cam") == calib()
    assert "cam" not in inf._previous_ground
    inf.release_camera("cam")
    assert "cam" not in inf._ground


def test_store_persists(tmp_path):
    path = tmp_path / "cal.json"
    store = CalibrationStore(path)
    store.set("cam_g1", calib())
    assert CalibrationStore(path).get("cam_g1") == calib()
    assert json.loads(path.read_text())["cam_g1"]["area_m2"] == 80.0
    store.delete("cam_g1")
    assert CalibrationStore(path).get("cam_g1") is None
