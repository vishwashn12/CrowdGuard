"""Anomaly branch: feature vector, pose geometry, tracking safeguards.

These tests need no GPU and no model weights: the features and the tracker are
pure state, which is exactly why they live outside the inference class.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ROOT  # noqa: F401  (ensures sys.path setup)

from crowdguard.config.schemas import AnomalyConfig
from crowdguard.models.contracts import AnomalyLabel
from crowdguard.models.posetrack.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    build_feature_vector,
    compute_pose_features,
)
from crowdguard.models.posetrack.tracker import CameraAnomalyTracker, PersonTrack


def standing_keypoints(scale: float = 1.0) -> np.ndarray:
    """COCO-17 keypoints for an upright person, confidence 1.0.

    Legs nearly straight: hip -> knee -> ankle is close to a line.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[0] = (100, 10 * scale, 1.0)    # nose
    kp[5] = (90, 40 * scale, 1.0)     # left shoulder
    kp[6] = (110, 40 * scale, 1.0)    # right shoulder
    kp[11] = (92, 100 * scale, 1.0)   # left hip
    kp[12] = (108, 100 * scale, 1.0)  # right hip
    kp[13] = (92, 150 * scale, 1.0)   # left knee
    kp[14] = (108, 150 * scale, 1.0)  # right knee
    kp[15] = (92, 200 * scale, 1.0)   # left ankle
    kp[16] = (108, 200 * scale, 1.0)  # right ankle
    return kp


def sitting_keypoints() -> np.ndarray:
    """COCO-17 keypoints for a seated person: knees sharply bent."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[0] = (100, 60, 1.0)    # nose, lower down
    kp[5] = (90, 90, 1.0)
    kp[6] = (110, 90, 1.0)
    kp[11] = (92, 140, 1.0)   # hips
    kp[12] = (108, 140, 1.0)
    kp[13] = (140, 145, 1.0)  # knees thrust forward
    kp[14] = (150, 145, 1.0)
    kp[15] = (140, 195, 1.0)  # ankles below knees
    kp[16] = (150, 195, 1.0)
    return kp


class TestFeatureContract:
    def test_feature_count_matches_names(self):
        assert FEATURE_COUNT == len(FEATURE_NAMES) == 10

    def test_feature_order_is_the_trained_order(self):
        """This order is baked into best_model.pkl and must not drift."""
        assert FEATURE_NAMES == (
            "displacement",
            "direction_sin",
            "direction_cos",
            "cos_sim",
            "straightness",
            "aspect_ratio",
            "pos_y_norm",
            "torso_height_ratio",
            "head_to_hip_ratio",
            "knee_bend_angle",
        )

    def test_vector_has_the_right_shape_and_dtype(self):
        vector = build_feature_vector(
            positions=[(0.0, 0.0), (10.0, 0.0)],
            step_speeds=[10.0],
            direction_rad=0.0,
            bbox=(0.0, 0.0, 30.0, 90.0),
            keypoints=standing_keypoints(),
            crowd_direction_rad=0.0,
            crowd_direction_valid=True,
            frame_height=480,
        )
        assert vector.shape == (FEATURE_COUNT,)
        assert vector.dtype == np.float32
        assert np.all(np.isfinite(vector))


class TestPoseFeatures:
    def test_standing_knee_angle_is_near_straight(self):
        _, _, knee = compute_pose_features(standing_keypoints(), bbox_height=200.0)
        assert knee > 150.0

    def test_sitting_knee_angle_is_sharply_bent(self):
        _, _, knee = compute_pose_features(sitting_keypoints(), bbox_height=160.0)
        assert knee < 130.0

    def test_sitting_bends_more_than_standing(self):
        _, _, standing = compute_pose_features(standing_keypoints(), bbox_height=200.0)
        _, _, sitting = compute_pose_features(sitting_keypoints(), bbox_height=160.0)
        assert sitting < standing

    def test_missing_keypoints_fall_back_to_straight_legs(self):
        """No visible legs must not be read as a bent knee, i.e. as sitting."""
        torso, head, knee = compute_pose_features(None, bbox_height=100.0)
        assert (torso, head, knee) == (0.0, 0.0, 180.0)

    def test_low_confidence_keypoints_are_ignored(self):
        kp = standing_keypoints()
        kp[:, 2] = 0.1  # everything below the confidence gates
        torso, head, knee = compute_pose_features(kp, bbox_height=200.0)
        assert (torso, head, knee) == (0.0, 0.0, 180.0)

    def test_one_visible_side_is_enough(self):
        kp = standing_keypoints()
        kp[6, 2] = 0.0   # right shoulder hidden
        kp[12, 2] = 0.0  # right hip hidden
        torso, _, _ = compute_pose_features(kp, bbox_height=200.0)
        assert torso > 0.0

    def test_zero_height_box_is_handled(self):
        torso, head, knee = compute_pose_features(standing_keypoints(), bbox_height=0.0)
        assert (torso, head, knee) == (0.0, 0.0, 180.0)

    def test_ratios_are_scale_invariant(self):
        """Ratios normalise by box height, so distance from camera shouldn't matter."""
        near_torso, near_head, _ = compute_pose_features(
            standing_keypoints(scale=1.0), bbox_height=200.0
        )
        far_torso, far_head, _ = compute_pose_features(
            standing_keypoints(scale=2.0), bbox_height=400.0
        )
        assert near_torso == pytest.approx(far_torso, abs=0.02)
        assert near_head == pytest.approx(far_head, abs=0.02)


class TestMotionFeatures:
    def test_displacement_is_end_to_end_distance(self):
        vector = build_feature_vector(
            positions=[(0.0, 0.0), (3.0, 4.0)],
            step_speeds=[5.0],
            direction_rad=0.0,
            bbox=(0.0, 0.0, 10.0, 20.0),
            keypoints=None,
            crowd_direction_rad=0.0,
            crowd_direction_valid=True,
            frame_height=100,
        )
        assert vector[0] == pytest.approx(5.0)

    def test_straight_path_has_straightness_near_one(self):
        positions = [(float(x), 0.0) for x in range(0, 50, 10)]
        vector = build_feature_vector(
            positions=positions,
            step_speeds=[10.0] * 4,
            direction_rad=0.0,
            bbox=(0.0, 0.0, 10.0, 20.0),
            keypoints=None,
            crowd_direction_rad=0.0,
            crowd_direction_valid=True,
            frame_height=100,
        )
        assert vector[4] == pytest.approx(1.0, abs=0.01)

    def test_pacing_back_and_forth_has_low_straightness(self):
        positions = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0), (10.0, 0.0), (0.0, 0.0)]
        vector = build_feature_vector(
            positions=positions,
            step_speeds=[10.0] * 4,
            direction_rad=0.0,
            bbox=(0.0, 0.0, 10.0, 20.0),
            keypoints=None,
            crowd_direction_rad=0.0,
            crowd_direction_valid=True,
            frame_height=100,
        )
        assert vector[4] < 0.2

    def test_cos_sim_is_one_when_aligned_with_the_crowd(self):
        vector = build_feature_vector(
            positions=[(0.0, 0.0), (10.0, 0.0)],
            step_speeds=[10.0],
            direction_rad=0.0,
            bbox=(0.0, 0.0, 10.0, 20.0),
            keypoints=None,
            crowd_direction_rad=0.0,
            crowd_direction_valid=True,
            frame_height=100,
        )
        assert vector[3] == pytest.approx(1.0)

    def test_cos_sim_is_minus_one_against_the_crowd(self):
        vector = build_feature_vector(
            positions=[(0.0, 0.0), (-10.0, 0.0)],
            step_speeds=[10.0],
            direction_rad=np.pi,
            bbox=(0.0, 0.0, 10.0, 20.0),
            keypoints=None,
            crowd_direction_rad=0.0,
            crowd_direction_valid=True,
            frame_height=100,
        )
        assert vector[3] == pytest.approx(-1.0)

    def test_cos_sim_is_zero_when_no_flow_is_established(self):
        """Unknown alignment must read as neutral, never as compliant."""
        vector = build_feature_vector(
            positions=[(0.0, 0.0), (10.0, 0.0)],
            step_speeds=[10.0],
            direction_rad=0.0,
            bbox=(0.0, 0.0, 10.0, 20.0),
            keypoints=None,
            crowd_direction_rad=0.0,
            crowd_direction_valid=False,
            frame_height=100,
        )
        assert vector[3] == 0.0

    def test_sitting_box_is_wider_than_standing_box(self):
        standing = build_feature_vector(
            positions=[(0.0, 0.0)], step_speeds=[], direction_rad=0.0,
            bbox=(0.0, 0.0, 30.0, 90.0), keypoints=None,
            crowd_direction_rad=0.0, crowd_direction_valid=False, frame_height=480,
        )
        sitting = build_feature_vector(
            positions=[(0.0, 0.0)], step_speeds=[], direction_rad=0.0,
            bbox=(0.0, 0.0, 60.0, 70.0), keypoints=None,
            crowd_direction_rad=0.0, crowd_direction_valid=False, frame_height=480,
        )
        assert sitting[5] > standing[5]

    def test_single_position_gives_zero_displacement(self):
        vector = build_feature_vector(
            positions=[(5.0, 5.0)], step_speeds=[], direction_rad=0.0,
            bbox=(0.0, 0.0, 10.0, 20.0), keypoints=None,
            crowd_direction_rad=0.0, crowd_direction_valid=False, frame_height=100,
        )
        assert vector[0] == 0.0
        assert vector[4] == 0.0


class TestTracking:
    def test_new_detection_creates_a_track(self, anomaly_config: AnomalyConfig):
        tracker = CameraAnomalyTracker("cam", anomaly_config)
        tracker.update_tracks([((10.0, 10.0, 40.0, 100.0), 1, None)])
        assert 1 in tracker.tracks
        assert tracker.tracks[1].frames_seen == 1

    def test_track_accumulates_history(self, anomaly_config: AnomalyConfig):
        tracker = CameraAnomalyTracker("cam", anomaly_config)
        for step in range(5):
            x = 10.0 + step * 5
            tracker.update_tracks([((x, 10.0, x + 30, 100.0), 1, None)])
        track = tracker.tracks[1]
        assert track.frames_seen == 5
        assert len(track.positions) == 5
        assert track.speed > 0

    def test_vanished_track_is_buffered_then_evicted(self, anomaly_config: AnomalyConfig):
        config = anomaly_config.model_copy(update={"lost_track_buffer_frames": 3})
        tracker = CameraAnomalyTracker("cam", config)

        tracker.update_tracks([((10.0, 10.0, 40.0, 100.0), 1, None)])
        for _ in range(3):
            tracker.update_tracks([])
        # Still buffered: a re-appearance here would be stitched.
        assert tracker.tracks == {}

        tracker.update_tracks([])
        # Past the buffer, so the same box starts a fresh track.
        tracker.update_tracks([((10.0, 10.0, 40.0, 100.0), 9, None)])
        assert tracker.tracks[9].frames_seen == 1

    def test_occluded_track_is_restitched_and_keeps_history(self, anomaly_config: AnomalyConfig):
        tracker = CameraAnomalyTracker("cam", anomaly_config)
        bbox = (10.0, 10.0, 40.0, 100.0)

        # Build a moving track so the stationary guard does not apply.
        for step in range(6):
            x = 10.0 + step * 6
            tracker.update_tracks([((x, 10.0, x + 30, 100.0), 1, None)])
        frames_before = tracker.tracks[1].frames_seen

        tracker.update_tracks([])  # occluded for a frame

        last_bbox = (40.0, 10.0, 70.0, 100.0)
        tracker.update_tracks([(last_bbox, 2, None)])  # reappears under a new id

        assert tracker.restitch_count == 1
        assert tracker.tracks[2].frames_seen == frames_before + 1

    def test_walker_cannot_steal_a_stationary_tracks_history(self, anomaly_config: AnomalyConfig):
        """The core ID-theft guard.

        A seated person's track must not be adopted by someone walking past,
        because the SITTING label would travel with it across the scene.
        """
        tracker = CameraAnomalyTracker("cam", anomaly_config)
        seated_bbox = (100.0, 100.0, 160.0, 170.0)

        # A track that barely moves: a candidate seated person.
        for _ in range(12):
            tracker.update_tracks([(seated_bbox, 1, None)])
        seated = tracker.tracks[1]
        assert seated.stationary_fraction > 0.5

        tracker.update_tracks([])  # seated person momentarily occluded

        # A pedestrian whose box overlaps heavily, but whose centre is far away.
        walker_bbox = (140.0, 100.0, 200.0, 170.0)
        tracker.update_tracks([(walker_bbox, 2, None)])

        assert tracker.restitch_count == 0
        assert tracker.tracks[2].frames_seen == 1

    def test_crowd_direction_needs_enough_moving_people(self, anomaly_config: AnomalyConfig):
        config = anomaly_config.model_copy(update={"min_crowd_people": 3})
        tracker = CameraAnomalyTracker("cam", config)

        # Two movers only: not enough to define a flow.
        for step in range(5):
            offset = step * 6
            tracker.update_tracks([
                ((offset, 10.0, offset + 30, 100.0), 1, None),
                ((offset, 120.0, offset + 30, 210.0), 2, None),
            ])
        _, valid = tracker.update_crowd_direction()
        assert valid is False

    def test_crowd_direction_points_along_the_flow(self, anomaly_config: AnomalyConfig):
        tracker = CameraAnomalyTracker("cam", anomaly_config)

        # Four people all walking in +x.
        for step in range(8):
            offset = step * 8
            tracker.update_tracks([
                ((offset, 10.0 + 40 * i, offset + 30, 100.0 + 40 * i), i + 1, None)
                for i in range(4)
            ])

        direction, valid = tracker.update_crowd_direction()
        assert valid is True
        assert np.cos(direction) == pytest.approx(1.0, abs=0.1)


class TestLabelling:
    def test_anomaly_needs_repeated_predictions(self, anomaly_config: AnomalyConfig):
        config = anomaly_config.model_copy(update={"confirm_frames": 3})
        tracker = CameraAnomalyTracker("cam", config)
        track = PersonTrack(history_length=60)
        track.bbox = (0.0, 0.0, 30.0, 90.0)

        assert not tracker.apply_prediction(track, AnomalyLabel.OPPOSITE)
        assert not tracker.apply_prediction(track, AnomalyLabel.OPPOSITE)
        assert tracker.apply_prediction(track, AnomalyLabel.OPPOSITE)
        assert track.label is AnomalyLabel.OPPOSITE

    def test_return_to_normal_is_immediate(self, anomaly_config: AnomalyConfig):
        """Clearing a stale alarm must never be delayed by smoothing."""
        config = anomaly_config.model_copy(update={"confirm_frames": 2})
        tracker = CameraAnomalyTracker("cam", config)
        track = PersonTrack(history_length=60)
        track.bbox = (0.0, 0.0, 30.0, 90.0)

        tracker.apply_prediction(track, AnomalyLabel.OPPOSITE)
        tracker.apply_prediction(track, AnomalyLabel.OPPOSITE)
        assert track.label is AnomalyLabel.OPPOSITE

        tracker.apply_prediction(track, AnomalyLabel.NORMAL)
        assert track.label is AnomalyLabel.NORMAL

    def test_fast_moving_sitting_label_is_vetoed(self, anomaly_config: AnomalyConfig):
        """The physics backstop: a person who is moving is not sitting."""
        config = anomaly_config.model_copy(
            update={"confirm_frames": 1, "sitting_veto_speed_px": 3.0}
        )
        tracker = CameraAnomalyTracker("cam", config)
        track = PersonTrack(history_length=60)
        track.bbox = (0.0, 0.0, 30.0, 90.0)
        track.step_speeds.extend([9.0, 11.0])  # clearly walking

        confirmed = tracker.apply_prediction(track, AnomalyLabel.SITTING)

        assert confirmed is False
        assert track.label is AnomalyLabel.NORMAL
        assert track.was_vetoed is True
        assert tracker.veto_count == 1

    def test_stationary_sitting_label_survives(self, anomaly_config: AnomalyConfig):
        config = anomaly_config.model_copy(
            update={"confirm_frames": 1, "sitting_veto_speed_px": 3.0}
        )
        tracker = CameraAnomalyTracker("cam", config)
        track = PersonTrack(history_length=60)
        track.bbox = (0.0, 0.0, 60.0, 70.0)
        track.step_speeds.extend([0.2, 0.1])

        assert tracker.apply_prediction(track, AnomalyLabel.SITTING)
        assert track.label is AnomalyLabel.SITTING
        assert track.was_vetoed is False

    def test_veto_needs_two_step_samples(self, anomaly_config: AnomalyConfig):
        """With one sample there is no reliable speed, so no veto."""
        config = anomaly_config.model_copy(update={"confirm_frames": 1})
        tracker = CameraAnomalyTracker("cam", config)
        track = PersonTrack(history_length=60)
        track.bbox = (0.0, 0.0, 60.0, 70.0)
        track.step_speeds.append(50.0)

        assert track.recent_step_speed == 0.0
        assert tracker.apply_prediction(track, AnomalyLabel.SITTING)

    def test_rule_fallback_flags_sustained_counterflow(self, anomaly_config: AnomalyConfig):
        config = anomaly_config.model_copy(update={"confirm_frames": 2})
        tracker = CameraAnomalyTracker("cam", config)
        tracker.crowd_direction = 0.0
        tracker.crowd_direction_valid = True

        track = PersonTrack(history_length=60)
        track.frames_seen = 10
        track.speed = 10.0
        track.direction = np.pi  # straight against the flow

        tracker.apply_rule_fallback(track)
        tracker.apply_rule_fallback(track)
        assert track.label is AnomalyLabel.OPPOSITE

    def test_rule_fallback_ignores_stationary_people(self, anomaly_config: AnomalyConfig):
        tracker = CameraAnomalyTracker("cam", anomaly_config)
        tracker.crowd_direction = 0.0
        tracker.crowd_direction_valid = True

        track = PersonTrack(history_length=60)
        track.frames_seen = 10
        track.speed = 0.1
        track.direction = np.pi

        for _ in range(10):
            tracker.apply_rule_fallback(track)
        assert track.label is AnomalyLabel.NORMAL

    def test_only_mature_tracks_are_classifiable(self, anomaly_config: AnomalyConfig):
        config = anomaly_config.model_copy(update={"min_track_frames": 5})
        tracker = CameraAnomalyTracker("cam", config)

        for _ in range(3):
            tracker.update_tracks([((10.0, 10.0, 40.0, 100.0), 1, None)])
        assert tracker.classifiable_tracks() == []

        for _ in range(3):
            tracker.update_tracks([((10.0, 10.0, 40.0, 100.0), 1, None)])
        assert [tid for tid, _ in tracker.classifiable_tracks()] == [1]


class TestStrideScaling:
    def test_stride_one_is_unchanged(self):
        config = AnomalyConfig()
        assert config.scaled_for_stride(1) is config

    def test_stride_scales_distances_up_and_windows_down(self):
        config = AnomalyConfig(
            direction_window=15, min_speed_px=2.0, sitting_veto_speed_px=3.0
        )
        scaled = config.scaled_for_stride(3)

        assert scaled.min_speed_px == pytest.approx(6.0)
        assert scaled.sitting_veto_speed_px == pytest.approx(9.0)
        assert scaled.direction_window == 5

    def test_window_never_drops_below_two(self):
        config = AnomalyConfig(direction_window=4)
        assert config.scaled_for_stride(100).direction_window == 2
