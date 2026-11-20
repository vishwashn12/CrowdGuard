"""Real-time playback: frame selection, keyframes and box interpolation.

Pure state, no GPU or threads needed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import ROOT  # noqa: F401  (ensures sys.path setup)

from crowdguard.config.schemas import DisplayConfig, PoseTrackModelConfig
from crowdguard.display.timeline import (
    DisplayBuffer,
    DisplayFrame,
    KeyframeHistory,
    people_for_frame,
)
from crowdguard.models.contracts import AnomalyLabel, AnomalyResult, PersonObservation


def frame(frame_id: int, scheduled: float) -> DisplayFrame:
    return DisplayFrame(frame_id, scheduled, np.zeros((4, 4, 3), dtype=np.uint8), 0.5)


def person(track_id: int, x: float, label: AnomalyLabel = AnomalyLabel.NORMAL) -> PersonObservation:
    return PersonObservation(
        track_id=track_id,
        bbox=(x, 100.0, x + 40.0, 200.0),
        label=label,
        frames_seen=20,
        speed_px_per_frame=2.0,
        direction_rad=0.0,
        cos_sim_to_crowd=1.0,
        confirmed=label.is_anomalous,
    )


def keyframe(frame_id: int, people: list[PersonObservation], loop_index: int = 0) -> AnomalyResult:
    return AnomalyResult(
        camera_id="cam",
        frame_id=frame_id,
        source_frame_index=frame_id,
        video_timestamp=frame_id / 27.0,
        wall_clock_timestamp=time.time(),
        people=tuple(people),
        tracked_count=len(people),
        loop_index=loop_index,
    )


class TestDisplayBuffer:
    def test_returns_newest_frame_already_due(self):
        buffer = DisplayBuffer(capacity=10)
        for i in range(5):
            buffer.push(frame(i, scheduled=10.0 + i * 0.1))

        assert buffer.frame_due(10.25).frame_id == 2
        assert buffer.frame_due(10.4).frame_id == 4

    def test_nothing_due_before_the_first_frame(self):
        """Right after Start, the delayed clock has not reached frame 0 yet."""
        buffer = DisplayBuffer(capacity=10)
        buffer.push(frame(1, scheduled=10.0))
        assert buffer.frame_due(9.9) is None

    def test_playback_speed_is_set_by_schedule_not_arrival(self):
        """Frames that arrive in a burst still play at their scheduled pace."""
        buffer = DisplayBuffer(capacity=30)
        # All pushed at once, as if decoding raced ahead.
        for i in range(27):
            buffer.push(frame(i, scheduled=100.0 + i / 27.0))

        # Half a second into playback, frame ~13 is due, not frame 26.
        due = buffer.frame_due(100.5)
        assert due.frame_id == 13

    def test_capacity_bounds_memory(self):
        buffer = DisplayBuffer(capacity=3)
        for i in range(10):
            buffer.push(frame(i, scheduled=float(i)))
        assert len(buffer) == 3
        assert buffer.frame_due(100.0).frame_id == 9

    def test_clear(self):
        buffer = DisplayBuffer(capacity=3)
        buffer.push(frame(1, 1.0))
        buffer.clear()
        assert buffer.frame_due(100.0) is None


class TestKeyframeHistory:
    def test_brackets_a_frame_between_keyframes(self):
        history = KeyframeHistory()
        for fid in (4, 8, 12):
            history.append(keyframe(fid, []))

        before, after = history.bracket(10)
        assert (before.frame_id, after.frame_id) == (8, 12)

    def test_exact_keyframe_is_its_own_before(self):
        history = KeyframeHistory()
        for fid in (4, 8):
            history.append(keyframe(fid, []))
        before, after = history.bracket(8)
        assert before.frame_id == 8
        assert after is None

    def test_frame_before_any_keyframe(self):
        history = KeyframeHistory()
        history.append(keyframe(10, []))
        before, after = history.bracket(5)
        assert before is None and after.frame_id == 10

    def test_ignores_out_of_order_results(self):
        history = KeyframeHistory()
        history.append(keyframe(8, []))
        history.append(keyframe(4, []))
        assert history.latest().frame_id == 8


class TestInterpolation:
    def test_box_moves_smoothly_between_keyframes(self):
        """Boxes glide with the video instead of jumping once per analysed frame."""
        a = keyframe(0, [person(1, x=0.0)])
        b = keyframe(4, [person(1, x=40.0)])

        positions = [people_for_frame(f, a, b, max_hold_frames=27)[0].bbox[0] for f in range(5)]
        assert positions == pytest.approx([0.0, 10.0, 20.0, 30.0, 40.0])

    def test_labels_come_from_the_earlier_keyframe(self):
        """An alarm is never shown before the analysis that raised it."""
        a = keyframe(0, [person(1, 0.0, AnomalyLabel.NORMAL)])
        b = keyframe(4, [person(1, 4.0, AnomalyLabel.SITTING)])
        assert people_for_frame(3, a, b, 27)[0].label is AnomalyLabel.NORMAL

    def test_person_missing_from_later_keyframe_holds_position(self):
        a = keyframe(0, [person(1, 0.0), person(2, 100.0)])
        b = keyframe(4, [person(1, 8.0)])
        people = {p.track_id: p for p in people_for_frame(2, a, b, 27)}
        assert people[2].bbox[0] == 100.0
        assert people[1].bbox[0] == pytest.approx(4.0)

    def test_held_keyframe_used_while_analysis_catches_up(self):
        a = keyframe(0, [person(1, 0.0)])
        assert len(people_for_frame(5, a, None, max_hold_frames=27)) == 1

    def test_boxes_withdrawn_once_analysis_stalls(self):
        """Stale boxes over a moving crowd are worse than none."""
        a = keyframe(0, [person(1, 0.0)])
        assert people_for_frame(28, a, None, max_hold_frames=27) == []

    def test_nothing_before_the_first_keyframe(self):
        assert people_for_frame(3, None, keyframe(8, [person(1, 0.0)]), 27) == []

    def test_never_interpolates_across_a_clip_loop(self):
        """Tracking restarts at a loop, so the same id may be someone else."""
        a = keyframe(100, [person(1, 0.0)], loop_index=0)
        b = keyframe(104, [person(1, 900.0)], loop_index=1)
        assert people_for_frame(102, a, b, 27)[0].bbox[0] == 0.0


class TestConfig:
    def test_display_buffer_must_outlast_the_delay(self):
        with pytest.raises(ValueError, match="must exceed delay_s"):
            DisplayConfig(delay_s=1.0, buffer_s=0.8)

    def test_default_display_settings(self):
        config = DisplayConfig()
        assert config.delay_s == 0.5
        assert config.buffer_s > config.delay_s

    def test_pose_defaults_are_the_chosen_model(self):
        config = PoseTrackModelConfig()
        assert config.model_path.endswith("yolov8m-pose.pt")
        assert config.imgsz == 1280
        assert config.fp16 is True
