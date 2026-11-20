"""Playback timeline: which frame to show now, and where its boxes go.

The operator panes play every source frame at the footage's own frame rate,
independent of how quickly the models run. Two facts make that work:

* Every frame read is stamped with its **scheduled time** on the session clock
  (``t0 + k / fps``). Cameras that started together share ``t0``, so frame *k*
  of every gate falls due at the same instant, and playback speed is set by the
  footage, never by processing speed.
* Playback runs a fixed ``delay`` behind that clock. By the time a frame is due
  for display, the analysis keyframes on either side of it have usually
  finished, so its boxes can be interpolated between two real results instead of
  trailing the people they belong to.

Everything here is pure state with no threads, OpenCV drawing or model code, so
it is tested directly.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from crowdguard.models.contracts import AnomalyLabel, AnomalyResult


@dataclass(frozen=True)
class DisplayFrame:
    """One frame queued for playback, already scaled for display."""

    frame_id: int
    scheduled_time: float   # session-clock time this frame is due (monotonic seconds)
    image: np.ndarray       # BGR, scaled to the display width
    scale: float            # display width / original width


@dataclass(frozen=True)
class DisplayPerson:
    """A tracked person positioned for one displayed frame."""

    track_id: int
    bbox: tuple[float, float, float, float]   # original-frame pixels
    label: AnomalyLabel
    confirmed: bool
    was_vetoed: bool
    trail: tuple[tuple[float, float], ...]


class DisplayBuffer:
    """Ring buffer of recent display frames for one camera.

    Written by the producer thread on every frame read, read by the renderer.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        self._frames: deque[DisplayFrame] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, frame: DisplayFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def frame_due(self, playback_time: float) -> DisplayFrame | None:
        """Newest frame whose scheduled time is at or before ``playback_time``.

        ``playback_time`` is the session clock minus the display delay. Returns
        None when nothing has fallen due yet, e.g. just after Start.
        """
        with self._lock:
            # Frames arrive in schedule order, so scan back from the newest.
            for frame in reversed(self._frames):
                if frame.scheduled_time <= playback_time:
                    return frame
            return None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


class KeyframeHistory:
    """Recent analysis results for one camera, ordered by frame id.

    The pose branch analyses one frame in every ``anomaly_stride``. Those results
    are the keyframes that displayed frames are interpolated between.
    """

    def __init__(self, capacity: int = 120) -> None:
        self._keyframes: deque[AnomalyResult] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, result: AnomalyResult) -> None:
        with self._lock:
            # Results arrive in order from a single worker; guard anyway so a
            # late duplicate can never make the history non-monotonic.
            if self._keyframes and result.frame_id <= self._keyframes[-1].frame_id:
                return
            self._keyframes.append(result)

    def bracket(self, frame_id: int) -> tuple[AnomalyResult | None, AnomalyResult | None]:
        """The keyframes either side of ``frame_id``: (at-or-before, after)."""
        with self._lock:
            before: AnomalyResult | None = None
            after: AnomalyResult | None = None
            for keyframe in self._keyframes:
                if keyframe.frame_id <= frame_id:
                    before = keyframe
                else:
                    after = keyframe
                    break
            return before, after

    def latest(self) -> AnomalyResult | None:
        with self._lock:
            return self._keyframes[-1] if self._keyframes else None

    def clear(self) -> None:
        with self._lock:
            self._keyframes.clear()


def _lerp_box(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))  # type: ignore[return-value]


def people_for_frame(
    frame_id: int,
    before: AnomalyResult | None,
    after: AnomalyResult | None,
    max_hold_frames: int,
) -> list[DisplayPerson]:
    """Position every tracked person for one displayed frame.

    * Between two keyframes, each person present in both has their box linearly
      interpolated, so boxes glide with the video instead of jumping once per
      analysed frame.
    * A person only in the earlier keyframe keeps their last box until the next
      keyframe decides whether they are still there.
    * Labels always come from the earlier keyframe. A label is a decision about
      the past, and it would be wrong to show an alarm before the analysis that
      raised it.
    * With no later keyframe, the earlier one is held for at most
      ``max_hold_frames``, then nothing is drawn. Stale boxes on a moving crowd
      are worse than no boxes, and their absence signals that analysis has
      fallen behind.
    * Interpolation never crosses a clip loop: tracking restarts there, so a
      track id on either side may belong to different people.
    """
    if before is None:
        return []

    if after is None:
        if frame_id - before.frame_id > max_hold_frames:
            return []
        return [_as_display(p, p.bbox) for p in before.people]

    span = after.frame_id - before.frame_id
    same_pass = before.loop_index == after.loop_index
    if span <= 0 or not same_pass or frame_id == before.frame_id:
        return [_as_display(p, p.bbox) for p in before.people]

    t = min(1.0, max(0.0, (frame_id - before.frame_id) / span))
    later_boxes = {p.track_id: p.bbox for p in after.people}

    positioned: list[DisplayPerson] = []
    for person in before.people:
        target = later_boxes.get(person.track_id)
        box = _lerp_box(person.bbox, target, t) if target is not None else person.bbox
        positioned.append(_as_display(person, box))
    return positioned


def _as_display(person, bbox) -> DisplayPerson:
    return DisplayPerson(
        track_id=person.track_id,
        bbox=tuple(float(v) for v in bbox),
        label=person.label,
        confirmed=person.confirmed,
        was_vetoed=person.was_vetoed,
        trail=person.trail,
    )
