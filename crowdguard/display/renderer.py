"""Playback renderer: shows every frame at the footage's frame rate.

Each camera gets its own render thread. On each tick, a thread asks its
camera's :class:`DisplayBuffer` which frame is due on the delayed session clock,
positions that frame's people from the analysis keyframes on either side of it,
draws the overlay, and publishes a JPEG.

**Why one thread per camera.** Three cameras at 27 fps is 81 frames a second to
draw and encode. Drawing costs about 2.5 ms, so a single thread manages that
on an idle machine. But every OpenCV call releases and re-acquires Python's GIL,
and on a loaded laptop each re-acquisition can wait behind the model threads.
Measured with a browser playing the streams, a single shared thread slowed to
~15 ms per frame and dropped playback to 10 fps. Separate threads overlap their
GIL-free work - resizing, encoding - across cores, and each only has to keep up
with its own camera.

Because frames are selected by scheduled time rather than by arrival, playback
speed is fixed by the footage. A faster GPU cannot make it run ahead and a slow
one cannot make it crawl: slow analysis only means boxes are held longer between
keyframes, and if analysis stalls entirely the boxes are withdrawn rather than
left frozen over a moving crowd. If drawing itself falls behind, the newest due
frame is always the one drawn, so a pane skips a frame rather than drifting late.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

import numpy as np

from crowdguard.config.schemas import DisplayConfig
from crowdguard.display.timeline import (
    DisplayBuffer,
    DisplayFrame,
    KeyframeHistory,
    people_for_frame,
)
from crowdguard.models.contracts import AnomalyResult
from crowdguard.web.overlay import encode_jpeg, render_display_frame, scale_for_display

logger = logging.getLogger(__name__)

# How long a thread waits before checking again when no frame is due yet.
_IDLE_TICK_S = 0.004

# How long a keyframe's boxes may be held with no newer analysis behind them.
_MAX_HOLD_S = 1.0

HeaderFn = Callable[[str], tuple[str | None, float | None, str | None]]
PublishFn = Callable[[str, bytes], None]


class _CameraPlayback:
    """One camera's buffer, keyframes, counters and render thread."""

    def __init__(self, camera_id: str, fps: float, config: DisplayConfig) -> None:
        self.camera_id = camera_id
        self.fps = fps if fps > 0 else 25.0
        self.frame_interval = 1.0 / self.fps
        capacity = max(4, int(math.ceil(self.fps * config.buffer_s)))
        self.buffer = DisplayBuffer(capacity=capacity)
        self.keyframes = KeyframeHistory(capacity=max(16, int(self.fps * 4)))
        self.max_hold_frames = max(1, int(self.fps * _MAX_HOLD_S))

        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.last_frame_id: int | None = None
        self.frames_rendered = 0
        self.frames_skipped = 0
        self.total_render_ms = 0.0
        self.first_render: float | None = None

    def reset(self) -> None:
        self.buffer.clear()
        self.keyframes.clear()
        self.last_frame_id = None
        self.frames_rendered = 0
        self.frames_skipped = 0
        self.total_render_ms = 0.0
        self.first_render = None


class PlaybackRenderer:
    """Draws and publishes every camera's frames at real-time speed."""

    def __init__(self, config: DisplayConfig, publish: PublishFn, header: HeaderFn) -> None:
        self.config = config
        self._publish = publish
        self._header = header

        self._cameras: dict[str, _CameraPlayback] = {}
        self._lock = threading.Lock()
        self._session_active = threading.Event()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, camera_id: str, fps: float) -> None:
        with self._lock:
            previous = self._cameras.get(camera_id)
            self._cameras[camera_id] = _CameraPlayback(camera_id, fps, self.config)
        if previous is not None:
            self._stop_thread(previous)

    def unregister(self, camera_id: str) -> None:
        with self._lock:
            playback = self._cameras.pop(camera_id, None)
        if playback is not None:
            self._stop_thread(playback)

    def _camera(self, camera_id: str) -> _CameraPlayback | None:
        with self._lock:
            return self._cameras.get(camera_id)

    # ------------------------------------------------------------------
    # Ingest (called from producer and pose-worker threads)
    # ------------------------------------------------------------------

    def ingest_frame(self, camera_id: str, frame_id: int, scheduled_time: float, frame: np.ndarray) -> None:
        """Queue one read frame for playback. Called for every frame, not just analysed ones."""
        playback = self._camera(camera_id)
        if playback is None:
            return
        image, scale = scale_for_display(frame, self.config.width)
        playback.buffer.push(DisplayFrame(frame_id, scheduled_time, image, scale))

    def ingest_keyframe(self, result: AnomalyResult) -> None:
        """Record an analysis result for boxes to be interpolated from."""
        playback = self._camera(result.camera_id)
        if playback is not None:
            playback.keyframes.append(result)

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def begin_session(self) -> None:
        """Clear buffered frames and keyframes, then start a render thread per camera."""
        with self._lock:
            cameras = list(self._cameras.values())
        for playback in cameras:
            self._stop_thread(playback)
            playback.reset()
        self._session_active.set()
        for playback in cameras:
            self._start_thread(playback)

    def end_session(self) -> None:
        self._session_active.clear()
        with self._lock:
            cameras = list(self._cameras.values())
        for playback in cameras:
            self._stop_thread(playback)

    @property
    def session_active(self) -> bool:
        return self._session_active.is_set()

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Kept for lifecycle symmetry; threads start with each session."""

    def stop(self, timeout: float = 2.0) -> None:
        self.end_session()

    def _start_thread(self, playback: _CameraPlayback) -> None:
        playback.stop_event.clear()
        playback.thread = threading.Thread(
            target=self._run_camera, args=(playback,),
            name=f"Playback-{playback.camera_id}", daemon=True,
        )
        playback.thread.start()

    @staticmethod
    def _stop_thread(playback: _CameraPlayback, timeout: float = 2.0) -> None:
        playback.stop_event.set()
        if playback.thread is not None and playback.thread is not threading.current_thread():
            playback.thread.join(timeout=timeout)
        playback.thread = None

    def _run_camera(self, playback: _CameraPlayback) -> None:
        delay = self.config.delay_s
        while not playback.stop_event.is_set() and self._session_active.is_set():
            frame = playback.buffer.frame_due(time.monotonic() - delay)

            if frame is None or frame.frame_id == playback.last_frame_id:
                playback.stop_event.wait(_IDLE_TICK_S)
                continue

            try:
                self._render(playback, frame)
            except Exception as exc:
                logger.warning("Playback render failed on %s: %s", playback.camera_id, exc)

            # Sleep until the next frame falls due rather than polling for it.
            next_due = frame.scheduled_time + delay + playback.frame_interval
            wait = next_due - time.monotonic()
            if wait > 0:
                playback.stop_event.wait(wait)

    def _render(self, playback: _CameraPlayback, frame: DisplayFrame) -> None:
        started = time.perf_counter()

        if playback.last_frame_id is not None and frame.frame_id > playback.last_frame_id + 1:
            playback.frames_skipped += frame.frame_id - playback.last_frame_id - 1
        playback.last_frame_id = frame.frame_id

        before, after = playback.keyframes.bracket(frame.frame_id)
        people = people_for_frame(frame.frame_id, before, after, playback.max_hold_frames)

        gate_id, crowd_count, level = self._header(playback.camera_id)
        annotated = render_display_frame(
            frame.image, frame.scale, people, before,
            gate_id=gate_id, crowd_count=crowd_count, congestion_level=level,
        )
        self._publish(playback.camera_id, encode_jpeg(annotated, self.config.jpeg_quality))

        playback.frames_rendered += 1
        playback.total_render_ms += (time.perf_counter() - started) * 1000.0
        if playback.first_render is None:
            playback.first_render = time.monotonic()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            cameras = list(self._cameras.values())
        result = {}
        for playback in cameras:
            elapsed = (time.monotonic() - playback.first_render) if playback.first_render else 0.0
            result[playback.camera_id] = {
                "source_fps": round(playback.fps, 2),
                "display_fps": round(playback.frames_rendered / elapsed, 2) if elapsed > 1 else 0.0,
                "frames_rendered": playback.frames_rendered,
                "frames_skipped": playback.frames_skipped,
                "avg_render_ms": round(
                    playback.total_render_ms / playback.frames_rendered, 2
                ) if playback.frames_rendered else 0.0,
                "buffered": len(playback.buffer),
                "thread_alive": playback.thread is not None and playback.thread.is_alive(),
            }
        return {
            "session_active": self.session_active,
            "delay_s": self.config.delay_s,
            "width": self.config.width,
            "cameras": result,
        }
