"""Frame producer: one read per camera, paced by a shared session clock.

Each camera is read exactly once, and every frame read goes three ways:

* the **display path** gets every frame, for real-time playback,
* the **anomaly path** keeps every ``anomaly_stride``-th frame for pose
  tracking, whose trajectory features are per-frame quantities, and
* the **density path** keeps every ``density_stride``-th frame for CSRNet,
  which only needs a periodic count.

Reading once and forking keeps decoding shared, and the single immutable
:class:`FramePacket` lets both model buffers hold the same instance without
copying the image again.

**Session clock.** A producer started with ``t0`` releases its *k*-th frame at
``t0 + k / fps``, measured from a fixed anchor rather than accumulated frame by
frame, so it cannot drift. Producers started with the same ``t0`` therefore play
their footage in lockstep from the same instant, which is what the Start button
relies on. That scheduled time travels with each frame to the display path, so
playback is timed by the footage itself, never by how fast frames happen to be
decoded or analysed.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from crowdguard.ingestion.frame_buffer import FrameBuffer
from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.ingestion.video_source import BaseFrameSource

logger = logging.getLogger(__name__)

# (camera_id, frame_id, scheduled_time, frame) for every frame read.
DisplaySink = Callable[[str, int, float, np.ndarray], None]

# If a producer falls this far behind its schedule - a stalled disk, a slow
# network stream - it re-anchors and drops the backlog instead of sprinting
# through it, which would burst frames at every consumer at once.
_RESYNC_AFTER_S = 1.0


class FrameProducer:
    """Reads one camera and fans its frames into display, anomaly and density paths.

    Runs on its own daemon thread. ``real_time_mode`` paces reads to the source
    frame rate; disabling it processes a file as fast as possible, which is only
    meaningful offline - the live display needs paced reading.
    """

    def __init__(
        self,
        camera_id: str,
        source: BaseFrameSource,
        density_buffer: FrameBuffer | None = None,
        anomaly_buffer: FrameBuffer | None = None,
        density_fps: float = 5.0,
        anomaly_stride: int = 1,
        real_time_mode: bool = True,
        loop_source: bool = False,
        display_sink: DisplaySink | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.density_buffer = density_buffer
        self.anomaly_buffer = anomaly_buffer
        self.density_fps = float(density_fps)
        self.anomaly_stride = max(1, int(anomaly_stride))
        self.real_time_mode = bool(real_time_mode)
        self.loop_source = bool(loop_source) and not source.is_live
        self.display_sink = display_sink

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._started = False
        self._complete = False

        # frame_id is monotonic for the producer's whole life, across sessions
        # and loops, so the synchroniser's "no future frames" rule always holds.
        self._frame_id = 0
        self._loop_index = 0
        self._anchor_request: float | None = None
        self._anchor: float = 0.0

        self.frames_read = 0
        self.frames_to_density = 0
        self.frames_to_anomaly = 0
        self.anomaly_frames_rejected = 0
        self.read_failures = 0
        self.resyncs = 0
        self._start_time: float | None = None

    # ------------------------------------------------------------------
    # Derived rates
    # ------------------------------------------------------------------

    @property
    def source_fps(self) -> float:
        fps = self.source.source_fps
        return fps if fps > 0 else 25.0

    @property
    def density_stride(self) -> int:
        """Frames skipped between CSRNet samples."""
        if self.density_fps <= 0:
            return 1
        return max(1, int(round(self.source_fps / self.density_fps)))

    @property
    def effective_anomaly_fps(self) -> float:
        return self.source_fps / self.anomaly_stride

    @property
    def started(self) -> bool:
        return self._started

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def read_fps(self) -> float:
        if self._start_time is None or self.frames_read == 0:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return self.frames_read / elapsed if elapsed > 1e-6 else 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, t0: float | None = None) -> None:
        """Start reading, with frame 0 released at monotonic time ``t0``.

        Pass the same ``t0`` to several producers to start them in lockstep. It
        should sit slightly in the future, so every source has time to open
        before its first frame falls due. Omitted, the producer starts at once.
        """
        if self.is_alive:
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._complete = False
        self._started = True
        self._anchor_request = t0
        self._loop_index = 0

        self.frames_read = 0
        self.frames_to_density = 0
        self.frames_to_anomaly = 0
        self.anomaly_frames_rejected = 0
        self.resyncs = 0

        self._thread = threading.Thread(target=self._run, name=f"Producer-{self.camera_id}", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop reading. The producer can be started again afterwards."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def shutdown(self) -> None:
        """Stop the thread and release the underlying source."""
        self.stop()
        self.source.close()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _wait_until(self, deadline: float) -> bool:
        """Sleep until ``deadline``, waking early on stop. Returns False if stopped."""
        remaining = deadline - time.monotonic()
        if remaining > 0:
            return not self._stop_event.wait(remaining)
        return not self._stop_event.is_set()

    def _run(self) -> None:
        try:
            self.source.open()
        except Exception as exc:
            logger.error("Camera %s failed to open: %s", self.camera_id, exc)
            self._complete = True
            return

        interval = 1.0 / self.source_fps if self.real_time_mode else 0.0
        self._anchor = self._anchor_request if self._anchor_request is not None else time.monotonic()
        self._start_time = self._anchor

        density_stride = self.density_stride
        anomaly_stride = self.anomaly_stride
        session_frames = 0  # frames released since this start, across loops

        logger.info(
            "Camera %s playing: %.1f fps source -> analysis 1 frame in %d (%.1f fps), "
            "density 1 in %d (%.2f fps)",
            self.camera_id, self.source_fps, anomaly_stride, self.effective_anomaly_fps,
            density_stride, self.source_fps / density_stride,
        )

        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    paused_at = time.monotonic()
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.01)
                    # Slide the schedule forward by the pause so playback resumes
                    # where it left off instead of racing to catch up.
                    self._anchor += time.monotonic() - paused_at
                    continue

                frame, source_index = self.source.read_next()

                if frame is None or source_index is None:
                    if self.loop_source and self.source.rewind():
                        self._loop_index += 1
                        logger.info("Camera %s looping (pass %d)", self.camera_id, self._loop_index + 1)
                        continue
                    logger.info("Camera %s reached end of stream after %d frames", self.camera_id, self.frames_read)
                    break

                # Pace every frame READ, before the stride checks. Pacing only the
                # frames sent to the models would let the loop race through the
                # rest, playing the clip stride-times too fast.
                scheduled = self._anchor + session_frames * interval
                if interval > 0:
                    lag = time.monotonic() - scheduled
                    if lag > _RESYNC_AFTER_S:
                        self._anchor = time.monotonic() - session_frames * interval
                        scheduled = self._anchor + session_frames * interval
                        self.resyncs += 1
                        logger.warning(
                            "Camera %s fell %.1fs behind schedule; resynchronised", self.camera_id, lag
                        )
                    elif not self._wait_until(scheduled):
                        break

                session_frames += 1
                self.frames_read += 1
                self._frame_id += 1

                if self.display_sink is not None:
                    try:
                        self.display_sink(self.camera_id, self._frame_id, scheduled, frame)
                    except Exception as exc:
                        logger.warning("Display sink failed on %s: %s", self.camera_id, exc)

                wants_density = self.density_buffer is not None and source_index % density_stride == 0
                wants_anomaly = self.anomaly_buffer is not None and source_index % anomaly_stride == 0
                if not (wants_density or wants_anomaly):
                    continue

                packet = self._build_packet(frame, source_index)

                if wants_anomaly:
                    if self.anomaly_buffer.put(packet):
                        self.frames_to_anomaly += 1
                    else:
                        # Refused by a newest-drop policy: the pose branch is
                        # saturated and track continuity is already broken, so
                        # surface it rather than let accuracy quietly rot.
                        self.anomaly_frames_rejected += 1
                        if self.anomaly_frames_rejected % 50 == 1:
                            logger.warning(
                                "Camera %s: pose branch saturated, %d frames refused. Raise "
                                "anomaly_stride or lower the pose model size.",
                                self.camera_id, self.anomaly_frames_rejected,
                            )

                if wants_density and self.density_buffer.put(packet):
                    self.frames_to_density += 1

        except Exception as exc:
            logger.exception("Camera %s producer error: %s", self.camera_id, exc)
            self.read_failures += 1
        finally:
            self._complete = True
            self.source.close()

    def _build_packet(self, frame: np.ndarray, source_index: int) -> FramePacket:
        timestamp_fn = getattr(self.source, "timestamp_for_index", None)
        video_timestamp = (
            timestamp_fn(source_index) if callable(timestamp_fn)
            else (source_index / self.source_fps if self.source_fps > 0 else float(source_index))
        )

        return FramePacket(
            camera_id=self.camera_id,
            frame_id=self._frame_id,
            source_frame_index=source_index,
            video_timestamp=video_timestamp,
            wall_clock_timestamp=time.time(),
            frame=frame,
            source_fps=self.source_fps,
            anomaly_stride=self.anomaly_stride,
            density_stride=self.density_stride,
            loop_index=self._loop_index,
        )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "source_type": self.source.source_type,
            "source_fps": round(self.source_fps, 2),
            "read_fps": round(self.read_fps, 2),
            "frames_read": self.frames_read,
            "frames_to_density": self.frames_to_density,
            "frames_to_anomaly": self.frames_to_anomaly,
            "anomaly_frames_rejected": self.anomaly_frames_rejected,
            "anomaly_stride": self.anomaly_stride,
            "density_stride": self.density_stride,
            "effective_anomaly_fps": round(self.effective_anomaly_fps, 2),
            "loop_index": self._loop_index,
            "resyncs": self.resyncs,
            "is_alive": self.is_alive,
            "is_paused": self.is_paused,
            "complete": self._complete,
            "frame_count": self.source.frame_count,
        }

    def __enter__(self) -> "FrameProducer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
