"""Worker thread for pose tracking, crowd motion, and anomaly labelling.

Frames are processed in order and never reordered: the 10-D feature vector is
built from a trajectory, and shuffling or cherry-picking frames would corrupt
``displacement``, ``straightness`` and ``cos_sim``.

**Bounded latency.** A frame that has waited longer than ``max_latency_s`` is
skipped rather than analysed. Without this bound, any sustained slowdown - a busy
machine, other GPU work - lets a backlog build until analysis runs seconds
behind live. Then every box is too
stale to draw, and every alert describes a crowd that has already moved on.
Skipping keeps the results that *are* produced current. The skipped frames show
up as continuity breaks, which drive the ``TRACKING_DEGRADED`` advisory, so the
degradation stays visible instead of silent.

Health figures are reported over a recent window rather than the whole session,
so a brief slowdown at start-up does not leave a fault showing for ever after.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Empty
from typing import Callable, Deque

from crowdguard.ingestion.frame_buffer import FrameBuffer
from crowdguard.models.contracts import AnomalyResult, MotionResult
from crowdguard.models.posetrack.infer import PoseTrackInference

logger = logging.getLogger(__name__)

# Span of the rolling window behind the health figures.
_HEALTH_WINDOW_S = 10.0


class PoseTrackWorker:
    """Consumes frames in order, publishing motion and anomaly results."""

    def __init__(
        self,
        input_queue: FrameBuffer,
        engine: PoseTrackInference,
        motion_callback: Callable[[MotionResult], None] | None = None,
        anomaly_callback: Callable[[AnomalyResult], None] | None = None,
        output_capacity: int = 30,
        max_latency_s: float | None = None,
    ) -> None:
        self.input_queue = input_queue
        self.engine = engine
        self.motion_callback = motion_callback
        self.anomaly_callback = anomaly_callback
        self.max_latency_s = max_latency_s

        self._motion_output: Deque[MotionResult] = deque(maxlen=output_capacity)
        self._latest_motion: dict[str, MotionResult] = {}
        self._latest_anomaly: dict[str, AnomalyResult] = {}

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.frames_processed = 0
        self.frames_skipped_late = 0
        self.continuity_breaks = 0
        self.errors = 0
        self.total_inference_ms = 0.0

        # (timestamp, outcome) with outcome in {"ok", "break", "late"}
        self._recent: Deque[tuple[float, str]] = deque()
        self._recent_latency: Deque[tuple[float, float]] = deque()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def avg_inference_ms(self) -> float:
        with self._lock:
            if self.frames_processed == 0:
                return 0.0
            return self.total_inference_ms / self.frames_processed

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_motion(self, camera_id: str) -> MotionResult | None:
        with self._lock:
            return self._latest_motion.get(camera_id)

    def latest_anomaly(self, camera_id: str) -> AnomalyResult | None:
        with self._lock:
            return self._latest_anomaly.get(camera_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="PoseTrackWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _record(self, outcome: str, latency_s: float | None = None) -> None:
        """Note an outcome in the rolling health window. Caller holds the lock."""
        now = time.monotonic()
        self._recent.append((now, outcome))
        if latency_s is not None:
            self._recent_latency.append((now, latency_s))
        cutoff = now - _HEALTH_WINDOW_S
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()
        while self._recent_latency and self._recent_latency[0][0] < cutoff:
            self._recent_latency.popleft()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self.input_queue.get(block=True, timeout=0.1)
            except Empty:
                continue

            waited = time.time() - packet.wall_clock_timestamp
            if self.max_latency_s is not None and waited > self.max_latency_s:
                with self._lock:
                    self.frames_skipped_late += 1
                    self._record("late")
                    skipped = self.frames_skipped_late
                if skipped % 100 == 1:
                    logger.warning(
                        "Pose analysis is falling behind (frame waited %.2fs); skipping late "
                        "frames to stay current. %d skipped so far. If this persists, raise "
                        "anomaly_stride or reduce other load on the machine.",
                        waited, skipped,
                    )
                continue

            try:
                motion, anomaly = self.engine.infer(packet)
            except Exception as exc:
                with self._lock:
                    self.errors += 1
                logger.error("Pose inference failed on %s: %s", packet.describe(), exc)
                continue

            latency = time.time() - packet.wall_clock_timestamp
            with self._lock:
                self._motion_output.append(motion)
                self._latest_motion[packet.camera_id] = motion
                self._latest_anomaly[packet.camera_id] = anomaly
                self.frames_processed += 1
                self.total_inference_ms += motion.inference_time_ms
                if not anomaly.continuity_ok:
                    self.continuity_breaks += 1
                self._record("ok" if anomaly.continuity_ok else "break", latency)

            if self.motion_callback is not None:
                try:
                    self.motion_callback(motion)
                except Exception as exc:
                    logger.warning("Motion result callback error: %s", exc)

            if self.anomaly_callback is not None:
                try:
                    self.anomaly_callback(anomaly)
                except Exception as exc:
                    logger.warning("Anomaly result callback error: %s", exc)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Zero the latency, continuity and throughput counters."""
        with self._lock:
            self.frames_processed = 0
            self.frames_skipped_late = 0
            self.continuity_breaks = 0
            self.errors = 0
            self.total_inference_ms = 0.0
            self._recent.clear()
            self._recent_latency.clear()

    def stats(self) -> dict:
        with self._lock:
            processed = self.frames_processed
            recent = [outcome for _, outcome in self._recent]
            total = len(recent)
            bad = sum(1 for o in recent if o != "ok")
            late = sum(1 for o in recent if o == "late")
            latencies = sorted(v for _, v in self._recent_latency)
            return {
                "frames_processed": processed,
                "frames_skipped_late": self.frames_skipped_late,
                "continuity_breaks": self.continuity_breaks,
                # Recent window: drives the TRACKING_DEGRADED advisory.
                "continuity_break_ratio": round(bad / total, 3) if total else 0.0,
                "late_skip_ratio": round(late / total, 3) if total else 0.0,
                "latency_p50_ms": round(1000 * latencies[len(latencies) // 2]) if latencies else 0,
                "latency_max_ms": round(1000 * latencies[-1]) if latencies else 0,
                "max_latency_ms": round(1000 * self.max_latency_s) if self.max_latency_s else None,
                "errors": self.errors,
                "avg_inference_ms": round(self.total_inference_ms / processed, 1) if processed else 0.0,
                "queue_size": len(self.input_queue),
                "is_alive": self.is_alive,
            }
