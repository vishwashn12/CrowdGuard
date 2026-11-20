"""Worker thread for CSRNet density inference.

The density path is latency-sensitive rather than continuity-sensitive: only the
most recent count for a gate matters, and an old one is worthless. So this
worker drains its backlog and keeps only the newest frame per camera, which
keeps counts current under load instead of working through a stale queue.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Empty
from typing import Callable, Deque

from crowdguard.ingestion.frame_buffer import FrameBuffer
from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.models.contracts import CSRNetResult
from crowdguard.models.csrnet.infer import CSRNetInference

logger = logging.getLogger(__name__)


class CSRNetWorker:
    """Consumes frames, publishes :class:`CSRNetResult` objects."""

    def __init__(
        self,
        input_queue: FrameBuffer,
        engine: CSRNetInference,
        result_callback: Callable[[CSRNetResult], None] | None = None,
        output_capacity: int = 30,
    ) -> None:
        self.input_queue = input_queue
        self.engine = engine
        self.result_callback = result_callback

        self._output: Deque[CSRNetResult] = deque(maxlen=output_capacity)
        self._latest: dict[str, CSRNetResult] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.frames_processed = 0
        self.frames_skipped = 0
        self.errors = 0
        self.total_inference_ms = 0.0

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

    def latest_result(self, camera_id: str) -> CSRNetResult | None:
        with self._lock:
            return self._latest.get(camera_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="CSRNetWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _drain_to_latest(self) -> list[FramePacket]:
        """Collect pending frames, keeping only the newest per camera."""
        drained: list[FramePacket] = []

        try:
            drained.append(self.input_queue.get(block=True, timeout=0.1))
            while True:
                drained.append(self.input_queue.get(block=False))
        except Empty:
            pass

        if not drained:
            return []

        newest: dict[str, FramePacket] = {}
        for packet in drained:
            current = newest.get(packet.camera_id)
            if current is None or packet.frame_id > current.frame_id:
                newest[packet.camera_id] = packet

        skipped = len(drained) - len(newest)
        if skipped > 0:
            with self._lock:
                self.frames_skipped += skipped

        return list(newest.values())

    def _run(self) -> None:
        while not self._stop_event.is_set():
            packets = self._drain_to_latest()
            if not packets:
                continue

            for packet in packets:
                if self._stop_event.is_set():
                    return

                try:
                    result = self.engine.estimate(packet)
                except Exception as exc:
                    with self._lock:
                        self.errors += 1
                    logger.error(
                        "CSRNet inference failed on %s: %s", packet.describe(), exc
                    )
                    continue

                with self._lock:
                    self._output.append(result)
                    self._latest[packet.camera_id] = result
                    self.frames_processed += 1
                    self.total_inference_ms += result.inference_time_ms

                if self.result_callback is not None:
                    try:
                        self.result_callback(result)
                    except Exception as exc:
                        logger.warning("CSRNet result callback error: %s", exc)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Zero the latency and throughput counters."""
        with self._lock:
            self.frames_processed = 0
            self.frames_skipped = 0
            self.errors = 0
            self.total_inference_ms = 0.0

    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_processed": self.frames_processed,
                "frames_skipped": self.frames_skipped,
                "errors": self.errors,
                "avg_inference_ms": round(
                    self.total_inference_ms / self.frames_processed, 1
                ) if self.frames_processed else 0.0,
                "queue_size": len(self.input_queue),
                "is_alive": self.is_alive,
            }
