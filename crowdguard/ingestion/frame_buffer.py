"""Thread-safe bounded frame buffer for producer/consumer ingestion."""

from __future__ import annotations

import threading
import time
from collections import deque
from queue import Empty
from typing import Deque

from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.utils.exceptions import FrameBufferError


class FrameBuffer:
    """Bounded queue with a configurable overflow policy and counters.

    Drop policies:

    * ``oldest``      - evict the head to admit the newest frame. Right for the
      density path, where only the latest estimate matters.
    * ``newest`` / ``drop_latest`` - refuse the incoming frame and keep the
      backlog. Right for the anomaly path, where trajectory continuity matters
      more than recency: dropping from the middle of a track corrupts the
      direction window, whereas trailing slightly behind does not.
    """

    def __init__(self, capacity: int = 30, drop_policy: str = "oldest", name: str = "buffer") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if drop_policy not in {"oldest", "newest", "drop_latest"}:
            raise ValueError("drop_policy must be one of: oldest, newest, drop_latest")

        self.capacity = int(capacity)
        self.drop_policy = drop_policy
        self.name = name

        self._queue: Deque[FramePacket] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

        self.frames_offered = 0
        self.frames_accepted = 0
        self.frames_consumed = 0
        self.frames_dropped = 0
        self.maximum_size = 0
        self._first_put_time: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_size(self) -> int:
        return len(self._queue)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def throughput_fps(self) -> float:
        """Average accepted frames per second since the first enqueue."""
        if self._first_put_time is None or self.frames_accepted == 0:
            return 0.0
        elapsed = max(time.monotonic() - self._first_put_time, 1e-6)
        return self.frames_accepted / elapsed

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def put(self, packet: FramePacket) -> bool:
        """Offer a packet to the buffer.

        Never blocks. A producer that blocked here would stall the shared video
        read for every other path, so overflow is resolved by the drop policy
        and reported through the return value instead.

        Returns:
            True when the packet was stored, False when the drop policy
            discarded it.

        Raises:
            FrameBufferError: If the buffer is closed.
            TypeError: If ``packet`` is not a FramePacket.
        """
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        with self._not_empty:
            if self._closed:
                raise FrameBufferError(f"Cannot enqueue into closed buffer {self.name!r}")

            self.frames_offered += 1
            if self._first_put_time is None:
                self._first_put_time = time.monotonic()

            if len(self._queue) >= self.capacity:
                if self.drop_policy == "oldest":
                    self._queue.popleft()
                    self._queue.append(packet)
                    self.frames_dropped += 1
                    self.frames_accepted += 1
                    self._not_empty.notify()
                    return True
                # newest / drop_latest: refuse the incoming frame
                self.frames_dropped += 1
                return False

            self._queue.append(packet)
            self.frames_accepted += 1
            self.maximum_size = max(self.maximum_size, len(self._queue))
            self._not_empty.notify()
            return True

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    def get(self, block: bool = True, timeout: float | None = None) -> FramePacket:
        """Pop the oldest packet.

        Raises:
            Empty: On timeout, when non-blocking and empty, or when the buffer
                is closed and drained.
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._not_empty:
            while not self._queue:
                if self._closed:
                    raise Empty(f"FrameBuffer {self.name!r} is closed and empty")
                if not block:
                    raise Empty(f"FrameBuffer {self.name!r} is empty")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise Empty(f"FrameBuffer {self.name!r} get timed out")
                self._not_empty.wait(timeout=remaining)

            packet = self._queue.popleft()
            self.frames_consumed += 1
            return packet

    def get_nowait(self) -> FramePacket:
        return self.get(block=False)

    def try_get(self, timeout: float | None = None) -> FramePacket | None:
        """Like :meth:`get` but returns None instead of raising."""
        try:
            return self.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the buffer and wake every waiter."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def reset_counters(self) -> None:
        """Zero the throughput counters.

        Called when the venue is reconfigured: the totals describe a layout that
        no longer exists, and leaving them in place would report historical churn
        as a current fault.
        """
        with self._lock:
            self.frames_offered = 0
            self.frames_accepted = 0
            self.frames_consumed = 0
            self.frames_dropped = 0
            self.maximum_size = 0
            self._first_put_time = None

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "capacity": self.capacity,
                "drop_policy": self.drop_policy,
                "size": len(self._queue),
                "max_size": self.maximum_size,
                "offered": self.frames_offered,
                "accepted": self.frames_accepted,
                "consumed": self.frames_consumed,
                "dropped": self.frames_dropped,
                "throughput_fps": round(self.throughput_fps, 2),
                "closed": self._closed,
            }

    def __len__(self) -> int:
        return len(self._queue)

    def empty(self) -> bool:
        return len(self._queue) == 0
