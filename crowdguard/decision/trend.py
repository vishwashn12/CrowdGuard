"""Per-gate trend tracking over a rolling window of congestion scores.

The window is split in half and the halves compared: a rising recent average
means the crowd is building. No dependency on torch, cv2, or any vision model.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import Trend


class TrendTracker:
    """Rolling-window trend tracker, keyed by gate.

    With a window of N observations::

        older  = scores[:N//2]
        recent = scores[N//2:]

        avg(recent) - avg(older) >  threshold -> INCREASING
        avg(recent) - avg(older) < -threshold -> DECREASING
        otherwise                             -> STABLE

    Thread-safe: observations arrive from worker threads while the dashboard and
    engine read from others.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        config = config or DecisionConfig()
        self._window_size = config.trend_window_size
        self._threshold = config.trend_change_threshold
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window_size)
        )
        self._lock = threading.Lock()

    def _classify(self, scores: list[float]) -> Trend:
        if len(scores) < 2:
            return Trend.STABLE

        mid = max(1, len(scores) // 2)
        older = scores[:mid]
        recent = scores[mid:]
        if not recent:
            return Trend.STABLE

        diff = (sum(recent) / len(recent)) - (sum(older) / len(older))

        if diff > self._threshold:
            return Trend.INCREASING
        if diff < -self._threshold:
            return Trend.DECREASING
        return Trend.STABLE

    def update(self, gate_id: str, congestion_score: float) -> Trend:
        """Record an observation and return the resulting trend."""
        with self._lock:
            window = self._history[gate_id]
            window.append(congestion_score)
            return self._classify(list(window))

    def get_trend(self, gate_id: str) -> Trend:
        """Current trend for a gate, without recording a new observation."""
        with self._lock:
            window = self._history.get(gate_id)
            return self._classify(list(window)) if window else Trend.STABLE

    def history(self, gate_id: str) -> list[float]:
        """Copy of the retained score window for a gate."""
        with self._lock:
            window = self._history.get(gate_id)
            return list(window) if window else []

    def reset(self, gate_id: str | None = None) -> None:
        """Clear trend history for one gate, or all of them."""
        with self._lock:
            if gate_id is not None:
                self._history.pop(gate_id, None)
            else:
                self._history.clear()
