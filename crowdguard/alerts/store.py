"""Alert store: deduplication, an incident log, and optional persistence.

The rule engine is stateless and re-evaluates every cycle, so a condition that
persists for a minute would otherwise produce hundreds of identical alerts. The
store collapses repeats into one active alert with an occurrence count, keeps a
bounded history of what has happened, and can append each new incident to a
JSON Lines file so there is a record after the process exits.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

from crowdguard.alerts.contracts import (
    SEVERITY_ORDER,
    Alert,
    AlertCategory,
    Severity,
    sort_alerts,
)

logger = logging.getLogger(__name__)


def _is_escalation(previous: Severity, current: Severity) -> bool:
    """Whether a severity change made the situation more urgent.

    SEVERITY_ORDER counts up as urgency falls, so an escalation is a decrease.
    Only escalations are logged as fresh incidents: a condition easing off is an
    update to the existing alert, not a new event to page anyone about.
    """
    return SEVERITY_ORDER[current] < SEVERITY_ORDER[previous]

# How long an active alert survives without being re-raised before it is
# considered resolved. Must comfortably exceed the evaluation period.
DEFAULT_RESOLVE_AFTER_S = 8.0


class AlertStore:
    """Thread-safe active-alert set plus a bounded incident history."""

    def __init__(
        self,
        history_size: int = 500,
        resolve_after_s: float = DEFAULT_RESOLVE_AFTER_S,
        log_path: str | Path | None = None,
    ) -> None:
        """
        Args:
            history_size: Incidents retained in memory.
            resolve_after_s: Silence required before an alert auto-resolves.
            log_path: Optional JSON Lines file appended to on each new incident.
        """
        self._active: dict[str, Alert] = {}
        self._history: deque[Alert] = deque(maxlen=history_size)
        self._resolve_after_s = resolve_after_s
        self._log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()

        self.total_raised = 0
        self.total_resolved = 0

        if self._log_path is not None:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Could not prepare alert log %s: %s", self._log_path, exc)
                self._log_path = None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def submit(self, alerts: list[Alert]) -> list[Alert]:
        """Merge a freshly evaluated alert list into the active set.

        An alert matching an existing one by ``dedup_key`` updates that entry in
        place, preserving ``first_seen`` and incrementing ``occurrences``, so the
        UI can show how long a condition has persisted. A genuinely new alert is
        recorded in the history and written to the log.

        Returns:
            Only the alerts that were newly raised by this call.
        """
        now = time.time()
        newly_raised: list[Alert] = []

        with self._lock:
            for alert in alerts:
                key = alert.dedup_key
                existing = self._active.get(key)

                if existing is None:
                    alert.first_seen = now
                    alert.timestamp = now
                    alert.occurrences = 1
                    self._active[key] = alert
                    self._history.append(alert)
                    self.total_raised += 1
                    newly_raised.append(alert)
                    continue

                # Same condition still true: refresh it rather than duplicate.
                existing.timestamp = now
                existing.occurrences += 1
                existing.description = alert.description
                existing.action = alert.action
                existing.detail = alert.detail
                existing.target_gate = alert.target_gate

                # An escalation is a new incident worth logging on its own.
                if alert.severity != existing.severity:
                    previous = existing.severity
                    existing.severity = alert.severity
                    existing.title = alert.title
                    if _is_escalation(previous, alert.severity):
                        self._history.append(alert)
                        self.total_raised += 1
                        newly_raised.append(alert)

            self._expire_locked(now)
            self._suppress_stale_all_clear_locked()

        for alert in newly_raised:
            self._append_to_log(alert)

        return newly_raised

    def _suppress_stale_all_clear_locked(self) -> None:
        """Drop an all-clear that is held open next to a live alert.

        The rule engine only emits ALL_CLEAR when a cycle finds nothing, but
        anomaly detections fluctuate between cycles, so a quiet cycle can raise
        it moments before a real alert lands. Both then sit inside the resolve
        window and the operator sees "all gates within safe limits" directly
        above "person on the ground", which is worse than showing nothing.

        Caller holds the lock.
        """
        if not any(a.severity is not Severity.INFO for a in self._active.values()):
            return
        for key, alert in list(self._active.items()):
            if alert.rule_id == "ALL_CLEAR":
                del self._active[key]

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    def _expire_locked(self, now: float) -> None:
        """Drop alerts that have not been re-raised recently. Caller holds the lock."""
        stale_keys = [
            key for key, alert in self._active.items()
            if now - alert.timestamp > self._resolve_after_s
        ]
        for key in stale_keys:
            del self._active[key]
            self.total_resolved += 1

    def expire_stale(self) -> None:
        """Run expiry without submitting anything, e.g. on a paused pipeline."""
        with self._lock:
            self._expire_locked(time.time())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active(self) -> list[Alert]:
        """Current alerts, most urgent first."""
        with self._lock:
            return sort_alerts(list(self._active.values()))

    def active_for_gate(self, gate_id: str) -> list[Alert]:
        with self._lock:
            return sort_alerts([a for a in self._active.values() if a.gate_id == gate_id])

    def active_by_category(self, category: AlertCategory) -> list[Alert]:
        with self._lock:
            return sort_alerts([a for a in self._active.values() if a.category is category])

    def history(self, limit: int = 100) -> list[Alert]:
        """Most recent incidents, newest first."""
        with self._lock:
            return list(self._history)[-limit:][::-1]

    def counts_by_severity(self) -> dict[str, int]:
        with self._lock:
            counts = {severity.value: 0 for severity in Severity}
            for alert in self._active.values():
                counts[alert.severity.value] += 1
            return counts

    def worst_severity(self) -> Severity | None:
        """Highest urgency currently active, or None when nothing is active."""
        active = self.active()
        return active[0].severity if active else None

    def clear(self) -> None:
        with self._lock:
            self._active.clear()
            self._history.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "active": len(self._active),
                "history": len(self._history),
                "total_raised": self.total_raised,
                "total_resolved": self.total_resolved,
                "resolve_after_s": self._resolve_after_s,
                "log_path": str(self._log_path) if self._log_path else None,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_to_log(self, alert: Alert) -> None:
        """Append one incident as a JSON line.

        Failures are logged and swallowed: losing the audit trail must never
        take down live monitoring.
        """
        if self._log_path is None:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(alert.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.warning("Could not append to alert log: %s", exc)
