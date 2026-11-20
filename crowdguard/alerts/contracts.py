"""Alert contracts shared by the rule engine and the alert store."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """Operational urgency of an alert."""

    EMERGENCY = "EMERGENCY"  # act now, risk to life
    ALERT = "ALERT"          # act within minutes
    WARNING = "WARNING"      # watch closely
    INFO = "INFO"            # informational


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.EMERGENCY: 0,
    Severity.ALERT: 1,
    Severity.WARNING: 2,
    Severity.INFO: 3,
}


class AlertCategory(str, Enum):
    """Which subsystem raised the alert.

    The split matters operationally: a congestion alert is answered by moving
    crowd flow, an anomaly alert by sending staff to an individual, and a system
    alert by fixing the monitoring setup itself.
    """

    CONGESTION = "CONGESTION"
    ANOMALY = "ANOMALY"
    SYSTEM = "SYSTEM"


@dataclass
class Alert:
    """One actionable advisory for the operator.

    Attributes:
        rule_id: Stable identifier of the rule that fired.
        severity: Operational urgency.
        category: Which subsystem raised it.
        gate_id: Gate it concerns, or ``"ALL"`` for venue-wide.
        title: One-line summary for a list view.
        description: What was observed, with the numbers behind it.
        action: What the operator should physically do about it.
        target_gate: Related gate, e.g. a diversion destination.
        detail: Structured extras for the UI (track ids, counts).
        alert_id: Unique identifier for this instance.
        timestamp: When it was raised.
        first_seen: When this rule first fired for this gate in the current run.
        occurrences: How many times it has fired while remaining active.
    """

    rule_id: str
    severity: Severity
    category: AlertCategory
    gate_id: str
    title: str
    description: str
    action: str
    target_gate: str | None = None
    detail: dict = field(default_factory=dict)

    alert_id: str = field(default_factory=lambda: f"ALR-{uuid.uuid4().hex[:8].upper()}")
    timestamp: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    occurrences: int = 1

    @property
    def dedup_key(self) -> str:
        """Identity used to recognise a repeat of the same condition."""
        return f"{self.rule_id}:{self.gate_id}"

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.first_seen)

    @property
    def is_actionable(self) -> bool:
        return self.severity in (Severity.EMERGENCY, Severity.ALERT)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "gate_id": self.gate_id,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "target_gate": self.target_gate,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "first_seen": self.first_seen,
            "age_s": round(self.age_s, 1),
            "occurrences": self.occurrences,
        }


def sort_alerts(alerts: list[Alert]) -> list[Alert]:
    """Order alerts most urgent first, newest first within a severity."""
    return sorted(
        alerts,
        key=lambda a: (SEVERITY_ORDER.get(a.severity, 99), -a.timestamp),
    )
