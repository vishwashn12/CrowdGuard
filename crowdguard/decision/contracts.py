"""Model-agnostic data contracts for the Decision Engine.

These define the interface between model inference and decision-making. They
import nothing from torch, cv2, CSRNet, YOLO, or any vision model code.

The engine consumes :class:`CrowdState` and produces :class:`DecisionResult`.

Note: crowd *anomalies* (sitting, counter-flow) deliberately do not appear here.
They travel on a separate alerting channel (``crowdguard.alerts``) so that
congestion routing stays a pure function of density, flow and capacity. See
``docs`` notes in the project README.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum


# ===================================================================
# Enumerations
# ===================================================================


class CongestionLevel(str, Enum):
    """Discrete congestion classification for a gate."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Trend(str, Enum):
    """Crowd density trend direction."""

    DECREASING = "DECREASING"
    STABLE = "STABLE"
    INCREASING = "INCREASING"


class ActionType(str, Enum):
    """Decision action types."""

    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    DIVERT = "DIVERT"
    RESTRICT_INFLOW = "RESTRICT_INFLOW"
    HOLD = "HOLD"


class MotionStatus(str, Enum):
    """Availability of the motion half of an observation."""

    OK = "OK"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


# Ordering used for threshold comparisons.
LEVEL_ORDER: dict[CongestionLevel, int] = {
    CongestionLevel.LOW: 0,
    CongestionLevel.MODERATE: 1,
    CongestionLevel.HIGH: 2,
    CongestionLevel.CRITICAL: 3,
}


def level_at_least(level: CongestionLevel, threshold: str | CongestionLevel) -> bool:
    """Whether ``level`` meets or exceeds a named threshold."""
    if isinstance(threshold, str):
        try:
            threshold = CongestionLevel(threshold.upper())
        except ValueError:
            threshold = CongestionLevel.HIGH
    return LEVEL_ORDER[level] >= LEVEL_ORDER[threshold]


# ===================================================================
# CrowdState - model-agnostic observation at a gate
# ===================================================================


@dataclass(frozen=True)
class CrowdState:
    """Model-independent observation of crowd conditions at a single gate.

    The only input contract the Decision Engine consumes. It never references
    CSRNet, YOLO, torch, cv2, or any model-specific type.

    Attributes:
        gate_id: Gate identifier from the venue config (never a camera_id).
        timestamp: Observation timestamp (wall-clock seconds since epoch).
        crowd_count: Estimated people present, from any density source.
        density_score: Normalised density in [0, 1] (1 = at or over capacity).
        velocity: Crowd movement magnitude in px/s (0 when motion is invalid).
        direction: Dominant movement direction in degrees [0, 360).
        flow_consistency: Flow coherence in [0, 1] (1 = perfectly uniform).
        congestion_score: Composite congestion score in [0, 1].
        congestion_level: Discrete congestion classification.
        trend: Density trend direction.
        confidence: Overall observation confidence in [0, 1].
        source_frame_id: Identifier of the frame this derives from.
        result_age_s: Age of the underlying model result, in seconds.
        has_density: True when real density data backs this observation.
        has_motion: True when valid, fresh motion data backs this observation.
        motion_status: Why motion is or is not usable.
        motion_provenance: Human-readable description of the motion source.
        motion_age_frames: Frame-id gap between the density and motion sources.
        is_synthetic: True when the state was injected for simulation or testing.
        velocity_unit: "px/s" for an uncalibrated camera, "m/s" for a calibrated one.
        area_m2: Area of the calibrated ground zone, or None when uncalibrated.
    """

    gate_id: str
    timestamp: float

    crowd_count: float = 0.0
    density_score: float = 0.0

    velocity: float = 0.0
    direction: float = 0.0
    flow_consistency: float = 0.0

    congestion_score: float = 0.0
    congestion_level: CongestionLevel = CongestionLevel.LOW
    trend: Trend = Trend.STABLE

    confidence: float = 1.0

    source_frame_id: int | None = None
    result_age_s: float = 0.0

    has_density: bool = True
    has_motion: bool = True
    motion_status: MotionStatus = MotionStatus.OK
    motion_provenance: str | None = None
    motion_age_frames: int = 0

    is_synthetic: bool = False

    velocity_unit: str = "px/s"
    area_m2: float | None = None

    def __post_init__(self) -> None:
        if not self.gate_id:
            raise ValueError("gate_id must not be empty")
        if not (0.0 <= self.density_score <= 1.0):
            raise ValueError(f"density_score must be in [0, 1], got {self.density_score}")
        if not (0.0 <= self.congestion_score <= 1.0):
            raise ValueError(f"congestion_score must be in [0, 1], got {self.congestion_score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not (0.0 <= self.flow_consistency <= 1.0):
            raise ValueError(f"flow_consistency must be in [0, 1], got {self.flow_consistency}")
        if self.velocity < 0:
            raise ValueError(f"velocity must be >= 0, got {self.velocity}")
        if not (0.0 <= self.direction < 360.0):
            raise ValueError(f"direction must be in [0, 360), got {self.direction}")

    @property
    def capacity_utilization(self) -> float:
        """Alias of ``density_score``, which is already count/capacity clamped."""
        return self.density_score

    @property
    def is_calibrated(self) -> bool:
        return self.area_m2 is not None and self.area_m2 > 0

    @property
    def density_per_m2(self) -> float | None:
        """Physical crowd density in the calibrated zone, people/m^2."""
        return self.crowd_count / self.area_m2 if self.is_calibrated else None


# ===================================================================
# CandidateEvaluation - routing candidate assessment
# ===================================================================


@dataclass(frozen=True)
class CandidateEvaluation:
    """Assessment of one candidate gate as a diversion target.

    Emitted for every reachable gate, viable or not, so an operator can see why
    a gate was passed over.
    """

    gate_id: str
    distance_m: float
    congestion_level: CongestionLevel
    congestion_score: float
    capacity_utilization: float
    trend: Trend
    routing_score: float
    is_viable: bool
    is_monitored: bool = True
    rejection_reason: str | None = None


# ===================================================================
# DecisionResult - output of the Decision Engine
# ===================================================================


@dataclass(frozen=True)
class DecisionResult:
    """Immutable outcome of evaluating a single source gate.

    Carries the action, its target, the reasoning, and every candidate
    evaluation, so each decision is auditable after the fact.
    """

    decision_id: str
    timestamp: float
    source_gate_id: str
    action: ActionType
    target_gate_id: str | None = None
    severity: CongestionLevel = CongestionLevel.LOW
    confidence: float = 1.0
    reason_codes: tuple[str, ...] = ()
    human_readable_reason: str = ""
    candidate_evaluations: tuple[CandidateEvaluation, ...] = ()
    route: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_gate_id:
            raise ValueError("source_gate_id must not be empty")
        if self.action == ActionType.DIVERT and self.target_gate_id is None:
            raise ValueError("DIVERT action requires a target_gate_id")

    @staticmethod
    def generate_id() -> str:
        """Generate a unique decision identifier."""
        return f"DEC-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> dict:
        """JSON-serialisable view for the dashboard and logs."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "source_gate_id": self.source_gate_id,
            "action": self.action.value,
            "target_gate_id": self.target_gate_id,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 3),
            "reason_codes": list(self.reason_codes),
            "reason": self.human_readable_reason,
            "route": list(self.route),
            "candidates": [
                {
                    "gate_id": c.gate_id,
                    "distance_m": round(c.distance_m, 1),
                    "congestion_level": c.congestion_level.value,
                    "congestion_score": round(c.congestion_score, 3),
                    "capacity_utilization": round(c.capacity_utilization, 3),
                    "trend": c.trend.value,
                    "routing_score": round(c.routing_score, 3),
                    "is_viable": c.is_viable,
                    "is_monitored": c.is_monitored,
                    "rejection_reason": c.rejection_reason,
                }
                for c in self.candidate_evaluations
            ],
        }
