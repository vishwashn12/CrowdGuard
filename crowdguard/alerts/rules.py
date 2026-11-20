"""Rule engine producing operator advisories.

Two independent families of rules run over every gate:

* **Congestion rules** read the fused :class:`CrowdState` and the engine's
  :class:`DecisionResult`. Within a gate these are mutually exclusive and
  evaluated most-severe first, because an operator facing a crush does not also
  need to be told the gate is busy.

* **Anomaly rules** read the per-person :class:`AnomalyResult`. These are
  evaluated *independently* of the congestion rules, because they call for a
  different response: a person on the ground needs staff sent to them, not a
  change in crowd routing. A gate can legitimately raise one of each.

Every rule is deterministic and threshold-based. Nothing here is learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crowdguard.alerts.contracts import Alert, AlertCategory, Severity, sort_alerts
from crowdguard.decision.contracts import (
    ActionType,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    MotionStatus,
    Trend,
)
from crowdguard.models.contracts import AnomalyResult

# --- congestion thresholds -------------------------------------------
CRUSH_VELOCITY_PX_S = 5.0
CRUSH_MIN_COUNT = 20.0
SLOW_FLOW_VELOCITY_PX_S = 10.0
SLOW_FLOW_MIN_COUNT = 15.0
SURGE_SCORE = 0.65
IMBALANCE_RATIO = 2.5
IMBALANCE_MIN_COUNT = 30.0

# --- anomaly thresholds ----------------------------------------------
# A crowd is "moving" once its median speed clears this, which is when a person
# on the ground stops being merely unusual and becomes a trip hazard.
MOVING_CROWD_VELOCITY_PX_S = 6.0
COUNTERFLOW_MIN_PEOPLE = 2
COUNTERFLOW_ALERT_PEOPLE = 4
ANOMALY_CLUSTER_RATIO = 0.25
ANOMALY_CLUSTER_MIN_PEOPLE = 3
CONTINUITY_BREAK_RATIO = 0.2


@dataclass
class GateView:
    """Everything known about one gate at evaluation time."""

    gate_id: str
    capacity: int
    state: CrowdState | None = None
    decision: DecisionResult | None = None
    anomaly: AnomalyResult | None = None
    camera_id: str | None = None
    is_monitored: bool = True
    continuity_break_ratio: float = 0.0
    max_expected_velocity: float = 75.0
    extra: dict = field(default_factory=dict)

    @property
    def crowd_count(self) -> float:
        return self.state.crowd_count if self.state else 0.0

    @property
    def utilization(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return self.crowd_count / self.capacity

    @property
    def velocity(self) -> float:
        return self.state.velocity if self.state else 0.0

    @property
    def congestion_level(self) -> CongestionLevel:
        return self.state.congestion_level if self.state else CongestionLevel.LOW

    @property
    def congestion_score(self) -> float:
        return self.state.congestion_score if self.state else 0.0

    @property
    def trend(self) -> Trend:
        return self.state.trend if self.state else Trend.STABLE

    @property
    def action(self) -> ActionType:
        return self.decision.action if self.decision else ActionType.NO_ACTION

    @property
    def target_gate(self) -> str | None:
        return self.decision.target_gate_id if self.decision else None

    @property
    def motion_status(self) -> MotionStatus:
        return self.state.motion_status if self.state else MotionStatus.UNAVAILABLE

    @property
    def crowd_is_moving(self) -> bool:
        """Whether the crowd here has meaningful bulk motion."""
        return (
            self.motion_status is MotionStatus.OK
            and self.velocity >= MOVING_CROWD_VELOCITY_PX_S
        )


# =====================================================================
# Congestion rules
# =====================================================================


def _congestion_alerts(view: GateView) -> list[Alert]:
    """Most-severe-first congestion assessment for one gate.

    Returns at most one alert: the chain is exclusive by design.
    """
    gate = view.gate_id
    count = view.crowd_count
    capacity = view.capacity
    util = view.utilization
    velocity = view.velocity
    score = view.congestion_score
    level = view.congestion_level
    target = view.target_gate

    detail = {
        "crowd_count": round(count, 1),
        "capacity": capacity,
        "utilization": round(util, 3),
        "velocity": round(velocity, 1),
        "congestion_score": round(score, 3),
        "congestion_level": level.value,
        "trend": view.trend.value,
    }

    # --- 1. crush risk: packed and barely moving ---------------------
    if (
        level in (CongestionLevel.HIGH, CongestionLevel.CRITICAL)
        and velocity < CRUSH_VELOCITY_PX_S
        and view.motion_status is not MotionStatus.UNAVAILABLE
        and count > CRUSH_MIN_COUNT
    ):
        return [Alert(
            rule_id="CRUSH_RISK",
            severity=Severity.EMERGENCY,
            category=AlertCategory.CONGESTION,
            gate_id=gate,
            title=f"Crush risk at {gate}: dense crowd, almost no movement",
            description=(
                f"{gate} holds about {count:.0f} people and the crowd is moving at only "
                f"{velocity:.1f} px/s. Density with no outlet is the precondition for a "
                f"crush. Congestion score {score:.0%}."
            ),
            action=(
                f"Immediately: stop new entries at {gate}, send ground staff, and open "
                f"barriers to create space. Redirect arrivals to the nearest alternative gate."
            ),
            target_gate=target,
            detail=detail,
        )]

    # --- 2. over capacity -------------------------------------------
    if util >= 1.0:
        return [Alert(
            rule_id="GATE_OVERLOAD",
            severity=Severity.EMERGENCY,
            category=AlertCategory.CONGESTION,
            gate_id=gate,
            title=f"{gate} is over capacity",
            description=(
                f"{gate} holds about {count:.0f} people against a stated capacity of "
                f"{capacity} ({util:.0%})."
            ),
            action=(
                f"Close {gate} to new entries now and staff the approach to hold the queue. "
                f"Route arrivals to other gates."
            ),
            target_gate=target,
            detail=detail,
        )]

    # --- 3. congested with nowhere to send people -------------------
    if view.action is ActionType.RESTRICT_INFLOW:
        return [Alert(
            rule_id="ALL_GATES_FULL",
            severity=Severity.EMERGENCY,
            category=AlertCategory.CONGESTION,
            gate_id=gate,
            title=f"{gate} congested and no diversion is available",
            description=(
                f"{gate} is {level.value} (score {score:.0%}) but every reachable gate is "
                f"itself congested or out of headroom, so there is nowhere to divert to."
            ),
            action=(
                f"Hold arrivals upstream of {gate} rather than redistributing them. "
                f"Escalate to venue control: the venue as a whole is at its limit."
            ),
            detail=detail,
        )]

    # --- 4. building rapidly ----------------------------------------
    if (
        view.trend is Trend.INCREASING
        and score > SURGE_SCORE
        and level in (CongestionLevel.HIGH, CongestionLevel.CRITICAL)
    ):
        return [Alert(
            rule_id="BOTTLENECK_FORMING",
            severity=Severity.ALERT,
            category=AlertCategory.CONGESTION,
            gate_id=gate,
            title=f"Crowd building rapidly at {gate}",
            description=(
                f"{gate} is at {count:.0f}/{capacity} with a congestion score of {score:.0%} "
                f"and still rising. On this trajectory it becomes a bottleneck within minutes."
            ),
            action=(
                f"Pre-position barriers and staff at {gate} for a diversion, and announce "
                f"alternative entrances before the queue sets."
            ),
            target_gate=target,
            detail=detail,
        )]

    # --- 5. diversion running ---------------------------------------
    if view.action is ActionType.DIVERT and target:
        route = " -> ".join(view.decision.route) if view.decision and view.decision.route else f"{gate} -> {target}"
        return [Alert(
            rule_id="DIVERT_IN_PROGRESS",
            severity=Severity.ALERT,
            category=AlertCategory.CONGESTION,
            gate_id=gate,
            title=f"Diversion active: {gate} to {target}",
            description=(
                f"Crowd is being redirected from {gate} (score {score:.0%}, "
                f"{count:.0f}/{capacity}) to {target} along the perimeter route {route}."
            ),
            action=(
                f"1. Staff the {gate} to {target} pathway. "
                f"2. Place directional signage toward {target}. "
                f"3. Announce: please proceed to {target}, {gate} is temporarily at capacity."
            ),
            target_gate=target,
            detail={**detail, "route": route},
        )]

    # --- 6. flow slowing --------------------------------------------
    if (
        level in (CongestionLevel.MODERATE, CongestionLevel.HIGH)
        and velocity < SLOW_FLOW_VELOCITY_PX_S
        and count > SLOW_FLOW_MIN_COUNT
        and view.motion_status is not MotionStatus.UNAVAILABLE
    ):
        return [Alert(
            rule_id="SLOW_FLOW_WARNING",
            severity=Severity.WARNING,
            category=AlertCategory.CONGESTION,
            gate_id=gate,
            title=f"Flow slowing at {gate}",
            description=(
                f"{gate} has about {count:.0f} people moving at {velocity:.1f} px/s. "
                f"Falling speed at steady density is how a crush begins."
            ),
            action=(
                f"Watch {gate} closely and open an additional lane if possible. "
                f"If speed falls below {CRUSH_VELOCITY_PX_S:.0f} px/s, divert."
            ),
            detail=detail,
        )]

    # --- 7. on the watchlist ----------------------------------------
    if view.action is ActionType.MONITOR and view.trend is Trend.INCREASING:
        return [Alert(
            rule_id="MONITOR_SURGE",
            severity=Severity.WARNING,
            category=AlertCategory.CONGESTION,
            gate_id=gate,
            title=f"Inflow growing at {gate}",
            description=(
                f"{gate} is at {count:.0f}/{capacity} ({util:.0%}) and still increasing. "
                f"Congestion score {score:.0%}. No action needed yet."
            ),
            action=(
                f"Alert staff at {gate} to stand by, and confirm which neighbouring gates "
                f"have spare capacity for overflow."
            ),
            detail=detail,
        )]

    return []


def _imbalance_alerts(views: list[GateView]) -> list[Alert]:
    """Flag a gate carrying far more than its share of the venue load.

    Acting here is cheap: shifting arrivals before a queue forms avoids the
    emergency entirely.
    """
    active = [v for v in views if v.crowd_count > 0]
    if len(active) < 2:
        return []

    average = sum(v.crowd_count for v in active) / len(active)
    if average <= 0:
        return []

    lightest = min(active, key=lambda v: v.crowd_count)
    alerts: list[Alert] = []

    for view in active:
        if view.crowd_count <= average * IMBALANCE_RATIO:
            continue
        if view.crowd_count <= IMBALANCE_MIN_COUNT:
            continue
        if view.gate_id == lightest.gate_id:
            continue

        alerts.append(Alert(
            rule_id="IMBALANCED_LOAD",
            severity=Severity.WARNING,
            category=AlertCategory.CONGESTION,
            gate_id=view.gate_id,
            title=f"Load imbalance: {view.gate_id} against {lightest.gate_id}",
            description=(
                f"{view.gate_id} holds about {view.crowd_count:.0f} people while "
                f"{lightest.gate_id} holds {lightest.crowd_count:.0f}. The venue load is "
                f"{view.crowd_count / average:.1f}x the average at this gate."
            ),
            action=(
                f"Direct a share of arrivals from {view.gate_id} to {lightest.gate_id} now, "
                f"before congestion forms. Place staff and signage on the approach."
            ),
            target_gate=lightest.gate_id,
            detail={
                "crowd_count": round(view.crowd_count, 1),
                "venue_average": round(average, 1),
                "lightest_gate": lightest.gate_id,
                "lightest_count": round(lightest.crowd_count, 1),
            },
        ))

    return alerts


# =====================================================================
# Anomaly rules
# =====================================================================


def _anomaly_alerts(view: GateView) -> list[Alert]:
    """Per-person behaviour assessment for one gate.

    Independent of the congestion chain: these call for staff to be sent to an
    individual, not for crowd flow to be rerouted.
    """
    anomaly = view.anomaly
    if anomaly is None:
        return []

    alerts: list[Alert] = []
    gate = view.gate_id

    # --- person down / seated in a moving crowd ---------------------
    if anomaly.sitting_count > 0:
        moving = view.crowd_is_moving
        congested = view.congestion_level in (CongestionLevel.HIGH, CongestionLevel.CRITICAL)

        # Someone on the ground is only an emergency when the crowd around them
        # is actually moving or packed: that is what turns a person into a trip
        # hazard and a stampede nucleus. A seated person in a calm, sparse area
        # is noteworthy, not urgent.
        if moving or congested:
            severity = Severity.EMERGENCY
            urgency = (
                "The crowd around them is moving" if moving
                else "The gate is congested around them"
            )
        else:
            severity = Severity.WARNING
            urgency = "The surrounding area is calm, so the immediate risk is lower"

        track_ids = [p.track_id for p in anomaly.anomalous_people() if p.label.name == "SITTING"]

        alerts.append(Alert(
            rule_id="PERSON_DOWN",
            severity=severity,
            category=AlertCategory.ANOMALY,
            gate_id=gate,
            title=(
                f"{anomaly.sitting_count} "
                f"{'person' if anomaly.sitting_count == 1 else 'people'} "
                f"on the ground at {gate}"
            ),
            description=(
                f"Pose analysis has {anomaly.sitting_count} "
                f"{'person' if anomaly.sitting_count == 1 else 'people'} sitting, squatting or "
                f"collapsed at {gate}, out of {anomaly.tracked_count} tracked. {urgency}. "
                f"A person at ground level in a flow is a trip hazard and a stampede trigger."
            ),
            action=(
                f"Send ground staff to {gate} to reach the "
                f"{'person' if anomaly.sitting_count == 1 else 'people'} and establish whether "
                f"medical help is needed. Create a protective space around them and slow the "
                f"approaching flow while they are assisted."
            ),
            detail={
                "sitting_count": anomaly.sitting_count,
                "tracked_count": anomaly.tracked_count,
                "track_ids": track_ids,
                "crowd_moving": moving,
                "crowd_velocity": round(view.velocity, 1),
            },
        ))

    # --- counter-flow ------------------------------------------------
    if anomaly.opposite_count >= COUNTERFLOW_MIN_PEOPLE and anomaly.crowd_direction_valid:
        severity = (
            Severity.ALERT if anomaly.opposite_count >= COUNTERFLOW_ALERT_PEOPLE
            else Severity.WARNING
        )
        track_ids = [p.track_id for p in anomaly.anomalous_people() if p.label.name == "OPPOSITE"]

        alerts.append(Alert(
            rule_id="COUNTERFLOW_DETECTED",
            severity=severity,
            category=AlertCategory.ANOMALY,
            gate_id=gate,
            title=f"{anomaly.opposite_count} people moving against the flow at {gate}",
            description=(
                f"{anomaly.opposite_count} of {anomaly.tracked_count} tracked people at {gate} "
                f"are moving against the dominant crowd direction "
                f"({anomaly.crowd_direction_deg:.0f} deg). Counter-flow creates turbulence and "
                f"blockages, and in dense conditions propagates as a shockwave."
            ),
            action=(
                f"Send staff to {gate} to identify why people are turning back - a blocked exit, "
                f"a lost group, or confusing signage - and resolve the cause. Consider a "
                f"physically separated return lane."
            ),
            detail={
                "opposite_count": anomaly.opposite_count,
                "tracked_count": anomaly.tracked_count,
                "track_ids": track_ids,
                "crowd_direction_deg": round(anomaly.crowd_direction_deg, 1),
            },
        ))

    # --- widespread abnormal behaviour -------------------------------
    if (
        anomaly.tracked_count >= ANOMALY_CLUSTER_MIN_PEOPLE
        and anomaly.anomaly_ratio >= ANOMALY_CLUSTER_RATIO
        and anomaly.anomaly_count >= ANOMALY_CLUSTER_MIN_PEOPLE
    ):
        alerts.append(Alert(
            rule_id="ANOMALY_CLUSTER",
            severity=Severity.ALERT,
            category=AlertCategory.ANOMALY,
            gate_id=gate,
            title=f"Abnormal behaviour is widespread at {gate}",
            description=(
                f"{anomaly.anomaly_count} of {anomaly.tracked_count} tracked people at {gate} "
                f"({anomaly.anomaly_ratio:.0%}) are behaving abnormally - "
                f"{anomaly.sitting_count} on the ground, {anomaly.opposite_count} against the "
                f"flow. A cluster this size usually means something at the gate itself has "
                f"gone wrong, not that individuals are behaving oddly."
            ),
            action=(
                f"Treat {gate} as a developing incident. Send a supervisor to assess the gate "
                f"directly and prepare to halt inflow while the cause is found."
            ),
            detail={
                "anomaly_count": anomaly.anomaly_count,
                "anomaly_ratio": round(anomaly.anomaly_ratio, 3),
                "sitting_count": anomaly.sitting_count,
                "opposite_count": anomaly.opposite_count,
                "tracked_count": anomaly.tracked_count,
            },
        ))

    return alerts


def _system_alerts(views: list[GateView]) -> list[Alert]:
    """Alerts about the monitoring system rather than the crowd.

    These exist so that degraded detection is never mistaken for a quiet venue.
    """
    alerts: list[Alert] = []

    for view in views:
        # Pose branch dropping frames: trajectory features are unreliable.
        if view.continuity_break_ratio >= CONTINUITY_BREAK_RATIO:
            alerts.append(Alert(
                rule_id="TRACKING_DEGRADED",
                severity=Severity.WARNING,
                category=AlertCategory.SYSTEM,
                gate_id=view.gate_id,
                title=f"Anomaly tracking is degraded at {view.gate_id}",
                description=(
                    f"Over the last 10 s, {view.continuity_break_ratio:.0%} of pose frames were "
                    f"skipped or arrived with gaps because analysis could not keep up with live "
                    f"video. Sitting and counter-flow detection are less reliable than normal "
                    f"here; congestion management is unaffected."
                ),
                action=(
                    "The machine cannot analyse frames as fast as they arrive. Close other heavy "
                    "programs, raise anomaly_stride, or use fewer cameras. If a browser on this "
                    "machine is showing the dashboard, opening it with scripts/open_dashboard.py "
                    "moves its drawing to the integrated GPU. Treat missing anomaly alerts here "
                    "as unconfirmed, not absent."
                ),
                detail={"continuity_break_ratio": round(view.continuity_break_ratio, 3)},
            ))

        # Velocity term pinned at its ceiling: the motion half of the congestion
        # score is contributing nothing, which looks like a calm venue rather
        # than a miscalibrated one.
        if (
            view.state is not None
            and view.is_monitored
            and view.state.motion_status is MotionStatus.OK
            and view.velocity >= view.max_expected_velocity
        ):
            alerts.append(Alert(
                rule_id="VELOCITY_UNCALIBRATED",
                severity=Severity.INFO,
                category=AlertCategory.SYSTEM,
                gate_id=view.gate_id,
                title=f"Velocity scale needs calibrating at {view.gate_id}",
                description=(
                    f"Observed crowd velocity at {view.gate_id} is {view.velocity:.0f} px/s, at or "
                    f"above the {view.max_expected_velocity:.0f} px/s ceiling that marks a freely "
                    f"flowing crowd. The inverse-velocity term is therefore pinned at zero and "
                    f"congestion is effectively being scored from density alone, so a slowing "
                    f"crowd will not register until it is already dense."
                ),
                action=(
                    "Raise congestion_weights.max_expected_velocity to roughly the px/s "
                    "this camera shows for a freely moving crowd. It is a pixel rate, so it "
                    "must be set per camera: resolution, zoom and distance all change it."
                ),
                detail={
                    "velocity": round(view.velocity, 1),
                    "max_expected_velocity": view.max_expected_velocity,
                },
            ))

        # Motion unavailable: congestion is being scored on density alone.
        if view.state is not None and view.state.motion_status is MotionStatus.UNAVAILABLE and view.is_monitored:
            alerts.append(Alert(
                rule_id="MOTION_UNAVAILABLE",
                severity=Severity.INFO,
                category=AlertCategory.SYSTEM,
                gate_id=view.gate_id,
                title=f"No motion data at {view.gate_id}",
                description=(
                    f"Congestion at {view.gate_id} is being scored from density and capacity "
                    f"only, because no usable motion observation is available. Confidence is "
                    f"reduced accordingly: {view.state.confidence:.0%}."
                ),
                action=(
                    "Check that the pose branch is running for this camera. Until it is, a "
                    "slow-moving dense crowd and a free-flowing one cannot be distinguished."
                ),
                detail={"confidence": round(view.state.confidence, 3)},
            ))

    return alerts


# =====================================================================
# Entry point
# =====================================================================


def generate_alerts(views: dict[str, GateView] | list[GateView]) -> list[Alert]:
    """Evaluate every rule family over the current gate views.

    Returns:
        Alerts sorted most urgent first. When nothing fires at all, a single
        INFO all-clear is returned so the operator can distinguish a healthy
        venue from a stalled pipeline.
    """
    view_list = list(views.values()) if isinstance(views, dict) else list(views)
    if not view_list:
        return []

    alerts: list[Alert] = []

    for view in view_list:
        if view.state is not None:
            alerts.extend(_congestion_alerts(view))
        alerts.extend(_anomaly_alerts(view))

    alerts.extend(_imbalance_alerts(view_list))
    alerts.extend(_system_alerts(view_list))

    # Suppress the softer load-imbalance notice where a gate already has a
    # hard congestion alert; the operator is acting on that gate regardless.
    hard = {
        (a.gate_id, a.rule_id) for a in alerts
        if a.rule_id in {"CRUSH_RISK", "GATE_OVERLOAD", "ALL_GATES_FULL"}
    }
    hard_gates = {gate for gate, _ in hard}
    alerts = [
        a for a in alerts
        if not (a.rule_id == "IMBALANCED_LOAD" and a.gate_id in hard_gates)
    ]

    if not any(a.severity is not Severity.INFO for a in alerts):
        total = sum(v.crowd_count for v in view_list)
        monitored = sum(1 for v in view_list if v.is_monitored)
        alerts.append(Alert(
            rule_id="ALL_CLEAR",
            severity=Severity.INFO,
            category=AlertCategory.SYSTEM,
            gate_id="ALL",
            title="All gates within safe limits",
            description=(
                f"{monitored} monitored "
                f"{'gate' if monitored == 1 else 'gates'} of {len(view_list)} are operating "
                f"normally. Estimated {total:.0f} people across the venue."
            ),
            action="No action required. Continue routine monitoring.",
            detail={"total_crowd": round(total, 1), "monitored_gates": monitored},
        ))

    return sort_alerts(alerts)
