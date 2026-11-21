"""Operator instructions: short, fixed-vocabulary commands and incidents.

The alert rules and the decision engine say a lot. Under stress an operator
needs much less: *what to do, where, and in three words why*. This module turns
their output into two lists:

* **Commands** - what to do with the crowd at a gate. At most one per gate,
  the most urgent one wins. Verbs: STOP ENTRY, HOLD OUTSIDE, DIVERT, GET READY.
* **Incidents** - send staff to a spot (person down, people walking against
  the crowd). Verb: SEND STAFF.

Two rules keep instructions consistent:

1. The decision engine is the ONLY source of where to send a crowd. A stop or
   hold at a gate that the engine is diverting carries the engine's target, so
   the operator never sees two different destinations for one gate.
2. Camera, tracking and calibration problems are not operator instructions and
   are not shown here at all.

Every item has a ``headline`` (verb + place), a ``reason`` of at most three
words, a one-line ``detail`` with the numbers, and a ``radio`` script that can
be read out to gate staff word for word.
"""

from __future__ import annotations

from dataclasses import dataclass

from crowdguard.alerts.contracts import Alert, AlertCategory, Severity
from crowdguard.decision.contracts import ActionType, CongestionLevel, CrowdState, DecisionResult

VERB_LABEL = {
    "STOP_ENTRY": "Stop entry",
    "HOLD_OUTSIDE": "Hold outside",
    "DIVERT": "Divert",
    "GET_READY": "Get ready",
    "SEND_STAFF": "Send staff",
}
_PRIORITY = {"STOP_ENTRY": 0, "HOLD_OUTSIDE": 1, "DIVERT": 2, "GET_READY": 3}
_LEVEL_FROM_SEVERITY = {
    Severity.EMERGENCY: "critical",
    Severity.ALERT: "high",
    Severity.WARNING: "watch",
    Severity.INFO: "watch",
}
_LEVEL_ORDER = {"critical": 0, "high": 1, "watch": 2}

# congestion alert rule -> (verb, reason)
_CONGESTION_RULES = {
    "CRUSH_RISK": ("STOP_ENTRY", "crush risk"),
    "GATE_OVERLOAD": ("STOP_ENTRY", "over capacity"),
    "ALL_GATES_FULL": ("HOLD_OUTSIDE", "venue full"),
    "BOTTLENECK_FORMING": ("GET_READY", "filling fast"),
    "SLOW_FLOW_WARNING": ("GET_READY", "flow slowing"),
    "MONITOR_SURGE": ("GET_READY", "inflow rising"),
    "IMBALANCED_LOAD": ("GET_READY", "uneven load"),
}
# anomaly alert rule -> (kind, reason)
_ANOMALY_RULES = {
    "PERSON_DOWN": ("person_down", "person down"),
    "COUNTERFLOW_DETECTED": ("counterflow", "walking against crowd"),
    "ANOMALY_CLUSTER": ("cluster", "many people affected"),
}
# rules that are never shown to the operator
_HIDDEN_RULES = {"DIVERT_IN_PROGRESS", "ALL_CLEAR"}


@dataclass(frozen=True)
class Instruction:
    channel: str            # "command" | "incident"
    verb: str               # STOP_ENTRY, HOLD_OUTSIDE, DIVERT, GET_READY, SEND_STAFF
    gate_id: str
    level: str              # critical | high | watch
    reason: str             # <= 3 words
    detail: str             # one line with the numbers
    radio: str              # read-out script for staff
    target: str | None = None
    kind: str | None = None
    since: float = 0.0
    share: float | None = None     # share of arrivals to divert; None = everyone

    @property
    def headline(self) -> str:
        head = f"{VERB_LABEL[self.verb]} {self.gate_id}"
        if self.target:
            head += f" → {self.target}"
        if self.verb == "DIVERT" and self.share is not None and self.share < 1:
            head += f" · {self.share:.0%}"
        return head

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "verb": self.verb,
            "verb_label": VERB_LABEL[self.verb],
            "gate_id": self.gate_id,
            "target": self.target,
            "kind": self.kind,
            "level": self.level,
            "headline": self.headline,
            "reason": self.reason,
            "detail": self.detail,
            "radio": self.radio,
            "since": self.since,
            "share": self.share,
        }


# ── wording ──────────────────────────────────────────────────────────


def _load_line(gate: str, state: CrowdState | None, capacity: int | None) -> str:
    if state is None:
        return f"{gate}: no reading"
    bits = [f"{gate}: {state.crowd_count:.0f} people"]
    if state.density_per_m2 is not None:
        bits.append(f"{state.density_per_m2:.1f}/m²")
    elif capacity:
        bits.append(f"{state.crowd_count / capacity:.0%} full")
    if state.velocity_unit == "m/s" and state.has_motion:
        bits.append(f"moving {state.velocity:.1f} m/s")
    return ", ".join(bits)


def _target_line(target: str | None, states: dict[str, CrowdState], decision: DecisionResult | None) -> str:
    if not target:
        return ""
    t = states.get(target)
    walk = ""
    if decision is not None:
        cand = next((c for c in decision.candidate_evaluations if c.gate_id == target), None)
        if cand is not None and cand.distance_m != float("inf"):
            walk = f", {cand.distance_m:.0f} m walk"
    load = f"{t.density_score:.0%} full" if t is not None else "load unknown"
    return f" · {target}: {load}{walk}"


_SHARE_WORDS = {1.0: "all", 0.75: "three in four", 0.5: "half of", 0.25: "one in four of"}


def _share_words(share: float | None) -> str:
    if share is None:
        return "all"
    return _SHARE_WORDS.get(round(share, 2), f"{share:.0%} of")


def _radio(verb: str, gate: str, target: str | None, kind: str | None = None, share: float | None = None) -> str:
    if verb == "STOP_ENTRY":
        s = f"{gate} team: stop entry now. Open barriers and make space."
        return s + (f" Send new arrivals to {target}." if target else "")
    if verb == "HOLD_OUTSIDE":
        return f"{gate} team: hold new arrivals outside. Let the queue clear."
    if verb == "DIVERT":
        who = _share_words(share)
        return f"{gate} team: send {who} new arrivals to {target} by the outer path."
    if verb == "GET_READY":
        return f"{gate} team: stand by to divert. Check the route{' to ' + target if target else ''} is clear."
    if kind == "person_down":
        return f"Staff to {gate}: person on the ground. Check on them and keep space around them."
    if kind == "counterflow":
        return f"Staff to {gate}: people walking against the crowd. Find out why and guide them out."
    return f"Staff to {gate}: several people need help. Check the area."


# ── building the lists ───────────────────────────────────────────────


class CommandBoard:
    """Turns decisions and alerts into commands and incidents.

    Remembers when each instruction first appeared, so "since" is stable while
    the same instruction stays in force.
    """

    def __init__(self) -> None:
        self._first_seen: dict[tuple, float] = {}

    def reset(self) -> None:
        self._first_seen.clear()

    def _since(self, key: tuple, now: float, live: set) -> float:
        live.add(key)
        return self._first_seen.setdefault(key, now)

    def build(
        self,
        decisions: dict[str, DecisionResult],
        alerts: list[Alert],
        states: dict[str, CrowdState],
        capacities: dict[str, int],
        now: float,
    ) -> tuple[list[Instruction], list[Instruction]]:
        live: set = set()
        candidates: dict[str, list[tuple[str, str, str]]] = {}   # gate -> [(verb, level, reason)]

        for gate, d in decisions.items():
            if d.action is ActionType.DIVERT and d.target_gate_id:
                level = "critical" if d.severity is CongestionLevel.CRITICAL else "high"
                if "FORECAST_HIGH" in d.reason_codes:
                    reason = "filling fast"
                elif "DIVERSION_HELD" in d.reason_codes:
                    reason = "still clearing"
                elif d.severity is CongestionLevel.CRITICAL:
                    reason = "critical crowding"
                else:
                    reason = "crowding"
                candidates.setdefault(gate, []).append(("DIVERT", level, reason))
            elif d.action is ActionType.RESTRICT_INFLOW:
                candidates.setdefault(gate, []).append(("HOLD_OUTSIDE", "critical", "no free gate"))

        incidents: list[Instruction] = []
        for a in alerts:
            if a.rule_id in _HIDDEN_RULES or a.category is AlertCategory.SYSTEM or not a.gate_id:
                continue
            level = _LEVEL_FROM_SEVERITY.get(a.severity, "watch")
            if a.category is AlertCategory.CONGESTION and a.rule_id in _CONGESTION_RULES:
                verb, reason = _CONGESTION_RULES[a.rule_id]
                candidates.setdefault(a.gate_id, []).append((verb, level, reason))
            elif a.category is AlertCategory.ANOMALY and a.rule_id in _ANOMALY_RULES:
                kind, reason = _ANOMALY_RULES[a.rule_id]
                count = a.detail.get("sitting_count") if kind == "person_down" else a.detail.get("opposite_count")
                if kind == "person_down" and count and count > 1:
                    reason = f"{count} people down"
                key = ("incident", a.gate_id, kind)
                incidents.append(Instruction(
                    channel="incident", verb="SEND_STAFF", gate_id=a.gate_id, level=level,
                    reason=reason, kind=kind,
                    detail=a.title,
                    radio=_radio("SEND_STAFF", a.gate_id, None, kind),
                    since=self._since(key, a.first_seen or now, live),
                ))

        commands: list[Instruction] = []
        for gate, options in candidates.items():
            verb, level, reason = min(options, key=lambda o: (_PRIORITY[o[0]], _LEVEL_ORDER[o[1]]))
            # escalate the level if any option for this gate is more urgent
            level = min((o[1] for o in options), key=lambda lv: _LEVEL_ORDER[lv])
            decision = decisions.get(gate)
            target = None
            if decision is not None and decision.action is ActionType.DIVERT and verb in (
                    "STOP_ENTRY", "DIVERT", "GET_READY"):
                target = decision.target_gate_id
            share = decision.divert_fraction if (decision is not None and verb == "DIVERT") else None
            key = ("command", gate, verb, target, share)
            commands.append(Instruction(
                channel="command", verb=verb, gate_id=gate, level=level, reason=reason, target=target,
                detail=_load_line(gate, states.get(gate), capacities.get(gate))
                + _target_line(target, states, decision),
                radio=_radio(verb, gate, target, share=share),
                since=self._since(key, now, live),
                share=share,
            ))

        # forget instructions that are no longer in force
        for key in list(self._first_seen):
            if key not in live:
                del self._first_seen[key]

        commands.sort(key=lambda c: (_PRIORITY[c.verb], _LEVEL_ORDER[c.level], c.since))
        incidents.sort(key=lambda c: (_LEVEL_ORDER[c.level], c.since))
        return commands, incidents


def present_alert(alert: Alert) -> dict | None:
    """One log line for an alert in the operator's vocabulary, or None if the
    alert is not something an operator acts on."""
    if alert.category is AlertCategory.SYSTEM or alert.rule_id == "ALL_CLEAR" or not alert.gate_id:
        return None
    if alert.rule_id == "DIVERT_IN_PROGRESS":
        verb, reason, kind = "DIVERT", "crowding", None
    elif alert.rule_id in _CONGESTION_RULES:
        (verb, reason), kind = _CONGESTION_RULES[alert.rule_id], None
    elif alert.rule_id in _ANOMALY_RULES:
        kind, reason = _ANOMALY_RULES[alert.rule_id]
        verb = "SEND_STAFF"
    else:
        return None
    target = alert.target_gate if verb in ("DIVERT", "STOP_ENTRY") else None
    head = f"{VERB_LABEL[verb]} {alert.gate_id}" + (f" → {target}" if target else "")
    return {
        "verb": verb,
        "verb_label": VERB_LABEL[verb],
        "gate_id": alert.gate_id,
        "target": target,
        "kind": kind,
        "level": _LEVEL_FROM_SEVERITY.get(alert.severity, "watch"),
        "headline": head,
        "reason": reason,
        "time": alert.first_seen or alert.timestamp,
        "last_seen": alert.timestamp,
        "occurrences": alert.occurrences,
    }
