"""Operator instructions: fixed vocabulary, one command per gate, one destination."""

from __future__ import annotations

import time

from conftest import make_state, make_venue

from crowdguard.alerts.commands import CommandBoard, present_alert
from crowdguard.alerts.contracts import Alert, AlertCategory, Severity
from crowdguard.decision.contracts import (
    ActionType,
    CandidateEvaluation,
    CongestionLevel,
    DecisionResult,
    Trend,
)

ALLOWED_VERBS = {"STOP_ENTRY", "HOLD_OUTSIDE", "DIVERT", "GET_READY", "SEND_STAFF"}


def decision(gate, action, target=None, severity=CongestionLevel.HIGH):
    cands = ()
    if target:
        cands = (CandidateEvaluation(target, 94.2, CongestionLevel.LOW, 0.2, 0.3, Trend.STABLE, 0.8, True),)
    return DecisionResult(decision_id="d", timestamp=time.time(), source_gate_id=gate, action=action,
                          target_gate_id=target, severity=severity, candidate_evaluations=cands)


def alert(rule, gate, severity=Severity.WARNING, category=AlertCategory.CONGESTION, **detail):
    return Alert(rule_id=rule, severity=severity, category=category, gate_id=gate, title=f"{rule} at {gate}",
                 description="long description " * 10, action="long action " * 10, detail=detail)


def build(decisions=None, alerts=None):
    states = {g: make_state(g, crowd_count=300) for g in ("G1", "G2", "G3", "G4")}
    return CommandBoard().build(decisions or {}, alerts or [], states, {g: 500 for g in states}, time.time())


def test_divert_becomes_a_short_command():
    commands, incidents = build({"G1": decision("G1", ActionType.DIVERT, "G3")})
    assert len(commands) == 1 and not incidents
    c = commands[0].to_dict()
    assert c["verb"] == "DIVERT" and c["target"] == "G3"
    assert c["headline"] == "Divert G1 → G3"
    assert c["radio"] == "G1 team: send new arrivals to G3 by the outer path."
    assert "94 m walk" in c["detail"]


def test_one_command_per_gate_most_urgent_wins():
    commands, _ = build(
        {"G1": decision("G1", ActionType.DIVERT, "G3")},
        [alert("BOTTLENECK_FORMING", "G1", Severity.ALERT), alert("CRUSH_RISK", "G1", Severity.EMERGENCY)],
    )
    assert len(commands) == 1
    assert commands[0].verb == "STOP_ENTRY" and commands[0].level == "critical"


def test_stop_entry_carries_the_engine_destination_only():
    """The crush rule used to say 'nearest alternative'; now the engine's target is the only one."""
    commands, _ = build({"G1": decision("G1", ActionType.DIVERT, "G4")},
                        [alert("CRUSH_RISK", "G1", Severity.EMERGENCY)])
    assert commands[0].target == "G4"
    assert "Send new arrivals to G4" in commands[0].radio


def test_restrict_becomes_hold_outside():
    commands, _ = build({"G2": decision("G2", ActionType.RESTRICT_INFLOW)})
    assert commands[0].verb == "HOLD_OUTSIDE" and commands[0].reason == "no free gate"


def test_anomalies_become_incidents_not_commands():
    commands, incidents = build(alerts=[
        alert("PERSON_DOWN", "G2", Severity.EMERGENCY, AlertCategory.ANOMALY, sitting_count=2),
        alert("COUNTERFLOW_DETECTED", "G3", Severity.WARNING, AlertCategory.ANOMALY, opposite_count=3),
    ])
    assert not commands
    assert [i.verb for i in incidents] == ["SEND_STAFF", "SEND_STAFF"]
    assert incidents[0].reason == "2 people down"          # critical first
    assert incidents[1].kind == "counterflow"


def test_system_and_duplicate_alerts_are_hidden():
    commands, incidents = build(alerts=[
        alert("TRACKING_DEGRADED", "G1", Severity.WARNING, AlertCategory.SYSTEM),
        alert("VELOCITY_UNCALIBRATED", "G2", Severity.INFO, AlertCategory.SYSTEM),
        alert("DIVERT_IN_PROGRESS", "G3", Severity.ALERT),
        alert("ALL_CLEAR", "ALL", Severity.INFO),
    ])
    assert commands == [] and incidents == []


def test_vocabulary_and_length_limits():
    commands, incidents = build(
        {"G1": decision("G1", ActionType.DIVERT, "G3"), "G2": decision("G2", ActionType.RESTRICT_INFLOW)},
        [alert(r, "G3") for r in ("SLOW_FLOW_WARNING", "MONITOR_SURGE")]
        + [alert("PERSON_DOWN", "G4", Severity.EMERGENCY, AlertCategory.ANOMALY, sitting_count=1)],
    )
    for item in commands + incidents:
        assert item.verb in ALLOWED_VERBS
        assert len(item.reason.split()) <= 3, item.reason
        assert len(item.headline.split()) <= 5, item.headline
        assert len(item.radio.split()) <= 20, item.radio


def test_since_is_stable_while_the_command_holds():
    board = CommandBoard()
    states = {"G1": make_state("G1", crowd_count=300)}
    d = {"G1": decision("G1", ActionType.DIVERT, "G3")}
    first, _ = board.build(d, [], states, {"G1": 500}, now=100.0)
    later, _ = board.build(d, [], states, {"G1": 500}, now=160.0)
    assert first[0].since == later[0].since == 100.0
    changed, _ = board.build({"G1": decision("G1", ActionType.DIVERT, "G2")}, [], states, {"G1": 500}, now=170.0)
    assert changed[0].since == 170.0


def test_log_uses_the_same_vocabulary():
    entry = present_alert(alert("CRUSH_RISK", "G1", Severity.EMERGENCY))
    assert entry["verb"] == "STOP_ENTRY" and entry["reason"] == "crush risk"
    assert present_alert(alert("TRACKING_DEGRADED", "G1", Severity.WARNING, AlertCategory.SYSTEM)) is None


def test_engine_and_board_agree_end_to_end():
    """A real engine decision flows into a command with the same destination."""
    from crowdguard.decision.engine import DecisionEngine
    venue = make_venue(4)
    states = {"G1": make_state("G1", crowd_count=480, velocity=8.0)}
    states.update({g: make_state(g, crowd_count=60) for g in ("G2", "G3", "G4")})
    decisions = {d.source_gate_id: d for d in DecisionEngine(venue).evaluate(states)}
    commands, _ = CommandBoard().build(decisions, [], states, {g: 500 for g in states}, time.time())
    assert len(commands) == 1
    assert commands[0].target == decisions["G1"].target_gate_id
