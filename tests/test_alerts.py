"""Alert rules and the alert store."""

from __future__ import annotations

import time


from conftest import (
    congested_flowing_state,
    make_state,
    make_venue,
    quiet_state,
    saturated_state,
)

from crowdguard.alerts.contracts import Alert, AlertCategory, Severity
from crowdguard.alerts.rules import GateView, generate_alerts
from crowdguard.alerts.store import AlertStore
from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.engine import DecisionEngine
from crowdguard.models.contracts import AnomalyLabel, AnomalyResult, PersonObservation


def make_anomaly(
    camera_id: str = "cam_1",
    sitting: int = 0,
    opposite: int = 0,
    tracked: int = 10,
    crowd_valid: bool = True,
    continuity_ok: bool = True,
) -> AnomalyResult:
    """Build an AnomalyResult with matching PersonObservation entries."""
    people: list[PersonObservation] = []
    track_id = 1

    for _ in range(sitting):
        people.append(PersonObservation(
            track_id=track_id, bbox=(0.0, 0.0, 50.0, 60.0),
            label=AnomalyLabel.SITTING, frames_seen=20, speed_px_per_frame=0.2,
            direction_rad=0.0, cos_sim_to_crowd=0.0, confirmed=True,
        ))
        track_id += 1

    for _ in range(opposite):
        people.append(PersonObservation(
            track_id=track_id, bbox=(0.0, 0.0, 30.0, 90.0),
            label=AnomalyLabel.OPPOSITE, frames_seen=20, speed_px_per_frame=8.0,
            direction_rad=3.14, cos_sim_to_crowd=-0.9, confirmed=True,
        ))
        track_id += 1

    return AnomalyResult(
        camera_id=camera_id,
        frame_id=100,
        source_frame_index=100,
        video_timestamp=4.0,
        wall_clock_timestamp=time.time(),
        people=tuple(people),
        sitting_count=sitting,
        opposite_count=opposite,
        tracked_count=tracked,
        crowd_direction_rad=0.0,
        crowd_direction_valid=crowd_valid,
        classifier_active=True,
        continuity_ok=continuity_ok,
    )


def rule_ids(alerts: list[Alert]) -> set[str]:
    return {a.rule_id for a in alerts}


class TestCongestionRules:
    def test_crush_risk_on_dense_and_static(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=450, velocity=1.0, config=config),
        )
        alerts = generate_alerts([view])
        crush = next(a for a in alerts if a.rule_id == "CRUSH_RISK")
        assert crush.severity is Severity.EMERGENCY
        assert crush.category is AlertCategory.CONGESTION

    def test_overload_when_over_capacity(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=100,
            state=make_state("G1", crowd_count=130, capacity=100, velocity=12.0, config=config),
        )
        assert "GATE_OVERLOAD" in rule_ids(generate_alerts([view]))

    def test_quiet_gate_produces_only_all_clear(self):
        config = DecisionConfig()
        view = GateView(gate_id="G1", capacity=500, state=quiet_state("G1", config))
        alerts = generate_alerts([view])
        assert rule_ids(alerts) == {"ALL_CLEAR"}
        assert alerts[0].severity is Severity.INFO

    def test_one_congestion_alert_per_gate(self):
        """The congestion chain is exclusive: no stacking on one gate."""
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=100,
            state=make_state("G1", crowd_count=150, capacity=100, velocity=0.5, config=config),
        )
        congestion = [
            a for a in generate_alerts([view])
            if a.category is AlertCategory.CONGESTION
        ]
        assert len(congestion) == 1

    def test_restrict_inflow_decision_raises_all_gates_full(self):
        config = DecisionConfig()
        venue = make_venue(2)
        # Flowing but full, so the crush rule does not mask the routing rule.
        states = {g: congested_flowing_state(g, config) for g in ("G1", "G2")}
        decisions = {d.source_gate_id: d for d in DecisionEngine(venue, config).evaluate(states)}

        view = GateView(
            gate_id="G1", capacity=500,
            state=states["G1"], decision=decisions["G1"],
        )
        alerts = generate_alerts([view])
        assert "ALL_GATES_FULL" in rule_ids(alerts)

    def test_active_diversion_is_announced_with_the_route(self):
        config = DecisionConfig()
        venue = make_venue(3)
        states = {
            "G1": congested_flowing_state("G1", config),
            "G2": quiet_state("G2", config),
            "G3": congested_flowing_state("G3", config),
        }
        decisions = {d.source_gate_id: d for d in DecisionEngine(venue, config).evaluate(states)}

        view = GateView(
            gate_id="G1", capacity=500,
            state=states["G1"], decision=decisions["G1"],
        )
        divert = next(a for a in generate_alerts([view]) if a.rule_id == "DIVERT_IN_PROGRESS")
        assert divert.target_gate == "G2"
        assert "G2" in divert.action

    def test_crush_risk_outranks_the_diversion_notice(self):
        """A gate that is both crushing and diverting must report the crush.

        The congestion chain is exclusive and ordered by urgency. The diversion
        target is still carried on the crush alert, so nothing is lost.
        """
        config = DecisionConfig()
        venue = make_venue(3)
        states = {
            "G1": saturated_state("G1", config),      # dense AND static
            "G2": quiet_state("G2", config),
            "G3": saturated_state("G3", config),
        }
        decisions = {d.source_gate_id: d for d in DecisionEngine(venue, config).evaluate(states)}

        view = GateView(
            gate_id="G1", capacity=500,
            state=states["G1"], decision=decisions["G1"],
        )
        alerts = generate_alerts([view])
        congestion = [a for a in alerts if a.category is AlertCategory.CONGESTION]

        assert [a.rule_id for a in congestion] == ["CRUSH_RISK"]
        assert congestion[0].target_gate == "G2"

    def test_load_imbalance_across_gates(self):
        config = DecisionConfig()
        views = [
            GateView(gate_id="G1", capacity=500,
                     state=make_state("G1", crowd_count=200, velocity=11.0, config=config)),
            GateView(gate_id="G2", capacity=500,
                     state=make_state("G2", crowd_count=10, velocity=13.0, config=config)),
            GateView(gate_id="G3", capacity=500,
                     state=make_state("G3", crowd_count=12, velocity=13.0, config=config)),
        ]
        imbalance = next(
            a for a in generate_alerts(views) if a.rule_id == "IMBALANCED_LOAD"
        )
        assert imbalance.gate_id == "G1"
        assert imbalance.target_gate in {"G2", "G3"}

    def test_imbalance_suppressed_when_gate_already_critical(self):
        """Don't add a soft notice to a gate already flagged as an emergency."""
        config = DecisionConfig()
        views = [
            GateView(gate_id="G1", capacity=100,
                     state=make_state("G1", crowd_count=150, capacity=100,
                                      velocity=0.5, config=config)),
            GateView(gate_id="G2", capacity=500,
                     state=make_state("G2", crowd_count=5, velocity=13.0, config=config)),
            GateView(gate_id="G3", capacity=500,
                     state=make_state("G3", crowd_count=6, velocity=13.0, config=config)),
        ]
        alerts = generate_alerts(views)
        g1_rules = {a.rule_id for a in alerts if a.gate_id == "G1"}
        assert "IMBALANCED_LOAD" not in g1_rules


class TestAnomalyRules:
    def test_person_down_in_a_moving_crowd_is_an_emergency(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=150, velocity=12.0, config=config),
            anomaly=make_anomaly(sitting=1, tracked=12),
        )
        alert = next(a for a in generate_alerts([view]) if a.rule_id == "PERSON_DOWN")
        assert alert.severity is Severity.EMERGENCY
        assert alert.category is AlertCategory.ANOMALY
        assert alert.detail["track_ids"] == [1]

    def test_person_down_in_a_calm_area_is_only_a_warning(self):
        """Severity must reflect the actual hazard, not just the detection."""
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=20, velocity=1.0, config=config),
            anomaly=make_anomaly(sitting=1, tracked=4),
        )
        alert = next(a for a in generate_alerts([view]) if a.rule_id == "PERSON_DOWN")
        assert alert.severity is Severity.WARNING

    def test_person_down_at_a_congested_gate_is_an_emergency(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=460, velocity=1.0, config=config),
            anomaly=make_anomaly(sitting=2, tracked=30),
        )
        alert = next(a for a in generate_alerts([view]) if a.rule_id == "PERSON_DOWN")
        assert alert.severity is Severity.EMERGENCY

    def test_counterflow_escalates_with_numbers(self):
        config = DecisionConfig()
        base_state = make_state("G1", crowd_count=120, velocity=11.0, config=config)

        few = GateView(gate_id="G1", capacity=500, state=base_state,
                       anomaly=make_anomaly(opposite=2, tracked=20))
        many = GateView(gate_id="G1", capacity=500, state=base_state,
                        anomaly=make_anomaly(opposite=6, tracked=20))

        few_alert = next(a for a in generate_alerts([few]) if a.rule_id == "COUNTERFLOW_DETECTED")
        many_alert = next(a for a in generate_alerts([many]) if a.rule_id == "COUNTERFLOW_DETECTED")

        assert few_alert.severity is Severity.WARNING
        assert many_alert.severity is Severity.ALERT

    def test_single_counterflow_person_does_not_alert(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=120, velocity=11.0, config=config),
            anomaly=make_anomaly(opposite=1, tracked=20),
        )
        assert "COUNTERFLOW_DETECTED" not in rule_ids(generate_alerts([view]))

    def test_counterflow_needs_a_valid_crowd_direction(self):
        """Without an established flow there is nothing to be counter to."""
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=120, velocity=11.0, config=config),
            anomaly=make_anomaly(opposite=5, tracked=20, crowd_valid=False),
        )
        assert "COUNTERFLOW_DETECTED" not in rule_ids(generate_alerts([view]))

    def test_widespread_anomalies_raise_a_cluster_alert(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=100, velocity=11.0, config=config),
            anomaly=make_anomaly(sitting=2, opposite=3, tracked=10),
        )
        cluster = next(a for a in generate_alerts([view]) if a.rule_id == "ANOMALY_CLUSTER")
        assert cluster.severity is Severity.ALERT

    def test_anomaly_and_congestion_alerts_coexist(self):
        """They demand different responses, so both must reach the operator."""
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=450, velocity=1.0, config=config),
            anomaly=make_anomaly(sitting=1, tracked=25),
        )
        ids = rule_ids(generate_alerts([view]))
        assert "CRUSH_RISK" in ids
        assert "PERSON_DOWN" in ids

    def test_no_anomaly_result_means_no_anomaly_alerts(self):
        config = DecisionConfig()
        view = GateView(gate_id="G1", capacity=500, state=quiet_state("G1", config))
        anomaly_alerts = [
            a for a in generate_alerts([view]) if a.category is AlertCategory.ANOMALY
        ]
        assert anomaly_alerts == []


class TestSystemRules:
    def test_degraded_tracking_is_surfaced(self):
        """Degraded detection must never look like a quiet venue."""
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=quiet_state("G1", config),
            camera_id="cam_1",
            continuity_break_ratio=0.5,
        )
        alert = next(a for a in generate_alerts([view]) if a.rule_id == "TRACKING_DEGRADED")
        assert alert.category is AlertCategory.SYSTEM
        assert "anomaly_stride" in alert.action

    def test_missing_motion_is_reported_for_monitored_gates(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=100, has_motion=False, config=config),
            camera_id="cam_1",
        )
        assert "MOTION_UNAVAILABLE" in rule_ids(generate_alerts([view]))

    def test_unmonitored_gate_does_not_report_missing_motion(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G5", capacity=500,
            state=make_state("G5", crowd_count=0, has_motion=False, config=config),
            camera_id=None, is_monitored=False,
        )
        assert "MOTION_UNAVAILABLE" not in rule_ids(generate_alerts([view]))

    def test_empty_views_produce_nothing(self):
        assert generate_alerts([]) == []


class TestOrdering:
    def test_most_urgent_first(self):
        config = DecisionConfig()
        views = [
            GateView(gate_id="G1", capacity=500,
                     state=make_state("G1", crowd_count=100, velocity=11.0, config=config),
                     anomaly=make_anomaly(opposite=2, tracked=20)),
            GateView(gate_id="G2", capacity=500,
                     state=make_state("G2", crowd_count=460, velocity=0.5, config=config)),
        ]
        alerts = generate_alerts(views)
        assert alerts[0].severity is Severity.EMERGENCY


class TestAlertStore:
    def _alert(self, rule_id: str = "CRUSH_RISK", gate_id: str = "G1",
               severity: Severity = Severity.EMERGENCY) -> Alert:
        return Alert(
            rule_id=rule_id, severity=severity, category=AlertCategory.CONGESTION,
            gate_id=gate_id, title="t", description="d", action="a",
        )

    def test_first_submission_is_newly_raised(self):
        store = AlertStore()
        assert len(store.submit([self._alert()])) == 1
        assert len(store.active()) == 1

    def test_repeat_is_deduplicated_and_counted(self):
        store = AlertStore()
        store.submit([self._alert()])
        assert store.submit([self._alert()]) == []

        active = store.active()
        assert len(active) == 1
        assert active[0].occurrences == 2

    def test_first_seen_is_preserved_across_repeats(self):
        store = AlertStore()
        store.submit([self._alert()])
        original = store.active()[0].first_seen

        time.sleep(0.02)
        store.submit([self._alert()])
        assert store.active()[0].first_seen == original

    def test_escalation_is_recorded_as_new(self):
        store = AlertStore()
        store.submit([self._alert(severity=Severity.WARNING)])
        raised = store.submit([self._alert(severity=Severity.EMERGENCY)])

        assert len(raised) == 1
        assert store.active()[0].severity is Severity.EMERGENCY

    def test_de_escalation_updates_without_a_new_incident(self):
        store = AlertStore()
        store.submit([self._alert(severity=Severity.EMERGENCY)])
        raised = store.submit([self._alert(severity=Severity.WARNING)])

        assert raised == []
        assert store.active()[0].severity is Severity.WARNING

    def test_alerts_expire_once_the_condition_stops(self):
        store = AlertStore(resolve_after_s=0.05)
        store.submit([self._alert()])
        assert len(store.active()) == 1

        time.sleep(0.08)
        store.expire_stale()
        assert store.active() == []
        assert store.total_resolved == 1

    def test_same_rule_on_different_gates_is_separate(self):
        store = AlertStore()
        store.submit([self._alert(gate_id="G1"), self._alert(gate_id="G2")])
        assert len(store.active()) == 2

    def test_history_survives_expiry(self):
        store = AlertStore(resolve_after_s=0.05)
        store.submit([self._alert()])
        time.sleep(0.08)
        store.expire_stale()

        assert store.active() == []
        assert len(store.history()) == 1

    def test_counts_by_severity(self):
        store = AlertStore()
        store.submit([
            self._alert(rule_id="A", severity=Severity.EMERGENCY),
            self._alert(rule_id="B", severity=Severity.WARNING),
            self._alert(rule_id="C", severity=Severity.WARNING),
        ])
        counts = store.counts_by_severity()
        assert counts["EMERGENCY"] == 1
        assert counts["WARNING"] == 2

    def test_worst_severity(self):
        store = AlertStore()
        assert store.worst_severity() is None

        store.submit([
            self._alert(rule_id="A", severity=Severity.WARNING),
            self._alert(rule_id="B", severity=Severity.EMERGENCY),
        ])
        assert store.worst_severity() is Severity.EMERGENCY

    def test_filter_by_gate_and_category(self):
        store = AlertStore()
        anomaly = Alert(
            rule_id="PERSON_DOWN", severity=Severity.EMERGENCY,
            category=AlertCategory.ANOMALY, gate_id="G2",
            title="t", description="d", action="a",
        )
        store.submit([self._alert(gate_id="G1"), anomaly])

        assert len(store.active_for_gate("G2")) == 1
        assert len(store.active_by_category(AlertCategory.ANOMALY)) == 1

    def test_persists_to_jsonl(self, tmp_path):
        import json

        log_file = tmp_path / "alerts.jsonl"
        store = AlertStore(log_path=log_file)
        store.submit([self._alert()])

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["rule_id"] == "CRUSH_RISK"

    def test_all_clear_is_dropped_next_to_a_live_alert(self):
        """An all-clear must never sit alongside a real alert.

        Anomaly detections fluctuate between cycles, so a quiet cycle can raise
        ALL_CLEAR moments before a genuine alert lands, leaving both inside the
        resolve window. Showing "all gates within safe limits" above "person on
        the ground" is actively misleading.
        """
        store = AlertStore()
        all_clear = Alert(
            rule_id="ALL_CLEAR", severity=Severity.INFO, category=AlertCategory.SYSTEM,
            gate_id="ALL", title="All gates within safe limits",
            description="d", action="a",
        )
        store.submit([all_clear])
        assert {a.rule_id for a in store.active()} == {"ALL_CLEAR"}

        store.submit([self._alert(rule_id="PERSON_DOWN", severity=Severity.EMERGENCY)])

        rule_ids_active = {a.rule_id for a in store.active()}
        assert "ALL_CLEAR" not in rule_ids_active
        assert "PERSON_DOWN" in rule_ids_active

    def test_all_clear_survives_alongside_info_only_alerts(self):
        """An INFO notice is not a reason to hide the all-clear."""
        store = AlertStore()
        all_clear = Alert(
            rule_id="ALL_CLEAR", severity=Severity.INFO, category=AlertCategory.SYSTEM,
            gate_id="ALL", title="t", description="d", action="a",
        )
        info = Alert(
            rule_id="MOTION_UNAVAILABLE", severity=Severity.INFO,
            category=AlertCategory.SYSTEM, gate_id="G1",
            title="t", description="d", action="a",
        )
        store.submit([all_clear, info])
        assert "ALL_CLEAR" in {a.rule_id for a in store.active()}

    def test_repeats_are_not_written_again(self):
        """The log records incidents, not evaluation ticks."""
        store = AlertStore()
        store.submit([self._alert()])
        store.submit([self._alert()])
        store.submit([self._alert()])
        assert len(store.history()) == 1


class TestVelocityCalibration:
    def test_saturated_velocity_term_is_flagged(self):
        """A pinned velocity term must be visible, not silently density-only."""
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=100, velocity=90.0, config=config),
            camera_id="cam_1",
            max_expected_velocity=75.0,
        )
        alert = next(
            a for a in generate_alerts([view]) if a.rule_id == "VELOCITY_UNCALIBRATED"
        )
        assert alert.severity is Severity.INFO
        assert alert.category is AlertCategory.SYSTEM
        assert alert.detail["velocity"] == 90.0

    def test_velocity_within_scale_is_not_flagged(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G1", capacity=500,
            state=make_state("G1", crowd_count=100, velocity=30.0, config=config),
            camera_id="cam_1",
            max_expected_velocity=75.0,
        )
        assert "VELOCITY_UNCALIBRATED" not in rule_ids(generate_alerts([view]))

    def test_unmonitored_gate_is_not_flagged(self):
        config = DecisionConfig()
        view = GateView(
            gate_id="G5", capacity=500,
            state=make_state("G5", crowd_count=0, has_motion=False, config=config),
            camera_id=None, is_monitored=False,
        )
        assert "VELOCITY_UNCALIBRATED" not in rule_ids(generate_alerts([view]))
