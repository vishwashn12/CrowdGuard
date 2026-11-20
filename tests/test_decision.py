"""Decision engine: congestion scoring, safety refusals, routing, hysteresis."""

from __future__ import annotations

import time

import pytest

from conftest import make_state, make_venue, quiet_state, saturated_state

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import (
    ActionType,
    CongestionLevel,
    CrowdState,
    MotionStatus,
    Trend,
    level_at_least,
)
from crowdguard.decision.engine import DecisionEngine
from crowdguard.decision.fusion import (
    build_crowd_state,
    build_unmonitored_state,
    classify_congestion,
    compute_congestion_score,
)
from crowdguard.decision.routing import rank_candidates, trend_bonus
from crowdguard.decision.trend import TrendTracker


class TestContractValidation:
    def test_rejects_out_of_range_density(self):
        with pytest.raises(ValueError, match="density_score"):
            CrowdState(gate_id="G1", timestamp=time.time(), density_score=1.5)

    def test_rejects_negative_velocity(self):
        with pytest.raises(ValueError, match="velocity"):
            CrowdState(gate_id="G1", timestamp=time.time(), velocity=-1.0)

    def test_rejects_empty_gate_id(self):
        with pytest.raises(ValueError, match="gate_id"):
            CrowdState(gate_id="", timestamp=time.time())

    def test_rejects_direction_out_of_range(self):
        with pytest.raises(ValueError, match="direction"):
            CrowdState(gate_id="G1", timestamp=time.time(), direction=360.0)

    def test_level_ordering(self):
        assert level_at_least(CongestionLevel.CRITICAL, "HIGH")
        assert level_at_least(CongestionLevel.HIGH, "HIGH")
        assert not level_at_least(CongestionLevel.MODERATE, "HIGH")


class TestCongestionScoring:
    def test_empty_gate_scores_low(self):
        config = DecisionConfig()
        state = build_crowd_state("G1", 500, crowd_count=0, velocity=12.0, config=config)
        assert state.congestion_level is CongestionLevel.LOW

    def test_full_and_static_scores_critical(self):
        config = DecisionConfig()
        state = build_crowd_state("G1", 500, crowd_count=500, velocity=0.0, config=config)
        assert state.congestion_level is CongestionLevel.CRITICAL

    def test_density_score_clamps_above_capacity(self):
        state = build_crowd_state("G1", 100, crowd_count=250)
        assert state.density_score == 1.0

    def test_faster_crowd_scores_lower_than_slower(self):
        config = DecisionConfig()
        slow = build_crowd_state("G1", 500, crowd_count=300, velocity=0.0, config=config)
        fast = build_crowd_state("G1", 500, crowd_count=300, velocity=15.0, config=config)
        assert fast.congestion_score < slow.congestion_score

    def test_thresholds_map_to_levels(self):
        config = DecisionConfig()
        assert classify_congestion(0.1, config) is CongestionLevel.LOW
        assert classify_congestion(0.5, config) is CongestionLevel.MODERATE
        assert classify_congestion(0.7, config) is CongestionLevel.HIGH
        assert classify_congestion(0.95, config) is CongestionLevel.CRITICAL

    def test_score_always_within_unit_range(self):
        config = DecisionConfig()
        for count in (0, 1, 250, 500, 5000):
            for velocity in (0.0, 5.0, 100.0):
                score = compute_congestion_score(
                    density_score=min(1.0, count / 500),
                    velocity=velocity,
                    capacity_utilization=min(1.0, count / 500),
                    config=config,
                )
                assert 0.0 <= score <= 1.0


class TestMissingMotion:
    """Motion gaps must neither invent congestion nor invent free flow."""

    def test_missing_motion_drops_the_velocity_term(self):
        config = DecisionConfig()
        with_motion_still = build_crowd_state(
            "G1", 500, crowd_count=250, velocity=0.0,
            has_motion=True, motion_status=MotionStatus.OK, config=config,
        )
        without_motion = build_crowd_state(
            "G1", 500, crowd_count=250,
            has_motion=False, motion_status=MotionStatus.UNAVAILABLE, config=config,
        )
        # A zero-velocity reading is worse than no reading at all.
        assert without_motion.congestion_score < with_motion_still.congestion_score

    def test_missing_motion_zeroes_motion_fields(self):
        state = build_crowd_state(
            "G1", 500, crowd_count=100, velocity=25.0, direction=180.0,
            flow_consistency=0.9, has_motion=False,
            motion_status=MotionStatus.UNAVAILABLE,
        )
        assert state.velocity == 0.0
        assert state.direction == 0.0
        assert state.flow_consistency == 0.0
        assert state.has_motion is False

    def test_stale_motion_is_treated_as_unusable(self):
        state = build_crowd_state(
            "G1", 500, crowd_count=100, velocity=20.0,
            has_motion=True, motion_status=MotionStatus.STALE,
        )
        assert state.has_motion is False
        assert state.velocity == 0.0

    def test_partial_observation_lowers_confidence(self):
        full = build_crowd_state("G1", 500, crowd_count=100, confidence=1.0)
        no_motion = build_crowd_state(
            "G1", 500, crowd_count=100, confidence=1.0,
            has_motion=False, motion_status=MotionStatus.UNAVAILABLE,
        )
        no_density = build_crowd_state(
            "G1", 500, crowd_count=0, confidence=1.0, has_density=False,
        )
        assert no_motion.confidence < full.confidence
        assert no_density.confidence < full.confidence


class TestUnmonitoredGates:
    def test_unmonitored_state_is_not_assumed_empty(self):
        """An unseen gate must not look like the most attractive target."""
        config = DecisionConfig()
        state = build_unmonitored_state("G9", 500, config=config)
        assert state.density_score == pytest.approx(config.assumed_unmonitored_utilization)
        assert state.crowd_count > 0

    def test_unmonitored_state_declares_both_halves_missing(self):
        state = build_unmonitored_state("G9", 500)
        assert state.has_density is False
        assert state.has_motion is False
        assert state.motion_status is MotionStatus.UNAVAILABLE
        assert state.confidence < 1.0


class TestTrendTracker:
    def test_rising_scores_give_increasing(self):
        tracker = TrendTracker(DecisionConfig())
        for score in (0.1, 0.2, 0.4, 0.6, 0.8):
            tracker.update("G1", score)
        assert tracker.get_trend("G1") is Trend.INCREASING

    def test_falling_scores_give_decreasing(self):
        tracker = TrendTracker(DecisionConfig())
        for score in (0.8, 0.6, 0.4, 0.2, 0.1):
            tracker.update("G1", score)
        assert tracker.get_trend("G1") is Trend.DECREASING

    def test_flat_scores_give_stable(self):
        tracker = TrendTracker(DecisionConfig())
        for _ in range(5):
            tracker.update("G1", 0.5)
        assert tracker.get_trend("G1") is Trend.STABLE

    def test_single_observation_is_stable(self):
        tracker = TrendTracker(DecisionConfig())
        assert tracker.update("G1", 0.5) is Trend.STABLE

    def test_gates_are_tracked_independently(self):
        tracker = TrendTracker(DecisionConfig())
        for score in (0.1, 0.3, 0.5, 0.7, 0.9):
            tracker.update("G1", score)
        for score in (0.9, 0.7, 0.5, 0.3, 0.1):
            tracker.update("G2", score)
        assert tracker.get_trend("G1") is Trend.INCREASING
        assert tracker.get_trend("G2") is Trend.DECREASING

    def test_reset_clears_one_gate_only(self):
        tracker = TrendTracker(DecisionConfig())
        for score in (0.1, 0.4, 0.8):
            tracker.update("G1", score)
            tracker.update("G2", score)
        tracker.reset("G1")
        assert tracker.get_trend("G1") is Trend.STABLE
        assert tracker.get_trend("G2") is Trend.INCREASING


class TestRouting:
    def test_trend_bonus_prefers_clearing_gates(self):
        assert trend_bonus(Trend.DECREASING) > trend_bonus(Trend.STABLE)
        assert trend_bonus(Trend.STABLE) > trend_bonus(Trend.INCREASING)

    def test_quiet_nearby_gate_outranks_busy_far_one(self):
        venue = make_venue(4)
        config = DecisionConfig()
        states = {
            "G1": saturated_state("G1", config),
            "G2": quiet_state("G2", config),
            "G3": make_state("G3", crowd_count=300, velocity=3.0, config=config),
            "G4": make_state("G4", crowd_count=200, velocity=8.0, config=config),
        }
        ranked = rank_candidates("G1", states, venue, config)
        assert ranked[0].gate_id == "G2"
        assert ranked[0].is_viable

    def test_congested_candidate_is_rejected_with_a_reason(self):
        venue = make_venue(2)
        config = DecisionConfig()
        states = {
            "G1": saturated_state("G1", config),
            "G2": saturated_state("G2", config),
        }
        ranked = rank_candidates("G1", states, venue, config)
        rejected = next(c for c in ranked if c.gate_id == "G2")
        assert not rejected.is_viable
        assert "Congestion" in (rejected.rejection_reason or "")

    def test_gate_with_no_state_is_rejected_not_dropped(self):
        venue = make_venue(3)
        config = DecisionConfig()
        states = {"G1": saturated_state("G1", config), "G2": quiet_state("G2", config)}
        ranked = rank_candidates("G1", states, venue, config)
        ids = {c.gate_id for c in ranked}
        assert ids == {"G2", "G3"}
        g3 = next(c for c in ranked if c.gate_id == "G3")
        assert not g3.is_viable
        assert "No crowd state" in (g3.rejection_reason or "")

    def test_unmonitored_target_is_penalised_but_allowed(self):
        config = DecisionConfig()
        venue = make_venue(3, monitored=2)

        monitored = quiet_state("G2", config)
        # Same numbers, but on the gate that has no camera.
        unmonitored = build_crowd_state(
            "G3", 500, crowd_count=40.0, velocity=14.0, config=config,
        )
        states = {"G1": saturated_state("G1", config), "G2": monitored, "G3": unmonitored}

        ranked = rank_candidates("G1", states, venue, config)
        g2 = next(c for c in ranked if c.gate_id == "G2")
        g3 = next(c for c in ranked if c.gate_id == "G3")

        assert g3.is_viable
        assert not g3.is_monitored
        assert g3.routing_score < g2.routing_score

    def test_unmonitored_target_can_be_forbidden(self):
        config = DecisionConfig(allow_unmonitored_targets=False)
        venue = make_venue(2, monitored=1)
        states = {
            "G1": saturated_state("G1", config),
            "G2": build_crowd_state("G2", 500, crowd_count=10, velocity=14.0, config=config),
        }
        ranked = rank_candidates("G1", states, venue, config)
        g2 = next(c for c in ranked if c.gate_id == "G2")
        assert not g2.is_viable
        assert "no camera" in (g2.rejection_reason or "")


class TestEngineSafety:
    def test_stale_observation_refuses_to_divert(self):
        config = DecisionConfig()
        venue = make_venue(2)
        stale = make_state(
            "G1", crowd_count=490, velocity=0.5,
            result_age_s=config.max_observation_age_s + 5, config=config,
        )
        states = {"G1": stale, "G2": quiet_state("G2", config)}

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        assert decision.action is ActionType.MONITOR
        assert "STALE_OBSERVATION" in decision.reason_codes

    def test_low_confidence_refuses_to_divert(self):
        config = DecisionConfig()
        venue = make_venue(2)
        unsure = make_state(
            "G1", crowd_count=490, velocity=0.5,
            confidence=config.min_confidence / 2, config=config,
        )
        states = {"G1": unsure, "G2": quiet_state("G2", config)}

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        assert decision.action is ActionType.MONITOR
        assert "LOW_CONFIDENCE" in decision.reason_codes

    def test_quiet_gate_takes_no_action(self):
        config = DecisionConfig()
        venue = make_venue(2)
        states = {"G1": quiet_state("G1", config), "G2": quiet_state("G2", config)}

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        assert decision.action is ActionType.NO_ACTION
        assert decision.target_gate_id is None

    def test_no_viable_target_restricts_inflow_instead_of_diverting(self):
        config = DecisionConfig()
        venue = make_venue(3)
        states = {gate: saturated_state(gate, config) for gate in ("G1", "G2", "G3")}

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        assert decision.action is ActionType.RESTRICT_INFLOW
        assert "NO_VIABLE_TARGET" in decision.reason_codes
        assert decision.target_gate_id is None

    def test_gates_without_states_are_skipped(self):
        config = DecisionConfig()
        venue = make_venue(4)
        states = {"G1": quiet_state("G1", config)}

        results = DecisionEngine(venue, config).evaluate(states)
        assert [r.source_gate_id for r in results] == ["G1"]

    def test_divert_requires_a_target(self):
        """The contract itself forbids an incoherent DIVERT."""
        from crowdguard.decision.contracts import DecisionResult

        with pytest.raises(ValueError, match="requires a target_gate_id"):
            DecisionResult(
                decision_id="X", timestamp=time.time(),
                source_gate_id="G1", action=ActionType.DIVERT,
            )


class TestEngineDiversion:
    def test_congested_gate_diverts_to_the_best_candidate(self):
        config = DecisionConfig()
        venue = make_venue(3)
        states = {
            "G1": saturated_state("G1", config),
            "G2": quiet_state("G2", config),
            "G3": make_state("G3", crowd_count=350, velocity=2.0, config=config),
        }

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        assert decision.action is ActionType.DIVERT
        assert decision.target_gate_id == "G2"
        assert "DIVERT_RECOMMENDED" in decision.reason_codes

    def test_decision_carries_a_perimeter_route(self):
        config = DecisionConfig()
        venue = make_venue(4)
        states = {
            "G1": saturated_state("G1", config),
            "G2": quiet_state("G2", config),
            "G3": saturated_state("G3", config),
            "G4": saturated_state("G4", config),
        }

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        assert decision.target_gate_id == "G2"
        assert decision.route == ("G1", "G2")

    def test_reason_is_explainable_and_quantified(self):
        config = DecisionConfig()
        venue = make_venue(2)
        states = {"G1": saturated_state("G1", config), "G2": quiet_state("G2", config)}

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        reason = decision.human_readable_reason
        assert "G1" in reason and "G2" in reason
        assert "CRITICAL" in reason
        assert decision.candidate_evaluations

    def test_unmonitored_target_is_flagged_in_the_reason(self):
        config = DecisionConfig()
        venue = make_venue(2, monitored=1)
        states = {
            "G1": saturated_state("G1", config),
            "G2": build_crowd_state("G2", 500, crowd_count=20, velocity=14.0, config=config),
        }

        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        assert decision.action is ActionType.DIVERT
        assert "no camera" in decision.human_readable_reason


class TestHysteresis:
    def test_target_is_held_during_cooldown(self):
        """Two near-equal candidates must not swap on every evaluation."""
        config = DecisionConfig()
        venue = make_venue(3)
        engine = DecisionEngine(venue, config)

        source = saturated_state("G1", config)
        states = {
            "G1": source,
            "G2": make_state("G2", crowd_count=100, velocity=12.0, config=config),
            "G3": make_state("G3", crowd_count=105, velocity=12.0, config=config),
        }

        first = next(d for d in engine.evaluate(states) if d.source_gate_id == "G1")
        assert first.action is ActionType.DIVERT
        chosen = first.target_gate_id

        # Nudge the other candidate slightly ahead - not enough to justify a switch.
        states["G2"] = make_state("G2", crowd_count=103, velocity=12.0, config=config)
        states["G3"] = make_state("G3", crowd_count=102, velocity=12.0, config=config)

        second = next(d for d in engine.evaluate(states) if d.source_gate_id == "G1")
        assert second.target_gate_id == chosen

    def test_clearly_better_candidate_takes_over(self):
        config = DecisionConfig(
            hysteresis={"cooldown_seconds": 60.0, "switching_margin": 0.05}
        )
        venue = make_venue(3)
        engine = DecisionEngine(venue, config)

        states = {
            "G1": saturated_state("G1", config),
            "G2": make_state("G2", crowd_count=300, velocity=6.0, config=config),
            "G3": make_state("G3", crowd_count=320, velocity=6.0, config=config),
        }
        first = next(d for d in engine.evaluate(states) if d.source_gate_id == "G1")
        assert first.action is ActionType.DIVERT

        # Now one candidate empties out entirely.
        states["G3"] = quiet_state("G3", config)
        second = next(d for d in engine.evaluate(states) if d.source_gate_id == "G1")
        assert second.target_gate_id == "G3"

    def test_diversion_clears_when_congestion_passes(self):
        config = DecisionConfig()
        venue = make_venue(2)
        engine = DecisionEngine(venue, config)

        states = {"G1": saturated_state("G1", config), "G2": quiet_state("G2", config)}
        engine.evaluate(states)
        assert "G1" in engine.active_diversions

        states["G1"] = quiet_state("G1", config)
        engine.evaluate(states)
        assert "G1" not in engine.active_diversions

    def test_reset_clears_all_hysteresis(self):
        config = DecisionConfig()
        venue = make_venue(2)
        engine = DecisionEngine(venue, config)
        engine.evaluate({"G1": saturated_state("G1", config), "G2": quiet_state("G2", config)})

        engine.reset_hysteresis()
        assert engine.active_diversions == {}


class TestSerialisation:
    def test_decision_to_dict_is_json_ready(self):
        import json

        config = DecisionConfig()
        venue = make_venue(2)
        states = {"G1": saturated_state("G1", config), "G2": quiet_state("G2", config)}
        decision = next(d for d in DecisionEngine(venue, config).evaluate(states)
                        if d.source_gate_id == "G1")

        payload = json.loads(json.dumps(decision.to_dict()))
        assert payload["action"] == "DIVERT"
        assert payload["source_gate_id"] == "G1"
        assert isinstance(payload["candidates"], list)
