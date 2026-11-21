"""Scenario suite for the rule-based decision engine.

Each test is a short, scripted story - a timeline of crowd states fed through
the real engine on a virtual clock - with the behaviour an operator would
expect written as assertions. Together they are the behavioural specification
of congestion control, and each one doubles as a slide-ready example.

Tests marked ``xfail(strict=True)`` document KNOWN WEAKNESSES of the current
engine. They are expected to fail today. ``strict`` means that when a later
engine fixes one, the suite reports it (XPASS -> failure) so the marker is
removed deliberately and the improvement is recorded.

States are built through ``build_crowd_state``, the same fusion path the live
system uses, so every score and level here is one production could produce.

Scoring reminder (defaults): score = 0.8 * count/capacity
+ 0.2 * (1 - velocity/75). With velocity 30 px/s the second term is 0.12, so
HIGH (0.65) begins at 331 of 500 people and CRITICAL (0.80) at 425.
"""

from __future__ import annotations

import statistics
import time

import pytest

from conftest import make_venue

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import (
    ActionType,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    MotionStatus,
    Trend,
)
from crowdguard.decision.engine import DecisionEngine
from crowdguard.decision.fusion import build_crowd_state, build_unmonitored_state

CAP = 500
FLOWING = 30.0  # px/s, a crowd that is moving but not freely


class Clock:
    """Virtual time, so cooldowns can be crossed without sleeping."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def st(
    gate_id: str,
    count: float,
    *,
    velocity: float = FLOWING,
    trend: Trend = Trend.STABLE,
    confidence: float = 1.0,
    age: float = 0.0,
    has_motion: bool = True,
) -> CrowdState:
    return build_crowd_state(
        gate_id=gate_id,
        gate_capacity=CAP,
        crowd_count=count,
        velocity=velocity,
        confidence=confidence,
        result_age_s=age,
        has_motion=has_motion,
        motion_status=MotionStatus.OK if has_motion else MotionStatus.UNAVAILABLE,
        trend=trend,
        config=DecisionConfig(),
    )


def sweep(engine: DecisionEngine, states: list[CrowdState]) -> dict[str, DecisionResult]:
    return {d.source_gate_id: d for d in engine.evaluate({s.gate_id: s for s in states})}


def ring(n: int, counts: dict[str, float] | None = None, default: float = 100.0, **kw) -> list[CrowdState]:
    counts = counts or {}
    return [st(f"G{i + 1}", counts.get(f"G{i + 1}", default), **kw) for i in range(n)]


def new_engine(n: int = 8, monitored: int | None = None) -> tuple[DecisionEngine, Clock]:
    clock = Clock()
    return DecisionEngine(make_venue(n, monitored=monitored, radius_m=120.0), clock=clock), clock


# ===================================================================
# Normal operation
# ===================================================================


class TestS01Calm:
    """S01 - a quiet venue produces no instructions."""

    def test_no_action_anywhere(self):
        engine, _ = new_engine()
        decisions = sweep(engine, ring(8, default=100.0, velocity=50.0))
        assert {d.action for d in decisions.values()} == {ActionType.NO_ACTION}
        assert engine.active_diversions == {}


class TestS02SingleSurge:
    """S02 - one gate fills up; the engine diverts to a free neighbour promptly."""

    def test_diverts_within_one_sweep_of_crossing_high(self):
        engine, clock = new_engine()
        first_divert_at = None
        crossed_at = None
        for tick, g1 in enumerate(range(150, 481, 30)):
            states = ring(8, {"G1": g1})
            decisions = sweep(engine, states)
            level = states[0].congestion_level
            if crossed_at is None and level in (CongestionLevel.HIGH, CongestionLevel.CRITICAL):
                crossed_at = tick
            if first_divert_at is None and decisions["G1"].action is ActionType.DIVERT:
                first_divert_at = tick
            clock.advance(0.5)
        assert crossed_at is not None
        assert first_divert_at == crossed_at

    def test_target_is_an_adjacent_free_gate_and_route_follows_the_perimeter(self):
        engine, _ = new_engine()
        d = sweep(engine, ring(8, {"G1": 470}))["G1"]
        assert d.action is ActionType.DIVERT
        assert d.target_gate_id in {"G2", "G8"}
        assert d.route == ("G1", d.target_gate_id)

    def test_other_gates_are_untouched(self):
        engine, _ = new_engine()
        decisions = sweep(engine, ring(8, {"G1": 470}))
        others = {g: d.action for g, d in decisions.items() if g != "G1"}
        assert set(others.values()) == {ActionType.NO_ACTION}


class TestS03SurgeSubsides:
    """S03 - once the surge clears, the diversion is withdrawn."""

    def test_diversion_is_cancelled(self):
        engine, clock = new_engine()
        assert sweep(engine, ring(8, {"G1": 470}))["G1"].action is ActionType.DIVERT
        clock.advance(30)
        d = sweep(engine, ring(8, {"G1": 150}))["G1"]
        assert d.action in (ActionType.NO_ACTION, ActionType.MONITOR)
        assert "G1" not in engine.active_diversions


# ===================================================================
# Saturation and target changes
# ===================================================================


class TestS04AllGatesFull:
    """S04 - with nowhere to send people, restrict inflow; never divert."""

    def test_every_gate_restricts(self):
        engine, _ = new_engine()
        decisions = sweep(engine, ring(8, default=470.0))
        assert {d.action for d in decisions.values()} == {ActionType.RESTRICT_INFLOW}
        for d in decisions.values():
            assert "NO_VIABLE_TARGET" in d.reason_codes


class TestS05TargetFillsMidDiversion:
    """S05 - if the target itself fills up, switch at once, cooldown or not."""

    def test_switches_immediately_when_target_becomes_unviable(self):
        engine, clock = new_engine()
        first = sweep(engine, ring(8, {"G1": 470, "G2": 20, "G8": 150}))["G1"]
        assert first.target_gate_id == "G2"
        clock.advance(2)  # well inside the 10 s cooldown
        second = sweep(engine, ring(8, {"G1": 470, "G2": 460, "G8": 150}))["G1"]
        assert second.action is ActionType.DIVERT
        assert second.target_gate_id != "G2"


class TestS06TargetStability:
    """S06 - a marginally better gate does not steal the target during cooldown."""

    def test_holds_target_then_moves_after_cooldown(self):
        engine, clock = new_engine()
        first = sweep(engine, ring(8, {"G1": 470, "G2": 100, "G8": 140}))["G1"]
        assert first.target_gate_id == "G2"

        clock.advance(3)
        held = sweep(engine, ring(8, {"G1": 470, "G2": 140, "G8": 100}))["G1"]
        assert held.target_gate_id == "G2", "small advantage must not switch inside cooldown"

        clock.advance(10)
        moved = sweep(engine, ring(8, {"G1": 470, "G2": 140, "G8": 100}))["G1"]
        assert moved.target_gate_id == "G8", "after cooldown the better gate wins"


# ===================================================================
# Safety refusals
# ===================================================================


class TestS07StaleSource:
    """S07 - data older than the limit never triggers a diversion."""

    def test_monitor_only(self):
        engine, _ = new_engine()
        states = ring(8)
        states[0] = st("G1", 470, age=8.0)
        d = sweep(engine, states)["G1"]
        assert d.action is ActionType.MONITOR
        assert "STALE_OBSERVATION" in d.reason_codes


class TestS08LowConfidence:
    """S08 - a low-confidence observation never triggers a diversion."""

    def test_monitor_only(self):
        engine, _ = new_engine()
        states = ring(8)
        states[0] = st("G1", 470, confidence=0.2)
        d = sweep(engine, states)["G1"]
        assert d.action is ActionType.MONITOR
        assert "LOW_CONFIDENCE" in d.reason_codes


class TestS09MotionMissing:
    """S09 - without motion the engine still acts on density, with less confidence."""

    def test_density_only_still_diverts(self):
        engine, _ = new_engine()
        states = ring(8)
        states[0] = st("G1", 470, has_motion=False)
        d = sweep(engine, states)["G1"]
        assert d.action is ActionType.DIVERT
        assert d.confidence == pytest.approx(0.7)


# ===================================================================
# Gates without cameras
# ===================================================================


class TestS10Unmonitored:
    """S10 - unseen gates are never a source and are a penalised target."""

    def _states(self, counts: dict[str, float]) -> list[CrowdState]:
        out = []
        for i in range(8):
            gid = f"G{i + 1}"
            if i < 4:
                out.append(st(gid, counts.get(gid, 100.0)))
            else:
                out.append(build_unmonitored_state(gid, CAP))
        return out

    def test_unmonitored_gate_is_never_a_source(self):
        engine, _ = new_engine(8, monitored=4)
        decisions = sweep(engine, self._states({}))
        for gid in ("G5", "G6", "G7", "G8"):
            assert decisions[gid].action not in (ActionType.DIVERT, ActionType.RESTRICT_INFLOW)

    def test_observed_gate_preferred_over_equal_unobserved_gate(self):
        engine, _ = new_engine(8, monitored=4)
        # G2 (observed) and G8 (unobserved, assumed 30% full) are both one hop
        # away and equally loaded; G3 and G4 are busy and out of the running.
        d = sweep(engine, self._states({"G1": 470, "G2": 150, "G3": 420, "G4": 420}))["G1"]
        assert d.target_gate_id == "G2"

    def test_unmonitored_gate_used_when_it_is_the_only_option(self):
        engine, _ = new_engine(8, monitored=4)
        d = sweep(engine, self._states({"G1": 470, "G2": 470, "G3": 470, "G4": 470}))["G1"]
        assert d.action is ActionType.DIVERT
        assert d.target_gate_id in {"G5", "G6", "G7", "G8"}
        assert "no camera" in d.human_readable_reason


# ===================================================================
# Ranking preferences
# ===================================================================


class TestS11TrendPreference:
    """S11 - between equal gates, prefer the one that is clearing."""

    def test_prefers_decreasing_over_increasing(self):
        engine, _ = new_engine()
        states = ring(8, {"G1": 470})
        states[1] = st("G2", 150, trend=Trend.INCREASING)
        states[7] = st("G8", 150, trend=Trend.DECREASING)
        assert sweep(engine, states)["G1"].target_gate_id == "G8"


class TestS12DistancePreference:
    """S12 - between equal gates, prefer the nearer one."""

    def test_prefers_nearer_gate(self):
        engine, _ = new_engine()
        counts = {f"G{i}": 420.0 for i in range(1, 9)}  # busy and non-viable
        counts.update({"G1": 470, "G2": 100, "G4": 100})
        assert sweep(engine, ring(8, counts))["G1"].target_gate_id == "G2"


# ===================================================================
# Engineering properties
# ===================================================================


class TestS13Determinism:
    """S13 - identical input gives identical output."""

    def test_two_engines_agree(self):
        states = ring(8, {"G1": 470, "G4": 400, "G6": 380})
        a = sweep(new_engine()[0], states)
        b = sweep(new_engine()[0], states)
        for gid in a:
            assert (a[gid].action, a[gid].target_gate_id, a[gid].route) == (
                b[gid].action, b[gid].target_gate_id, b[gid].route)
            assert [c.routing_score for c in a[gid].candidate_evaluations] == [
                c.routing_score for c in b[gid].candidate_evaluations]


class TestS14Scale:
    """S14 - a 12-gate venue (the maximum) is evaluated well inside the 0.5 s decision period."""

    def test_sweep_latency(self):
        engine, _ = new_engine(12)
        states = ring(12, {"G1": 470, "G4": 460, "G7": 450, "G10": 440}, default=250.0)
        timings = []
        for _ in range(20):
            t0 = time.perf_counter()
            sweep(engine, states)
            timings.append(time.perf_counter() - t0)
        assert statistics.median(timings) < 0.05


class TestS15Explainability:
    """S15 - every decision says why, naming the gates involved."""

    def test_reasons_name_source_and_target(self):
        engine, _ = new_engine()
        decisions = sweep(engine, ring(8, {"G1": 470, "G5": 360}))
        for gid, d in decisions.items():
            assert d.reason_codes, gid
            assert gid in d.human_readable_reason
            if d.action is ActionType.DIVERT:
                assert d.target_gate_id in d.human_readable_reason


# ===================================================================
# Known weaknesses of the current engine (expected to fail)
# ===================================================================


class TestK1Herding:
    """K1 - two congested gates must not both send their crowd to one gate
    that can absorb only one of them.

    G1 and G3 are critical; G2 between them is the only viable target, with
    room for about 80 people before it is HIGH itself. Each source has about
    150 people above its HIGH line, so sending both streams overloads G2.
    Acceptable outcomes: one source diverts and the other restricts, or the
    two share G2 with explicit proportions. The current engine evaluates each
    gate in isolation and sends both.
    """

    @pytest.mark.xfail(strict=True, reason="known weakness: gates decide independently (herding)")
    def test_single_small_target_is_not_given_two_sources(self):
        engine, _ = new_engine()
        counts = {f"G{i}": 440.0 for i in range(1, 9)}
        counts.update({"G1": 480, "G3": 480, "G2": 250})
        decisions = sweep(engine, ring(8, counts))
        to_g2 = [g for g, d in decisions.items()
                 if d.action is ActionType.DIVERT and d.target_gate_id == "G2"]
        assert len(to_g2) <= 1, f"{to_g2} all sent to G2"


class TestK2Chatter:
    """K2 - a score hovering around the HIGH line must not toggle the
    diversion on and off every half second.

    The current engine has hysteresis only on *which* target is used, not on
    *whether* to divert, so each dip below 0.65 cancels the diversion.
    """

    @pytest.mark.xfail(strict=True, reason="known weakness: no on/off hysteresis")
    def test_on_off_toggles_are_bounded(self):
        engine, clock = new_engine()
        toggles = 0
        previous = None
        for tick in range(40):  # 20 s at 2 Hz
            g1 = 344 if tick % 2 == 0 else 319  # score 0.67 / 0.63
            diverting = sweep(engine, ring(8, {"G1": g1}))["G1"].action is ActionType.DIVERT
            if previous is not None and diverting != previous:
                toggles += 1
            previous = diverting
            clock.advance(0.5)
        assert toggles <= 2, f"{toggles} on/off toggles in 20 s"


class TestK3StaleTarget:
    """K3 - a gate whose own data is stale should not be chosen as a target.

    Staleness is checked for the source gate only. A target whose camera froze
    30 s ago still looks empty and wins the ranking.
    """

    @pytest.mark.xfail(strict=True, reason="known weakness: target freshness is not checked")
    def test_stale_target_not_chosen(self):
        engine, _ = new_engine()
        states = ring(8, {"G1": 470, "G8": 200})
        states[1] = st("G2", 20, age=30.0)
        assert sweep(engine, states)["G1"].target_gate_id != "G2"


class TestK4AllOrNothing:
    """K4 - a diversion should say how much of the crowd to send.

    Sending 100 % of arrivals usually just moves the queue to the target. The
    current decision carries a target but no share.
    """

    @pytest.mark.xfail(strict=True, reason="known weakness: diversion has no proportion")
    def test_divert_carries_a_share(self):
        engine, _ = new_engine()
        d = sweep(engine, ring(8, {"G1": 470}))["G1"]
        assert d.action is ActionType.DIVERT
        share = getattr(d, "divert_fraction", None)
        assert share is not None and 0.0 < share < 1.0
