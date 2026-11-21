"""Engine v2: the shared behaviour scenarios, plus the weaknesses it fixes.

The general scenarios (S01, S02, S04, S05, S07-S15) are re-run unchanged
against v2 by swapping the engine the scenario module builds. Scenarios whose
expected behaviour v2 changes on purpose (S03: a diversion is now held while
the crowd clears; S06: the target is held for a minute) have v2 versions here,
as do the known-weakness tests K1, K2 and K4, which v2 must PASS.
"""

from __future__ import annotations

import pytest

import test_decision_scenarios as S
from test_decision_scenarios import (  # noqa: F401  (re-run against v2)
    TestS01Calm, TestS02SingleSurge, TestS04AllGatesFull, TestS05TargetFillsMidDiversion,
    TestS07StaleSource, TestS08LowConfidence, TestS09MotionMissing, TestS10Unmonitored,
    TestS11TrendPreference, TestS12DistancePreference, TestS13Determinism, TestS14Scale,
    TestS15Explainability,
)
from conftest import make_venue

from crowdguard.decision.contracts import ActionType
from crowdguard.decision.engine_v2 import DecisionEngineV2, V2Config, V2Features


def new_v2(n: int = 8, monitored: int | None = None, **v2):
    clock = S.Clock()
    engine = DecisionEngineV2(make_venue(n, monitored=monitored, radius_m=120.0), clock=clock,
                              v2_config=V2Config(**v2))
    return engine, clock


@pytest.fixture(autouse=True)
def _scenarios_use_v2(monkeypatch):
    monkeypatch.setattr(S, "new_engine", new_v2)


ring, sweep = S.ring, S.sweep


class TestV2HeldDiversion:
    """S03 for v2: a diversion is held while the crowd clears, then withdrawn."""

    def test_held_then_released(self):
        engine, clock = new_v2()
        assert sweep(engine, ring(8, {"G1": 470}))["G1"].action is ActionType.DIVERT
        clock.advance(5)
        held = sweep(engine, ring(8, {"G1": 150}))["G1"]
        assert held.action is ActionType.DIVERT and "DIVERSION_HELD" in held.reason_codes
        for _ in range(140):                    # 70 s of a calm gate at 2 Hz
            clock.advance(0.5)
            last = sweep(engine, ring(8, {"G1": 150}))["G1"]
        assert last.action in (ActionType.NO_ACTION, ActionType.MONITOR)

    def test_unsafe_data_stops_a_held_diversion_at_once(self):
        engine, clock = new_v2()
        sweep(engine, ring(8, {"G1": 470}))
        clock.advance(1)
        states = ring(8)
        states[0] = S.st("G1", 470, age=8.0)
        assert sweep(engine, states)["G1"].action is ActionType.MONITOR


class TestV2TargetHold:
    """S06 for v2: a target is kept for target_hold_s; after that only a
    clearly better gate (switching margin) takes over."""

    def test_holds_for_a_minute_then_moves_only_for_a_clear_gain(self):
        engine, clock = new_v2()
        assert sweep(engine, ring(8, {"G1": 470, "G2": 100, "G8": 140}))["G1"].target_gate_id == "G2"
        clock.advance(30)   # inside the hold: even a clearly better gate waits
        assert sweep(engine, ring(8, {"G1": 470, "G2": 330, "G8": 40}))["G1"].target_gate_id == "G2"
        clock.advance(40)   # hold over, small gain: stay
        assert sweep(engine, ring(8, {"G1": 470, "G2": 140, "G8": 100}))["G1"].target_gate_id == "G2"
        clock.advance(1)    # hold over, clear gain: move
        assert sweep(engine, ring(8, {"G1": 470, "G2": 330, "G8": 40}))["G1"].target_gate_id == "G8"


class TestV2FixesKnownWeaknesses:
    def test_k1_no_unconditional_double_diversion(self):
        engine, _ = new_v2()
        counts = {f"G{i}": 440.0 for i in range(1, 9)}
        counts.update({"G1": 480, "G3": 480, "G2": 250})
        decisions = sweep(engine, ring(8, counts))
        to_g2 = [d for d in decisions.values() if d.action is ActionType.DIVERT and d.target_gate_id == "G2"]
        full = [d for d in to_g2 if (d.divert_fraction or 1.0) >= 1.0]
        assert len(full) <= 1, "at most one source may send everyone to G2"

    def test_k2_on_off_toggles_are_bounded(self):
        engine, clock = new_v2()
        toggles, previous = 0, None
        for tick in range(40):
            g1 = 344 if tick % 2 == 0 else 319
            diverting = sweep(engine, ring(8, {"G1": g1}))["G1"].action is ActionType.DIVERT
            if previous is not None and diverting != previous:
                toggles += 1
            previous = diverting
            clock.advance(0.5)
        assert toggles <= 2

    def test_k4_divert_carries_a_share(self):
        engine, _ = new_v2()
        d = sweep(engine, ring(8, {"G1": 470}))["G1"]
        assert d.action is ActionType.DIVERT and 0 < d.divert_fraction <= 1

    def test_a_diverting_gate_is_not_used_as_a_target(self):
        engine, clock = new_v2()
        # G2 starts diverting first, then G1 fills: G1 must not send people to G2.
        sweep(engine, ring(8, {"G2": 470}))
        clock.advance(1)
        d = sweep(engine, ring(8, {"G1": 470, "G2": 300}))["G1"]
        assert d.action is ActionType.DIVERT and d.target_gate_id != "G2"


class TestV2Ablation:
    def test_all_switches_off_is_v1(self):
        from crowdguard.decision.engine import DecisionEngine
        clock = S.Clock()
        venue = make_venue(8, radius_m=120.0)
        off = DecisionEngineV2(venue, clock=clock, v2_config=V2Config(
            features=V2Features(steady_diversion=False, coordination=False)))
        v1 = DecisionEngine(venue, clock=clock)
        for counts in ({"G1": 470}, {"G1": 470, "G3": 480, "G2": 250}, {"G4": 350}):
            a, b = sweep(off, ring(8, counts)), sweep(v1, ring(8, counts))
            assert {g: (d.action, d.target_gate_id) for g, d in a.items()} == \
                   {g: (d.action, d.target_gate_id) for g, d in b.items()}
