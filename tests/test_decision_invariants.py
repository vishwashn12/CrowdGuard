"""Invariants of the decision engine, checked on thousands of random venues.

Where the scenario suite checks specific stories, these check rules that must
hold for EVERY input: a diversion never points at the source, never at a
rejected gate, never from an unobserved gate, and so on. Inputs are generated
from a fixed seed, so a failure is reproducible from the printed case number.
"""

from __future__ import annotations

import random

import pytest

from conftest import make_venue

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import (
    ActionType,
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

CASES = 600
SEED = 20261121


def random_case(rng: random.Random):
    n = rng.randint(2, 12)
    monitored = rng.randint(1, n)
    venue = make_venue(n, monitored=monitored, radius_m=rng.choice([40.0, 120.0, 300.0]))
    config = DecisionConfig()
    states: dict[str, CrowdState] = {}
    for gid in venue.gate_ids:
        cap = venue.get_gate_capacity(gid)
        if not venue.is_monitored(gid):
            states[gid] = build_unmonitored_state(gid, cap, config=config, trend=rng.choice(list(Trend)))
            continue
        if rng.random() < 0.05:
            continue  # occasionally a monitored gate has no state at all
        has_motion = rng.random() > 0.2
        states[gid] = build_crowd_state(
            gate_id=gid,
            gate_capacity=cap,
            crowd_count=rng.uniform(0, 1.3 * cap),
            velocity=rng.uniform(0, 120),
            confidence=rng.uniform(0, 1),
            result_age_s=rng.choice([0.0, 0.0, 0.0, rng.uniform(0, 12)]),
            has_motion=has_motion,
            motion_status=MotionStatus.OK if has_motion else MotionStatus.UNAVAILABLE,
            trend=rng.choice(list(Trend)),
            config=config,
        )
    return venue, config, states


def all_cases():
    rng = random.Random(SEED)
    return [random_case(rng) for _ in range(CASES)]


CASES_DATA = all_cases()


@pytest.mark.parametrize("case", range(0, CASES, 1), ids=lambda i: f"case{i}")
def test_engine_invariants(case):
    venue, config, states = CASES_DATA[case]
    decisions = DecisionEngine(venue, config).evaluate(states)

    # one decision per gate that has a state, none for the others
    assert sorted(d.source_gate_id for d in decisions) == sorted(states)

    for d in decisions:
        src = states[d.source_gate_id]
        assert 0.0 <= d.confidence <= 1.0
        assert d.reason_codes and d.human_readable_reason

        acting = d.action in (ActionType.DIVERT, ActionType.RESTRICT_INFLOW)
        if acting:
            # only a congested, fresh, trustworthy, observed gate acts
            assert level_at_least(src.congestion_level, config.divert_above_level)
            assert src.result_age_s <= config.max_observation_age_s
            assert src.confidence >= config.min_confidence
            assert venue.is_monitored(d.source_gate_id)

        viable = [c for c in d.candidate_evaluations if c.is_viable]
        if d.action is ActionType.RESTRICT_INFLOW:
            assert not viable

        if d.action is ActionType.DIVERT:
            tgt = d.target_gate_id
            assert tgt is not None and tgt != d.source_gate_id
            assert tgt in venue.gate_ids
            chosen = next(c for c in d.candidate_evaluations if c.gate_id == tgt)
            assert chosen.is_viable
            # a fresh engine has no hysteresis history, so the best viable wins
            assert chosen.routing_score == max(c.routing_score for c in viable)
            # route walks the perimeter, neighbour to neighbour
            assert d.route[0] == d.source_gate_id and d.route[-1] == tgt
            for a, b in zip(d.route, d.route[1:]):
                assert b in venue.neighbors(a)

        for c in d.candidate_evaluations:
            assert c.gate_id != d.source_gate_id
            if c.is_viable:
                assert c.rejection_reason is None
                assert c.congestion_score <= config.max_candidate_congestion
                assert 1.0 - c.capacity_utilization >= config.min_capacity_remaining_ratio
            else:
                assert c.rejection_reason


def test_engine_is_deterministic():
    for venue, config, states in CASES_DATA[:100]:
        a = DecisionEngine(venue, config).evaluate(states)
        b = DecisionEngine(venue, config).evaluate(states)
        assert [(d.action, d.target_gate_id, d.route) for d in a] == [
            (d.action, d.target_gate_id, d.route) for d in b]


def test_score_is_monotone_in_crowd_and_speed():
    """More people never lowers the score; a faster crowd never raises it."""
    rng = random.Random(SEED + 1)
    config = DecisionConfig()
    for _ in range(3000):
        u1, u2 = sorted(rng.uniform(0, 1) for _ in range(2))
        v = rng.uniform(0, 150)
        assert compute_congestion_score(u1, v, u1, config) <= compute_congestion_score(u2, v, u2, config)
        v1, v2 = sorted(rng.uniform(0, 150) for _ in range(2))
        u = rng.uniform(0, 1)
        assert compute_congestion_score(u, v2, u, config) <= compute_congestion_score(u, v1, u, config)


def test_levels_follow_thresholds():
    config = DecisionConfig()
    t = config.congestion_thresholds
    rng = random.Random(SEED + 2)
    for _ in range(3000):
        s = rng.uniform(0, 1)
        level = classify_congestion(s, config).value
        expected = ("CRITICAL" if s >= t.critical_threshold else "HIGH" if s >= t.high_threshold
                    else "MODERATE" if s >= t.moderate_threshold else "LOW")
        assert level == expected
