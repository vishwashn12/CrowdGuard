"""Routing candidate scorer for diversion decisions.

Scores candidate gates on a weighted combination of low congestion, available
capacity, short distance, and a favourable trend. Unobserved gates are legal
targets but carry a configurable penalty.

No dependency on torch, cv2, or any vision model code.
"""

from __future__ import annotations

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import (
    CandidateEvaluation,
    CongestionLevel,
    CrowdState,
    Trend,
)
from crowdguard.venue.topology import VenueTopology


def trend_bonus(trend: Trend) -> float:
    """Routing bonus for a candidate's trend.

    A clearing gate is the best place to send people; a filling one is the worst.
    """
    if trend == Trend.DECREASING:
        return 1.0
    if trend == Trend.STABLE:
        return 0.5
    return 0.0


def evaluate_candidate(
    source_gate_id: str,
    candidate_state: CrowdState,
    venue: VenueTopology,
    config: DecisionConfig,
    max_venue_distance: float,
) -> CandidateEvaluation:
    """Evaluate one candidate gate as a diversion target.

    A candidate is rejected when it is itself congested, has too little
    headroom, is unreachable, or is unobserved while policy forbids unobserved
    targets. The score is computed either way so the operator can see the
    ranking behind a rejection.
    """
    candidate_id = candidate_state.gate_id
    distance = venue.shortest_distance(source_gate_id, candidate_id)
    is_monitored = venue.is_monitored(candidate_id)

    capacity_utilization = candidate_state.density_score
    remaining_ratio = 1.0 - capacity_utilization

    rejection_reason: str | None = None

    if distance == float("inf"):
        rejection_reason = "Gate is unreachable from the source gate"
    elif candidate_state.congestion_score > config.max_candidate_congestion:
        rejection_reason = (
            f"Congestion {candidate_state.congestion_score:.2f} exceeds the "
            f"maximum of {config.max_candidate_congestion:.2f} for a target"
        )
    elif remaining_ratio < config.min_capacity_remaining_ratio:
        rejection_reason = (
            f"Remaining capacity ratio {remaining_ratio:.2f} is below the "
            f"minimum of {config.min_capacity_remaining_ratio:.2f}"
        )
    elif not is_monitored and not config.allow_unmonitored_targets:
        rejection_reason = "Gate has no camera and unmonitored targets are disabled"

    is_viable = rejection_reason is None

    w = config.routing_weights
    norm_distance = min(1.0, distance / max_venue_distance) if max_venue_distance > 0 else 0.5

    score = (
        w.w_congestion * (1.0 - candidate_state.congestion_score)
        + w.w_capacity * remaining_ratio
        + w.w_distance * (1.0 - norm_distance)
        + w.w_trend * trend_bonus(candidate_state.trend)
    )

    weight_sum = w.w_congestion + w.w_capacity + w.w_distance + w.w_trend
    if weight_sum > 0:
        score /= weight_sum

    # Prefer a gate we can actually see, all else being close to equal.
    if not is_monitored:
        score = max(0.0, score - config.unmonitored_target_penalty)

    return CandidateEvaluation(
        gate_id=candidate_id,
        distance_m=distance,
        congestion_level=candidate_state.congestion_level,
        congestion_score=candidate_state.congestion_score,
        capacity_utilization=capacity_utilization,
        trend=candidate_state.trend,
        routing_score=score,
        is_viable=is_viable,
        is_monitored=is_monitored,
        rejection_reason=rejection_reason,
    )


def rank_candidates(
    source_gate_id: str,
    all_states: dict[str, CrowdState],
    venue: VenueTopology,
    config: DecisionConfig,
) -> list[CandidateEvaluation]:
    """Evaluate and rank every reachable gate as a diversion target.

    Viable candidates sort ahead of rejected ones; within each group the higher
    routing score wins. Gates with no state at all are reported as rejected
    rather than silently dropped.
    """
    max_dist = venue.max_distance()
    evaluations: list[CandidateEvaluation] = []

    for gate_id in venue.reachable_gates(source_gate_id):
        candidate_state = all_states.get(gate_id)
        if candidate_state is None:
            evaluations.append(CandidateEvaluation(
                gate_id=gate_id,
                distance_m=venue.shortest_distance(source_gate_id, gate_id),
                congestion_level=CongestionLevel.LOW,
                congestion_score=0.0,
                capacity_utilization=0.0,
                trend=Trend.STABLE,
                routing_score=0.0,
                is_viable=False,
                is_monitored=venue.is_monitored(gate_id),
                rejection_reason="No crowd state available for this gate",
            ))
            continue

        evaluations.append(evaluate_candidate(
            source_gate_id, candidate_state, venue, config, max_dist
        ))

    evaluations.sort(key=lambda e: (-int(e.is_viable), -e.routing_score))
    return evaluations
