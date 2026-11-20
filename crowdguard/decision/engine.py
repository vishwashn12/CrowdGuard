"""Rule-based Decision Engine for crowd management.

Input:  ``dict[gate_id, CrowdState]`` plus a :class:`VenueTopology`
Output: ``list[DecisionResult]``

The engine has zero dependencies on torch, cv2, CSRNet, YOLO, FramePacket, or
any part of the ingestion stack. It knows only what is happening at each gate
and how the gates connect.

Safety rules, each of which produces an explainable refusal rather than a
diversion:

* the source observation is stale,
* the source observation is too low-confidence,
* the source gate is not actually congested,
* no viable destination exists.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import (
    ActionType,
    CandidateEvaluation,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    level_at_least,
)
from crowdguard.decision.routing import rank_candidates
from crowdguard.venue.topology import VenueTopology

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Pure rule-based crowd management Decision Engine.

    For every gate whose congestion reaches the configured level:

    1. evaluate all reachable candidate gates,
    2. pick the best viable one by routing score,
    3. apply hysteresis so the target does not oscillate,
    4. emit an explainable :class:`DecisionResult`.
    """

    def __init__(
        self,
        venue: VenueTopology,
        config: DecisionConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Args:
            venue: Gate graph the engine routes over.
            config: Thresholds and weights; defaults when omitted.
            clock: Source of "now" in seconds, used for hysteresis timing and
                decision timestamps. Defaults to wall-clock time. Simulation
                and tests pass a virtual clock so time can run faster than real
                time without breaking the cooldown logic.
        """
        self._venue = venue
        self._config = config or DecisionConfig()
        self._clock: Callable[[], float] = clock or time.time

        # gate_id -> (target_gate_id, decision_timestamp)
        self._active_diversions: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._evaluation_count = 0

    @property
    def venue(self) -> VenueTopology:
        return self._venue

    @property
    def config(self) -> DecisionConfig:
        return self._config

    @property
    def evaluation_count(self) -> int:
        with self._lock:
            return self._evaluation_count

    @property
    def active_diversions(self) -> dict[str, str]:
        """Current source gate -> target gate diversions."""
        with self._lock:
            return {src: target for src, (target, _) in self._active_diversions.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, states: dict[str, CrowdState]) -> list[DecisionResult]:
        """Evaluate every gate that has an observation.

        Args:
            states: gate_id -> CrowdState for all gates with a state.

        Returns:
            One :class:`DecisionResult` per evaluated gate, in venue gate order.
        """
        results: list[DecisionResult] = []
        now = self._clock()

        for gate_id in self._venue.gate_ids:
            state = states.get(gate_id)
            if state is None:
                continue
            results.append(self._evaluate_gate(gate_id, state, states, now))

        with self._lock:
            self._evaluation_count += len(results)

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_gate(
        self,
        gate_id: str,
        state: CrowdState,
        all_states: dict[str, CrowdState],
        now: float,
    ) -> DecisionResult:
        """Evaluate one gate and produce its decision."""

        # --- Safety: the observation is too old to act on ---
        if state.result_age_s > self._config.max_observation_age_s:
            self._clear_diversion(gate_id)
            return self._make_result(
                gate_id, ActionType.MONITOR, state.congestion_level,
                confidence=state.confidence * 0.5,
                reason_codes=("STALE_OBSERVATION",),
                reason=(
                    f"The observation for {gate_id} is {state.result_age_s:.1f}s old, past the "
                    f"{self._config.max_observation_age_s:.1f}s limit. Holding at monitor only: "
                    f"issuing a diversion on stale data could send people into a gate that has "
                    f"since filled up."
                ),
            )

        # --- Safety: confidence too low to act on ---
        if state.confidence < self._config.min_confidence:
            self._clear_diversion(gate_id)
            return self._make_result(
                gate_id, ActionType.MONITOR, state.congestion_level,
                confidence=state.confidence,
                reason_codes=("LOW_CONFIDENCE",),
                reason=(
                    f"Confidence in the {gate_id} observation is {state.confidence:.2f}, below the "
                    f"{self._config.min_confidence:.2f} minimum "
                    f"({'density missing' if not state.has_density else 'motion ' + state.motion_status.value.lower()}). "
                    f"Monitoring only."
                ),
            )

        # --- Congestion below the action threshold ---
        if not level_at_least(state.congestion_level, self._config.divert_above_level):
            self._clear_diversion(gate_id)
            action = (
                ActionType.MONITOR
                if state.congestion_level == CongestionLevel.MODERATE
                else ActionType.NO_ACTION
            )
            return self._make_result(
                gate_id, action, state.congestion_level,
                confidence=state.confidence,
                reason_codes=("BELOW_THRESHOLD",),
                reason=(
                    f"{gate_id} congestion is {state.congestion_level.value} "
                    f"(score {state.congestion_score:.2f}, {state.crowd_count:.0f} people, "
                    f"{state.density_score:.0%} of capacity). No diversion needed."
                ),
            )

        # --- Congested: rank the alternatives ---
        candidates = rank_candidates(gate_id, all_states, self._venue, self._config)
        viable = [c for c in candidates if c.is_viable]

        if not viable:
            self._clear_diversion(gate_id)
            blockers = "; ".join(
                f"{c.gate_id}: {c.rejection_reason}" for c in candidates[:4] if c.rejection_reason
            )
            return self._make_result(
                gate_id, ActionType.RESTRICT_INFLOW, state.congestion_level,
                confidence=state.confidence,
                reason_codes=("CONGESTED", "NO_VIABLE_TARGET"),
                reason=(
                    f"{gate_id} is {state.congestion_level.value} (score {state.congestion_score:.2f}) "
                    f"but there is nowhere to send people: every reachable gate is congested or out of "
                    f"headroom. Restrict inflow at {gate_id} instead of diverting. Blocked targets - {blockers}"
                ),
                candidates=tuple(candidates),
            )

        best, target_id = self._apply_hysteresis(gate_id, viable, now)

        with self._lock:
            self._active_diversions[gate_id] = (target_id, now)

        route = tuple(self._venue.perimeter_path(gate_id, target_id))
        monitored_note = (
            "" if best.is_monitored
            else " Note: this gate has no camera, so its load is assumed rather than measured."
        )

        reason = (
            f"{gate_id} has {state.congestion_level.value} congestion "
            f"(score {state.congestion_score:.2f}, {state.crowd_count:.0f} people) with "
            f"{state.trend.value.lower()} density. Divert to {target_id}: "
            f"{best.distance_m:.0f}m away along the perimeter, {best.congestion_level.value} congestion "
            f"(score {best.congestion_score:.2f}), {best.capacity_utilization:.0%} of capacity used."
            f"{monitored_note}"
        )

        return self._make_result(
            gate_id, ActionType.DIVERT, state.congestion_level,
            target_gate_id=target_id,
            confidence=state.confidence,
            reason_codes=("CONGESTED", "DIVERT_RECOMMENDED"),
            reason=reason,
            candidates=tuple(candidates),
            route=route,
        )

    def _apply_hysteresis(
        self,
        gate_id: str,
        viable: list[CandidateEvaluation],
        now: float,
    ) -> tuple[CandidateEvaluation, str]:
        """Keep the incumbent target during cooldown unless clearly beaten.

        Without this, two gates with near-identical scores would trade the
        diversion back and forth every evaluation, and the signage and staff on
        the ground could never keep up.
        """
        best = viable[0]

        with self._lock:
            active = self._active_diversions.get(gate_id)

        if active is None:
            return best, best.gate_id

        previous_target, previous_time = active
        if (now - previous_time) >= self._config.hysteresis.cooldown_seconds:
            return best, best.gate_id

        incumbent = next((c for c in viable if c.gate_id == previous_target), None)
        if incumbent is None:
            # The incumbent is no longer viable; switching is required.
            return best, best.gate_id

        margin = best.routing_score - incumbent.routing_score
        if margin < self._config.hysteresis.switching_margin:
            return incumbent, previous_target

        return best, best.gate_id

    def _clear_diversion(self, gate_id: str) -> None:
        """Drop the hysteresis record once a gate stops needing a diversion."""
        with self._lock:
            self._active_diversions.pop(gate_id, None)

    def _make_result(
        self,
        source_gate_id: str,
        action: ActionType,
        severity: CongestionLevel,
        target_gate_id: str | None = None,
        confidence: float = 1.0,
        reason_codes: tuple[str, ...] = (),
        reason: str = "",
        candidates: tuple[CandidateEvaluation, ...] = (),
        route: tuple[str, ...] = (),
    ) -> DecisionResult:
        return DecisionResult(
            decision_id=DecisionResult.generate_id(),
            timestamp=self._clock(),
            source_gate_id=source_gate_id,
            action=action,
            target_gate_id=target_gate_id,
            severity=severity,
            confidence=min(1.0, max(0.0, confidence)),
            reason_codes=reason_codes,
            human_readable_reason=reason,
            candidate_evaluations=candidates,
            route=route,
        )

    def reset_hysteresis(self, gate_id: str | None = None) -> None:
        """Clear hysteresis state for one gate, or all of them."""
        with self._lock:
            if gate_id is not None:
                self._active_diversions.pop(gate_id, None)
            else:
                self._active_diversions.clear()
