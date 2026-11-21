"""Decision engine v2: v1's rules plus measured fixes, each behind a switch.

Every improvement is a flag in :class:`V2Features`, so the evaluation can
switch any one off (ablation), and v1 stays available unchanged. Each fix
targets a weakness measured in the v1 baseline (docs/12, docs/14).

``steady_diversion`` (K2, W1, W3)
    v1 flips its instruction on and off hundreds of times an hour, because a
    diversion is cancelled the moment the score dips below HIGH. v2 starts a
    diversion exactly when v1 would, at 100 %, but holds it until the source
    score has stayed below ``exit_score`` for ``release_s`` seconds (and never
    before ``min_on_s``). It also fixes v1's target cooldown, which was timed
    from the latest decision - so it never expired while diverting - instead
    of from the last target change.

``coordination`` (K1, W2, W4)
    v1 evaluates every gate on its own, so two crowded gates can send their
    crowds to the same target, and a target can be diverting onward at the
    same time. v2 decides venue-wide each round: the most crowded gates choose
    first; a gate already receiving a diversion, or diverting itself, is not
    offered as a target; only if nothing else is viable may a source share a
    target, and then it sends at most ``shared_target_share`` (the decision's
    ``divert_fraction``).

``fresh_targets`` (K3)
    v1 checks the age of the source's data only. A target whose camera froze
    still looks as empty as it last was and keeps winning. v2 does not offer a
    monitored gate whose observation is older than ``max_observation_age_s``.

``nearest_with_room`` (W7)
    v1 scores distance only relative to the venue's largest distance, so an
    empty gate far away beats a gate with room close by, and diverted people
    walk further than needed. v2 sends people to the nearest gate (by walking
    distance) that clearly has room (score below ``room_score``); only when
    no such gate exists does v1's ranking decide.

``forecast`` (W2)
    v1 reacts only once a gate is HIGH, but an instruction takes about a
    minute to reach the crowd. v2 fits a line to the gate's score over the
    last ``forecast_window_s`` and starts a diversion early when the gate is
    at least MODERATE and is projected to reach HIGH within ``horizon_s``.

Tried and rejected on the evidence (see docs/14): latching without
coordination (overshoot near capacity: 35 -> 3,196 person-minutes), a slow
stepped share (under-reacted), showing v1's averaged duty cycle (lagged), and
shedding share when the target fills (no measurable effect).

Like v1, this module imports no vision or model code.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import (
    ActionType,
    CandidateEvaluation,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    Trend,
)
from crowdguard.decision.engine import DecisionEngine
from crowdguard.decision.routing import rank_candidates
from crowdguard.venue.topology import VenueTopology

_SAFETY_CODES = {"STALE_OBSERVATION", "LOW_CONFIDENCE"}


class V2Features(BaseModel):
    """Which v2 improvements are on. All on is "v2"; one off is an ablation."""

    steady_diversion: bool = True
    coordination: bool = True
    fresh_targets: bool = True
    nearest_with_room: bool = False   # tested and rejected; kept switchable for the record
    forecast: bool = True


class V2Config(BaseModel):
    features: V2Features = Field(default_factory=V2Features)

    exit_score: float = Field(
        default=0.6, ge=0, le=1,
        description="A held diversion continues while the source score is at or above this.")
    release_s: float = Field(
        default=20.0, ge=0,
        description="Seconds the score must stay below exit_score before the diversion stops.")
    min_on_s: float = Field(
        default=60.0, ge=0,
        description="A diversion is never withdrawn sooner than this after it starts: "
                    "instructions take about a minute to reach the crowd.")
    target_hold_s: float = Field(
        default=60.0, ge=0,
        description="Keep a new target this long unless it stops being viable or a "
                    "challenger beats it by the switching margin.")
    room_score: float = Field(
        default=0.45, ge=0, le=1,
        description="A target 'clearly has room' below this congestion score (just above "
                    "the MODERATE line).")
    horizon_s: float = Field(default=60.0, ge=0, description="How far ahead the forecast looks.")
    forecast_window_s: float = Field(default=30.0, gt=0, description="Score history used for the trend line.")
    room_needs_steady: bool = Field(
        default=True,
        description="A gate whose crowd is growing does not count as having room.")
    shared_target_share: float = Field(
        default=0.5, gt=0, le=1,
        description="Largest share a source may send to a target another source already uses.")


@dataclass
class _Hold:
    share: float = 0.0               # 0 = not diverting
    since: float = 0.0               # diversion started
    below_since: float | None = None
    target: str | None = None
    target_since: float = 0.0


class DecisionEngineV2(DecisionEngine):
    """v1 plus the fixes switched on in ``v2_config.features``."""

    def __init__(
        self,
        venue: VenueTopology,
        config: DecisionConfig | None = None,
        clock: Callable[[], float] | None = None,
        v2_config: V2Config | None = None,
    ) -> None:
        super().__init__(venue, config, clock)
        self.v2 = v2_config or V2Config()
        self._holds: dict[str, _Hold] = {}
        self._claims: dict[str, str] = {}
        self._history: dict[str, deque] = {}

    @property
    def features(self) -> V2Features:
        return self.v2.features

    def reset_hysteresis(self, gate_id: str | None = None) -> None:
        super().reset_hysteresis(gate_id)
        if gate_id is None:
            self._holds.clear()
        else:
            self._holds.pop(gate_id, None)

    # ------------------------------------------------------------------

    def evaluate(self, states: dict[str, CrowdState]) -> list[DecisionResult]:
        """One round. With coordination, the most crowded gates choose first."""
        if not self.features.coordination:
            return super().evaluate(states)
        now = self._clock()
        self._claims = {}
        order = sorted((g for g in self._venue.gate_ids if g in states),
                       key=lambda g: -states[g].congestion_score)
        results = {g: self._evaluate_gate(g, states[g], states, now) for g in order}
        with self._lock:
            self._evaluation_count += len(results)
        return [results[g] for g in self._venue.gate_ids if g in results]

    def _evaluate_gate(
        self,
        gate_id: str,
        state: CrowdState,
        all_states: dict[str, CrowdState],
        now: float,
    ) -> DecisionResult:
        v1 = super()._evaluate_gate(gate_id, state, all_states, now)
        if not (self.features.steady_diversion or self.features.coordination):
            return v1
        h = self._holds.setdefault(gate_id, _Hold())
        projected = self._project(gate_id, state, now)

        # Restricting inflow, or refusing on unsafe data, always wins.
        if v1.action is ActionType.RESTRICT_INFLOW or _SAFETY_CODES & set(v1.reason_codes):
            self._stand_down(h, gate_id)
            return v1

        diverting_now = v1.action is ActionType.DIVERT
        early = False
        if (self.features.forecast and not diverting_now and h.share == 0
                and state.congestion_level is CongestionLevel.MODERATE
                and projected >= self._config.congestion_thresholds.high_threshold):
            diverting_now = early = True
        if diverting_now:
            h.below_since = None
            if h.share == 0:
                h.share, h.since = 1.0, now
        elif h.share > 0 and self.features.steady_diversion:
            if state.congestion_score >= self.v2.exit_score:
                h.below_since = None
            elif h.below_since is None:
                h.below_since = now
            if (h.below_since is not None and now - h.below_since >= self.v2.release_s
                    and now - h.since >= self.v2.min_on_s):
                self._stand_down(h, gate_id)
                return v1
        else:
            self._stand_down(h, gate_id)
            return v1

        candidates = list(v1.candidate_evaluations) or rank_candidates(gate_id, all_states, self._venue, self._config)
        viable = [c for c in candidates if c.is_viable]
        if self.features.fresh_targets:
            max_age = self._config.max_observation_age_s
            viable = [c for c in viable
                      if not (c.is_monitored and c.gate_id in all_states
                              and all_states[c.gate_id].result_age_s > max_age)]
        cap = 1.0
        if self.features.coordination:
            free = [c for c in viable
                    if c.gate_id not in self._claims
                    and self._holds.get(c.gate_id, _Hold()).share <= 0]
            if free:
                viable = free
            elif viable:
                cap = self.v2.shared_target_share
        if not viable:
            self._stand_down(h, gate_id)
            return v1

        best, target = self._choose_target(viable, h, now)
        if self.features.coordination:
            self._claims[target] = gate_id
        with self._lock:
            self._active_diversions[gate_id] = (target, h.target_since)
        return self._divert_result(gate_id, state, best, target, candidates, cap,
                                   holding=not diverting_now, early=early, projected=projected)

    def _project(self, gate_id: str, state: CrowdState, now: float) -> float:
        """Score projected ``horizon_s`` ahead from a line fitted to recent scores."""
        hist = self._history.setdefault(gate_id, deque())
        hist.append((now, state.congestion_score))
        while hist and now - hist[0][0] > self.v2.forecast_window_s:
            hist.popleft()
        if len(hist) < 4 or hist[-1][0] - hist[0][0] < self.v2.forecast_window_s / 2:
            return state.congestion_score
        ts = [t for t, _ in hist]
        ys = [y for _, y in hist]
        tm, ym = sum(ts) / len(ts), sum(ys) / len(ys)
        var = sum((t - tm) ** 2 for t in ts)
        slope = sum((t - tm) * (y - ym) for t, y in zip(ts, ys)) / var if var > 0 else 0.0
        return state.congestion_score + max(0.0, slope) * self.v2.horizon_s

    def _stand_down(self, h: _Hold, gate_id: str) -> None:
        h.share, h.below_since, h.target = 0.0, None, None
        with self._lock:
            self._active_diversions.pop(gate_id, None)

    def _choose_target(self, viable: list[CandidateEvaluation], h: _Hold, now: float) -> tuple[CandidateEvaluation, str]:
        best = viable[0]
        if self.features.nearest_with_room:
            roomy = [c for c in viable if c.congestion_score < self.v2.room_score
                     and not (self.v2.room_needs_steady and c.trend is Trend.INCREASING)]
            if roomy:
                best = min(roomy, key=lambda c: (c.distance_m, -c.routing_score))
        incumbent = next((c for c in viable if c.gate_id == h.target), None)
        if incumbent is not None:
            if incumbent is best:
                return incumbent, incumbent.gate_id
            margin = best.routing_score - incumbent.routing_score
            if (self.features.nearest_with_room and best.congestion_score < self.v2.room_score
                    and best.distance_m < 0.7 * incumbent.distance_m):
                # a much shorter walk to a gate with room counts as a clear gain
                margin = max(margin, self._config.hysteresis.switching_margin)
            if now - h.target_since < self.v2.target_hold_s or margin < self._config.hysteresis.switching_margin:
                return incumbent, incumbent.gate_id
        h.target, h.target_since = best.gate_id, now
        return best, best.gate_id

    def _divert_result(
        self,
        gate_id: str,
        state: CrowdState,
        best: CandidateEvaluation,
        target: str,
        candidates: list[CandidateEvaluation],
        share: float,
        holding: bool,
        early: bool = False,
        projected: float = 0.0,
    ) -> DecisionResult:
        route = tuple(self._venue.perimeter_path(gate_id, target))
        note = "" if best.is_monitored else " Note: this gate has no camera, so its load is assumed rather than measured."
        if early:
            why = (f"{gate_id} is {state.congestion_level.value} (score {state.congestion_score:.2f}) and rising: "
                   f"projected to reach {projected:.2f} within {self.v2.horizon_s:.0f}s, about the time an "
                   f"instruction takes to reach the crowd, so the diversion starts now.")
        elif holding:
            why = (f"{gate_id} is easing (score {state.congestion_score:.2f}); the diversion is held until it "
                   f"stays below {self.v2.exit_score:.2f} for {self.v2.release_s:.0f}s, so the instruction stays steady.")
        else:
            why = (f"{gate_id} has {state.congestion_level.value} congestion (score {state.congestion_score:.2f}, "
                   f"{state.crowd_count:.0f} people) with {state.trend.value.lower()} density.")
        shared = " Another gate already sends people there, so only part of the crowd is sent." if share < 1 else ""
        reason = (f"{why} Send {share:.0%} of arriving people to {target}: {best.distance_m:.0f}m away along the "
                  f"perimeter, {best.congestion_level.value} congestion (score {best.congestion_score:.2f}), "
                  f"{best.capacity_utilization:.0%} of capacity used.{shared}{note}")
        codes = (("FORECAST_HIGH", "DIVERT_RECOMMENDED") if early else
                 ("DIVERSION_HELD", "DIVERT_RECOMMENDED") if holding else ("CONGESTED", "DIVERT_RECOMMENDED"))
        severity = state.congestion_level if state.congestion_level in (
            CongestionLevel.HIGH, CongestionLevel.CRITICAL) else CongestionLevel.HIGH
        return self._make_result(
            gate_id, ActionType.DIVERT, severity,
            target_gate_id=target,
            confidence=state.confidence,
            reason_codes=codes,
            reason=reason,
            candidates=tuple(candidates),
            route=route,
            divert_fraction=share,
        )
