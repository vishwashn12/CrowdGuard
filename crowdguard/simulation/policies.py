"""Control policies compared in the closed-loop evaluation.

* ``no_control``     - baseline: nobody is redirected.
* ``nearest_gate``   - the obvious manual heuristic: when a gate reads HIGH,
                       send everyone to the nearest gate that does not, else
                       restrict. Uses the same observations as the engine.
* ``engine_v1``      - the deployed DecisionEngine, unchanged, on a virtual
                       clock. DIVERT sends all (compliant) arrivals to the
                       target; RESTRICT_INFLOW holds arrivals back.
* ``informed``       - a heuristic reference with PERFECT information (true queues,
                       people already walking, true throughput) that sends
                       each arrival to the gate with the shortest expected
                       time-to-entry (walk + queue). Greedy, so NOT an
                       optimum or upper bound, and not
                       deployable; it shows what better information alone
                       can buy.
"""

from __future__ import annotations

from typing import Callable

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import ActionType, CrowdState, level_at_least
from crowdguard.decision.engine import DecisionEngine
from crowdguard.decision.engine_v2 import DecisionEngineV2, V2Config, V2Features
from crowdguard.simulation.crowd_model import (
    DANGER_DENSITY,
    WALK_SPEED_MS,
    Directive,
    Truth,
    throughput_factor,
)
from crowdguard.venue.topology import VenueTopology


class NoControl:
    name = "no_control"

    def reset(self, venue: VenueTopology, clock: Callable[[], float]) -> None:
        pass

    def decide(self, t, states, truth) -> dict[str, Directive]:
        return {}


class NearestGate:
    name = "nearest_gate"

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()

    def reset(self, venue: VenueTopology, clock: Callable[[], float]) -> None:
        self.venue = venue

    def decide(self, t, states: dict[str, CrowdState], truth: Truth) -> dict[str, Directive]:
        out: dict[str, Directive] = {}
        level = self.config.divert_above_level
        for g, s in states.items():
            if not self.venue.is_monitored(g) or not level_at_least(s.congestion_level, level):
                continue
            options = sorted(
                (self.venue.shortest_distance(g, o), o) for o, so in states.items()
                if o != g and not level_at_least(so.congestion_level, level)
            )
            out[g] = Directive("divert", options[0][1]) if options else Directive("restrict")
        return out


class EngineV1:
    """The deployed rule-based engine, exactly as configured in production."""

    name = "engine_v1"

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()

    def reset(self, venue: VenueTopology, clock: Callable[[], float]) -> None:
        self.engine = DecisionEngine(venue, self.config, clock=clock)

    def decide(self, t, states: dict[str, CrowdState], truth: Truth) -> dict[str, Directive]:
        out: dict[str, Directive] = {}
        for d in self.engine.evaluate(states):
            if d.action is ActionType.DIVERT:
                out[d.source_gate_id] = Directive("divert", d.target_gate_id, 1.0)
            elif d.action is ActionType.RESTRICT_INFLOW:
                out[d.source_gate_id] = Directive("restrict")
        return out


class EngineV2:
    """Engine v2 with a chosen set of improvements (all on by default)."""

    def __init__(self, name: str = "engine_v2", config: DecisionConfig | None = None,
                 v2_config: V2Config | None = None) -> None:
        self.name = name
        self.config = config or DecisionConfig()
        self.v2_config = v2_config or V2Config()

    def reset(self, venue: VenueTopology, clock: Callable[[], float]) -> None:
        self.engine = DecisionEngineV2(venue, self.config, clock=clock, v2_config=self.v2_config)

    def decide(self, t, states: dict[str, CrowdState], truth: Truth) -> dict[str, Directive]:
        out: dict[str, Directive] = {}
        for d in self.engine.evaluate(states):
            if d.action is ActionType.DIVERT:
                out[d.source_gate_id] = Directive("divert", d.target_gate_id, d.divert_fraction or 1.0)
            elif d.action is ActionType.RESTRICT_INFLOW:
                out[d.source_gate_id] = Directive("restrict")
        return out


def v2_variant(**features_off: bool) -> Callable[[], EngineV2]:
    """Factory for v2 with some features switched off (ablation)."""
    def make() -> EngineV2:
        feats = V2Features(**{k: False for k in features_off})
        name = "v2_no_" + "_".join(features_off) if features_off else "engine_v2"
        return EngineV2(name=name, v2_config=V2Config(features=feats))
    return make


class Informed:
    """Perfect-information balancer: a reference point, not a deployable rule."""

    name = "informed"
    MARGIN_S = 20.0

    def reset(self, venue: VenueTopology, clock: Callable[[], float]) -> None:
        self.venue = venue

    def decide(self, t, states, truth: Truth) -> dict[str, Directive]:
        ids = list(truth.queue)

        def wait(g: str) -> float:
            mu = truth.service_rate[g] * throughput_factor(truth.density[g])
            return (truth.queue[g] + truth.inflight[g] + truth.held[g]) / max(mu, 1e-6)

        waits = {g: wait(g) for g in ids}
        out: dict[str, Directive] = {}
        for g in ids:
            best_cost, best = min(
                (self.venue.shortest_distance(g, o) / WALK_SPEED_MS + waits[o], o) for o in ids if o != g
            )
            if waits[g] > best_cost + self.MARGIN_S:
                # Join-the-shortest-expected-time: every compliant arrival goes
                # where it will get in soonest, counting people already walking.
                out[g] = Directive("divert", best, 1.0)
            elif truth.density[g] >= 0.9 * DANGER_DENSITY:
                out[g] = Directive("restrict")
        return out


POLICIES: dict[str, Callable[[], object]] = {
    "no_control": NoControl,
    "nearest_gate": NearestGate,
    "engine_v1": EngineV1,
    "engine_v2": EngineV2,
    "informed": Informed,
}

POLICY_LABELS = {
    "no_control": "No control",
    "nearest_gate": "Nearest free gate",
    "engine_v1": "CrowdGuard engine v1",
    "engine_v2": "CrowdGuard engine v2",
    "informed": "Shortest-wait (perfect info)",
}
