"""Macroscopic multi-gate crowd model.

Each gate is a queue in front of a bottleneck (turnstiles, a ghat entrance, a
station exit). People arrive at random, wait in the gate's approach area, and
are served at the gate's throughput. The approach area has a fixed size, so
the queue length sets a physical crowd density in people per square metre.

Two pieces of crowd science make congestion self-reinforcing, which is what
gives diversion its value:

* **Throughput falls when the approach is overcrowded** ("faster-is-slower",
  clogging at bottlenecks). Above an onset density the gate serves fewer
  people per second, down to a floor at very high density.
* **Walking speed falls with density** (Weidmann's fundamental diagram). The
  simulated cameras report this as the crowd velocity the engine sees.

Instructions are not instantaneous: an operator reads the recommendation,
radios staff and moves signage, so a change of instruction reaches the crowd
``actuation_delay_s`` later. A policy that flips its instruction every few
seconds therefore pays for it here, as it would on site.

Control acts on ARRIVALS only: a diversion asks newly arriving people to walk
along the perimeter to another gate; a restriction holds a share of arrivals
in a safe holding area upstream. Each person complies with probability
``compliance``. Diverted people join the target queue after walking and are
not diverted a second time.

Every constant below is an explicit, documented modelling assumption. They are
chosen to be plausible for a stadium-scale venue, not fitted to data.
"""

from __future__ import annotations

import math
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import CrowdState, MotionStatus
from crowdguard.decision.fusion import build_crowd_state, build_unmonitored_state
from crowdguard.decision.trend import TrendTracker
from crowdguard.venue.builder import build_ring_venue
from crowdguard.venue.topology import VenueTopology

# ── crowd physics (assumptions) ──────────────────────────────────────
FREE_SPEED_MS = 1.34        # Weidmann free walking speed, m/s
JAM_DENSITY = 5.4           # Weidmann jam density, people/m^2
WALK_SPEED_MS = 1.2         # walking speed of diverted people along the concourse
CAPACITY_DENSITY = 5.0      # "capacity" of a gate area = area * this density
DANGER_DENSITY = 5.0        # density treated as dangerous in the metrics, people/m^2
CLOG_ONSET = 3.5            # density at which gate throughput starts to fall
CLOG_FULL = 7.0             # density at which throughput reaches its floor
CLOG_FLOOR = 0.5            # throughput at very high density, as a share of nominal
MAX_DENSITY = 9.0           # physical ceiling; beyond it the queue spills outside the watched area

# ── camera model ─────────────────────────────────────────────────────
# Free-flowing crowd in the reference footage moves at ~75 px/s, which is the
# engine's max_expected_velocity, so 1.34 m/s maps to 75 px/s.
PX_PER_MS = 75.0 / FREE_SPEED_MS
OBSERVED_CONFIDENCE = 0.9   # live blend of CSRNet (1.0) and typical motion (0.8)


def weidmann_speed(density: float) -> float:
    """Walking speed (m/s) at a crowd density (people/m^2)."""
    if density <= 1e-6:
        return FREE_SPEED_MS
    if density >= JAM_DENSITY:
        return 0.0
    return FREE_SPEED_MS * (1.0 - math.exp(-1.913 * (1.0 / density - 1.0 / JAM_DENSITY)))


def area_density(queue: float, area_m2: float) -> float:
    """Density inside the watched approach area. A queue longer than the area
    can physically hold spills outside it, so density saturates."""
    return min(queue / area_m2, MAX_DENSITY)


def throughput_factor(density: float) -> float:
    """Share of nominal gate throughput available at a density."""
    if density <= CLOG_ONSET:
        return 1.0
    if density >= CLOG_FULL:
        return CLOG_FLOOR
    return 1.0 - (1.0 - CLOG_FLOOR) * (density - CLOG_ONSET) / (CLOG_FULL - CLOG_ONSET)


# ── scenario description ─────────────────────────────────────────────


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    service_rate: float          # nominal throughput, people/s
    area_m2: float               # approach area the camera watches
    monitored: bool = True

    @property
    def capacity(self) -> int:
        return int(round(self.area_m2 * CAPACITY_DENSITY))


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    description: str
    gates: tuple[GateSpec, ...]
    demand: Callable[[float], np.ndarray]   # people/s arriving at each gate at time t
    demand_end_s: float
    max_time_s: float
    radius_m: float = 120.0
    dropouts: tuple[tuple[str, float, float], ...] = ()   # (gate, start, end) camera outages
    initial_queue: tuple[float, ...] | None = None

    @property
    def gate_ids(self) -> list[str]:
        return [g.gate_id for g in self.gates]

    def venue(self) -> VenueTopology:
        n = len(self.gates)
        return VenueTopology(build_ring_venue(
            n,
            radius_m=self.radius_m,
            capacities={g.gate_id: g.capacity for g in self.gates},
            camera_bindings={g.gate_id: f"sim_{g.gate_id}" for g in self.gates if g.monitored},
            venue_name=self.title,
        ))


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1.0                  # simulation step, s
    decision_period: float = 1.0     # how often the policy is consulted, s
    obs_period: float = 2.0          # crowd-count refresh per camera, s (live: 0.5 fps)
    obs_latency_s: float = 1.0       # processing delay before an observation is usable
    count_noise: float = 0.10        # relative s.d. of the count estimate
    speed_noise: float = 0.10        # relative s.d. of the speed estimate
    compliance: float = 0.6          # probability a person follows an instruction
    restrict_strength: float = 0.5   # share of compliant arrivals held back when restricting
    actuation_delay_s: float = 60.0  # recommendation -> operator -> staff/signage -> crowd
    seed: int = 0


# ── control interface ────────────────────────────────────────────────


@dataclass(frozen=True)
class Directive:
    """What a policy asks of one gate for the next period."""

    action: str = "none"             # "none" | "divert" | "restrict"
    target: str | None = None
    share: float = 1.0               # share of arrivals asked to divert


@dataclass
class Truth:
    """Ground truth, available only to the perfect-information reference policy."""

    queue: dict[str, float]
    inflight: dict[str, float]
    held: dict[str, float]
    service_rate: dict[str, float]
    density: dict[str, float]


class Policy(Protocol):
    name: str

    def reset(self, venue: VenueTopology, clock: Callable[[], float]) -> None: ...

    def decide(self, t: float, states: dict[str, CrowdState], truth: Truth) -> dict[str, Directive]: ...


# ── results ──────────────────────────────────────────────────────────


@dataclass
class RunResult:
    scenario: str
    policy: str
    seed: int
    metrics: dict[str, float]
    series: dict[str, np.ndarray] | None = None
    events: list[tuple[float, str, str]] = field(default_factory=list)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, q * cum[-1], side="left").clip(0, v.size - 1)])


# ── the simulation ───────────────────────────────────────────────────


class Simulation:
    """One closed-loop run of a scenario under a policy."""

    def __init__(self, scenario: Scenario, config: SimConfig, decision_config: DecisionConfig | None = None):
        self.sc = scenario
        self.cfg = config
        self.dcfg = decision_config or DecisionConfig()
        self.venue = scenario.venue()
        self.ids = scenario.gate_ids
        self.spec = {g.gate_id: g for g in scenario.gates}
        # Independent random streams, so every policy faces the SAME arrivals
        # and noise for a given seed (common random numbers -> paired comparisons).
        ss = np.random.SeedSequence([config.seed, zlib.crc32(scenario.name.encode())])
        arrivals_ss, comply_ss, noise_ss = ss.spawn(3)
        self.rng_arrivals = np.random.default_rng(arrivals_ss)
        self.rng_comply = np.random.default_rng(comply_ss)
        self.rng_noise = np.random.default_rng(noise_ss)
        self.t = 0.0

    def clock(self) -> float:
        return self.t

    def run(self, policy: Policy, record: bool = False) -> RunResult:
        sc, cfg = self.sc, self.cfg
        ids = self.ids
        n = len(ids)
        dist = {(a, b): self.venue.shortest_distance(a, b) for a in ids for b in ids}

        queues: dict[str, deque] = {g: deque() for g in ids}     # [amount, origin_time]
        qlen = {g: 0.0 for g in ids}
        held: dict[str, deque] = {g: deque() for g in ids}
        held_len = {g: 0.0 for g in ids}
        inflight: list[tuple[float, str, float, float]] = []      # (due, target, amount, origin)
        if sc.initial_queue is not None:
            for g, q in zip(ids, sc.initial_queue):
                if q > 0:
                    queues[g].append([float(q), 0.0])
                    qlen[g] = float(q)

        trends = TrendTracker(self.dcfg)
        last_obs: dict[str, tuple[float, float, float]] = {}      # gate -> (count, velocity_px, t_obs)
        directives: dict[str, Directive] = {g: Directive() for g in ids}   # in force on the ground
        pending: deque = deque()                                            # (t_effective, gate, Directive)
        last_key: dict[str, tuple] = {g: ("none", None) for g in ids}

        policy.reset(self.venue, self.clock)

        waits: list[float] = []
        weights: list[float] = []
        served_total = 0.0
        arrived_total = float(sum(sc.initial_queue or ()))
        diverted_total = 0.0
        diverted_walk_m = 0.0
        held_total = 0.0
        danger_ps = 0.0
        danger_time = 0.0
        peak_density = 0.0
        switches = 0
        divert_on_s = 0.0
        restrict_on_s = 0.0
        clearance = None

        steps = int(math.ceil(sc.max_time_s / cfg.dt))
        rec_q = np.zeros((steps, n)) if record else None
        rec_act = np.zeros((steps, n), dtype=np.int8) if record else None
        events: list[tuple[float, str, str]] = []

        next_obs = 0.0
        next_decision = 0.0

        for step in range(steps):
            t = step * cfg.dt
            self.t = t

            # ---- observations (what the cameras report) ----
            if t >= next_obs:
                next_obs += cfg.obs_period
                for g in ids:
                    spec = self.spec[g]
                    if not spec.monitored or self._in_dropout(g, t):
                        continue
                    rho = area_density(qlen[g], spec.area_m2)
                    in_view = rho * spec.area_m2
                    count = max(0.0, in_view * (1.0 + self.rng_noise.normal(0.0, cfg.count_noise)))
                    vel = max(0.0, weidmann_speed(rho) * PX_PER_MS * (1.0 + self.rng_noise.normal(0.0, cfg.speed_noise)))
                    last_obs[g] = (count, vel, t)

            # ---- decisions ----
            if t >= next_decision:
                next_decision += cfg.decision_period
                states = self._states(last_obs, trends, t)
                truth = Truth(
                    queue=dict(qlen),
                    inflight={g: sum(a for _, tg, a, _ in inflight if tg == g) for g in ids},
                    held=dict(held_len),
                    service_rate={g: self.spec[g].service_rate for g in ids},
                    density={g: area_density(qlen[g], self.spec[g].area_m2) for g in ids},
                )
                new = policy.decide(t, states, truth)
                for g in ids:
                    d = new.get(g, Directive())
                    key = (d.action, d.target if d.action == "divert" else None)
                    if key != last_key[g] and not (key[0] == "none" and last_key[g][0] == "none"):
                        switches += 1
                        events.append((t, g, f"{d.action}->{d.target}" if d.action == "divert" else d.action))
                    last_key[g] = key
                    pending.append((t + cfg.actuation_delay_s, g, d))

            # instructions reach the ground after the actuation delay
            while pending and pending[0][0] <= t:
                _, g, d = pending.popleft()
                directives[g] = d

            # ---- arrivals and control ----
            if t < sc.demand_end_s:
                lam = sc.demand(t)
                for i, g in enumerate(ids):
                    k = int(self.rng_arrivals.poisson(max(0.0, lam[i]) * cfg.dt))
                    if k == 0:
                        continue
                    arrived_total += k
                    d = directives[g]
                    if d.action == "divert" and d.target and d.target != g:
                        moved = int(self.rng_comply.binomial(k, min(1.0, max(0.0, d.share)) * cfg.compliance))
                        if moved:
                            walk = dist[(g, d.target)]
                            inflight.append((t + walk / WALK_SPEED_MS, d.target, float(moved), t))
                            diverted_total += moved
                            diverted_walk_m += moved * walk
                            k -= moved
                    elif d.action == "restrict":
                        hold = int(self.rng_comply.binomial(k, cfg.restrict_strength * cfg.compliance))
                        if hold:
                            held[g].append([float(hold), t])
                            held_len[g] += hold
                            held_total += hold
                            k -= hold
                    if k:
                        queues[g].append([float(k), t])
                        qlen[g] += k

            # diverted people reach their new gate
            if inflight:
                still = []
                for item in inflight:
                    due, tgt, amount, origin = item
                    if due <= t:
                        queues[tgt].append([amount, origin])
                        qlen[tgt] += amount
                    else:
                        still.append(item)
                inflight = still

            # held people are released once the restriction is lifted
            for g in ids:
                if held_len[g] > 0 and directives[g].action != "restrict":
                    self._move(held[g], queues[g], self.spec[g].service_rate * cfg.dt, held_len, qlen, g)

            # ---- service and metrics ----
            for i, g in enumerate(ids):
                spec = self.spec[g]
                rho = area_density(qlen[g], spec.area_m2)
                peak_density = max(peak_density, rho)
                if rho >= DANGER_DENSITY:
                    danger_ps += rho * spec.area_m2 * cfg.dt
                capacity_now = spec.service_rate * throughput_factor(rho) * cfg.dt
                remaining = capacity_now
                q = queues[g]
                while remaining > 1e-9 and q:
                    amount, origin = q[0]
                    take = min(amount, remaining)
                    waits.append(t + cfg.dt - origin)
                    weights.append(take)
                    remaining -= take
                    if take >= amount - 1e-9:
                        q.popleft()
                    else:
                        q[0][0] = amount - take
                served = capacity_now - remaining
                qlen[g] = max(0.0, qlen[g] - served)
                served_total += served
                if record:
                    rec_q[step, i] = qlen[g]
                    rec_act[step, i] = {"none": 0, "divert": 1, "restrict": 2}[directives[g].action]

            if any(area_density(qlen[g], self.spec[g].area_m2) >= DANGER_DENSITY for g in ids):
                danger_time += cfg.dt
            divert_on_s += cfg.dt * sum(1 for g in ids if directives[g].action == "divert")
            restrict_on_s += cfg.dt * sum(1 for g in ids if directives[g].action == "restrict")

            if t >= sc.demand_end_s and clearance is None:
                left = sum(qlen.values()) + sum(held_len.values()) + sum(a for _, _, a, _ in inflight)
                if left < 0.5:
                    clearance = t + cfg.dt
                    if not record:
                        break

        w = np.asarray(waits)
        wt = np.asarray(weights)
        hours = max(self.t, 1.0) / 3600.0
        metrics = {
            "peak_density": peak_density,
            "danger_person_s": danger_ps,
            "danger_time_s": danger_time,
            "mean_wait_s": float((w * wt).sum() / wt.sum()) if wt.sum() > 0 else 0.0,
            "p95_wait_s": _weighted_percentile(w, wt, 0.95),
            "clearance_s": clearance if clearance is not None else sc.max_time_s,
            "unserved": max(0.0, arrived_total - served_total),
            "diverted_share": diverted_total / arrived_total if arrived_total else 0.0,
            "extra_walk_m": diverted_walk_m / diverted_total if diverted_total else 0.0,
            "held_share": held_total / arrived_total if arrived_total else 0.0,
            "switches_per_hour": switches / hours,
            "divert_gate_s": divert_on_s,
            "restrict_gate_s": restrict_on_s,
            "arrived": arrived_total,
            "served": served_total,
            "remaining": sum(qlen.values()) + sum(held_len.values()) + sum(a for _, _, a, _ in inflight),
        }
        series = None
        if record:
            last = steps if clearance is None else int(clearance / cfg.dt)
            series = {"t": np.arange(last) * cfg.dt, "queue": rec_q[:last], "action": rec_act[:last],
                      "capacity": np.array([self.spec[g].capacity for g in ids])}
        return RunResult(sc.name, policy.name, cfg.seed, metrics, series, events)

    # ---- helpers ----

    def _in_dropout(self, gate: str, t: float) -> bool:
        return any(g == gate and a <= t < b for g, a, b in self.sc.dropouts)

    @staticmethod
    def _move(src: deque, dst: deque, amount: float, src_len: dict, dst_len: dict, g: str) -> None:
        while amount > 1e-9 and src:
            a, origin = src[0]
            take = min(a, amount)
            dst.append([take, origin])
            src_len[g] -= take
            dst_len[g] += take
            amount -= take
            if take >= a - 1e-9:
                src.popleft()
            else:
                src[0][0] = a - take

    def _states(self, last_obs: dict, trends: TrendTracker, t: float) -> dict[str, CrowdState]:
        """Build CrowdStates exactly as the live orchestrator does."""
        states: dict[str, CrowdState] = {}
        for g in self.ids:
            spec = self.spec[g]
            if not spec.monitored:
                states[g] = build_unmonitored_state(
                    g, spec.capacity, config=self.dcfg, trend=trends.get_trend(g), timestamp=t)
                continue
            obs = last_obs.get(g)
            if obs is None:
                continue
            count, vel, t_obs = obs
            state = build_crowd_state(
                gate_id=g,
                gate_capacity=spec.capacity,
                crowd_count=count,
                velocity=vel,
                confidence=OBSERVED_CONFIDENCE,
                result_age_s=(t - t_obs) + self.cfg.obs_latency_s,
                has_motion=True,
                motion_status=MotionStatus.OK,
                trend=trends.get_trend(g),
                config=self.dcfg,
                timestamp=t,
                is_synthetic=True,
            )
            trends.update(g, state.congestion_score)
            states[g] = state
        return states
