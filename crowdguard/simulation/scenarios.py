"""Evaluation scenarios.

Gate sizing (assumptions, stated once):
  * a gate of 10 turnstiles at ~660 people/hour each serves ~1.8 people/s;
  * its camera watches a ~100 m^2 approach area, which holds 500 people at
    5 people/m^2 - the same 500 CrowdGuard uses as its default capacity.

Scenarios are built by name inside each worker process, so demand functions
can be closures and nothing needs to be pickled.
"""

from __future__ import annotations

import math

import numpy as np

from crowdguard.simulation.crowd_model import GateSpec, Scenario


def _gates(n: int, rate: float, area: float, unmonitored: tuple[int, ...] = ()) -> tuple[GateSpec, ...]:
    return tuple(GateSpec(f"G{i + 1}", rate, area, monitored=(i + 1) not in unmonitored) for i in range(n))


def _window(t: float, start: float, end: float, ramp: float = 60.0) -> float:
    """1 inside [start, end], with linear ramps of `ramp` seconds either side."""
    if t <= start - ramp or t >= end + ramp:
        return 0.0
    if t < start:
        return (t - (start - ramp)) / ramp
    if t > end:
        return ((end + ramp) - t) / ramp
    return 1.0


def single_surge() -> Scenario:
    """A train unloads at the North gate: one gate gets a 15-minute surge."""
    base = np.full(8, 1.2)

    def demand(t: float) -> np.ndarray:
        lam = base.copy()
        lam[0] += 1.9 * _window(t, 300, 1200)
        return lam

    return Scenario(
        name="single_surge",
        title="Single-gate surge",
        description="8 gates at ~67% load; G1 receives +1.9 people/s for 15 min "
                    "(e.g. a train or bus convoy unloading at the nearest gate).",
        gates=_gates(8, 1.8, 100.0),
        demand=demand, demand_end_s=2400, max_time_s=5400,
    )


def stadium_entry() -> Scenario:
    """Pre-match entry: 24,000 fans over 45 min, skewed towards transit-side gates."""
    weights = np.array([0.21, 0.17, 0.10, 0.07, 0.06, 0.08, 0.13, 0.18])
    total, duration, peak = 24000.0, 2700.0, 0.72

    # Beta(3, 1.8)-shaped arrival curve peaking ~72% of the way to kick-off.
    a, b = 3.0, 1.8
    norm = math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    def demand(t: float) -> np.ndarray:
        x = t / duration
        if x <= 0 or x >= 1:
            return np.zeros(8)
        rate = total / duration * (x ** (a - 1)) * ((1 - x) ** (b - 1)) / norm
        return rate * weights

    del peak
    return Scenario(
        name="stadium_entry",
        title="Stadium entry peak",
        description="8 gates of 2.2 people/s; 24,000 fans arrive over 45 min with "
                    "a late peak; transit-side gates G1, G2, G8 get 56% of arrivals.",
        gates=_gates(8, 2.2, 120.0),
        demand=demand, demand_end_s=duration, max_time_s=6000,
    )


def twin_surge() -> Scenario:
    """Two gates either side of one free gate surge together (tests herding)."""
    base = np.array([1.2, 1.0, 1.2, 1.45, 1.45, 1.45, 1.45, 1.45])

    def demand(t: float) -> np.ndarray:
        lam = base.copy()
        w = _window(t, 300, 1200)
        lam[0] += 1.8 * w
        lam[2] += 1.8 * w
        return lam

    return Scenario(
        name="twin_surge",
        title="Twin surge around one free gate",
        description="G1 and G3 surge together; G2 between them is the obvious free "
                    "gate for both; the far side of the venue is already busy.",
        gates=_gates(8, 1.8, 100.0),
        demand=demand, demand_end_s=2400, max_time_s=5400,
    )


def near_saturation() -> Scenario:
    """Total demand slightly above total capacity: diversion cannot help much."""
    lam = np.array([2.30, 2.10, 1.70, 1.50, 1.60, 1.90, 2.20, 2.20])

    return Scenario(
        name="near_saturation",
        title="Venue near saturation",
        description="Gates loaded at 83-128% of throughput for 30 min; total "
                    "demand exceeds total capacity by ~8%.",
        gates=_gates(8, 1.8, 100.0),
        demand=lambda t: lam.copy(), demand_end_s=1800, max_time_s=5400,
    )


def festival_waves() -> Scenario:
    """Pilgrim waves at bathing ghats: surges rotate around six entry points."""
    base = np.full(6, 1.2)
    order = [0, 2, 4, 1, 3, 5]

    def demand(t: float) -> np.ndarray:
        lam = base.copy()
        for k, g in enumerate(order):
            start = 300 + 600 * k
            lam[g] += 2.6 * _window(t, start, start + 360)
        return lam

    return Scenario(
        name="festival_waves",
        title="Festival waves (ghats)",
        description="6 entry points of 2.0 people/s around a 200 m site at 60% "
                    "load; every 10 min a 6-minute wave of +2.6 people/s hits the next ghat.",
        gates=_gates(6, 2.0, 120.0),
        demand=demand, demand_end_s=4200, max_time_s=7200, radius_m=200.0,
    )


def degraded_sensing() -> Scenario:
    """The single surge with imperfect sensing: two gates have no camera, and
    the camera of G2 - the natural target - freezes for 5 minutes mid-surge."""
    s = single_surge()
    return Scenario(
        name="degraded_sensing",
        title="Surge with degraded sensing",
        description="As single_surge, but G5 and G6 have no camera and G2's camera "
                    "freezes from 10 to 15 min, while G2 is the natural target.",
        gates=_gates(8, 1.8, 100.0, unmonitored=(5, 6)),
        demand=s.demand, demand_end_s=s.demand_end_s, max_time_s=s.max_time_s,
        dropouts=(("G2", 600.0, 900.0),),
    )


SCENARIOS = {
    "single_surge": single_surge,
    "stadium_entry": stadium_entry,
    "twin_surge": twin_surge,
    "near_saturation": near_saturation,
    "festival_waves": festival_waves,
    "degraded_sensing": degraded_sensing,
}


def get_scenario(name: str) -> Scenario:
    return SCENARIOS[name]()
