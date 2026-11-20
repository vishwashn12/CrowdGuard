"""Monte Carlo runner: many seeds per (scenario, policy, setting), in parallel.

Every policy sees the same arrivals and sensor noise for a given seed (common
random numbers), so differences between policies are paired and their
confidence intervals are tight.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from crowdguard.simulation.crowd_model import SimConfig, Simulation
from crowdguard.simulation.policies import POLICIES
from crowdguard.simulation.scenarios import get_scenario

METRICS = (
    "peak_density", "danger_person_s", "danger_time_s", "mean_wait_s", "p95_wait_s",
    "clearance_s", "diverted_share", "extra_walk_m", "held_share", "switches_per_hour",
)


def run_one(job: tuple[str, str, int, dict]) -> dict:
    scenario, policy, seed, overrides = job
    cfg = replace(SimConfig(seed=seed), **overrides)
    result = Simulation(get_scenario(scenario), cfg).run(POLICIES[policy]())
    return {"scenario": scenario, "policy": policy, "seed": seed, **overrides, **result.metrics}


def run_grid(
    scenarios: list[str],
    policies: list[str],
    seeds: int,
    overrides: dict | None = None,
    processes: int | None = None,
) -> list[dict]:
    overrides = overrides or {}
    jobs = [(s, p, k, overrides) for s in scenarios for p in policies for k in range(seeds)]
    processes = processes or max(1, (os.cpu_count() or 2) // 2)
    if processes == 1:
        return [run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        return list(pool.map(run_one, jobs, chunksize=max(1, len(jobs) // (processes * 8))))


def _t95(n: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    except Exception:  # pragma: no cover - scipy is optional
        return 1.96


def mean_ci(values) -> tuple[float, float]:
    """Mean and half-width of its 95% confidence interval."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return float(x.mean()) if x.size else float("nan"), float("nan")
    return float(x.mean()), _t95(x.size) * float(x.std(ddof=1)) / math.sqrt(x.size)


def summarise(rows: list[dict], keys: tuple[str, ...] = ("scenario", "policy")) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in groups.items():
        row = dict(zip(keys, key))
        row["n"] = len(rs)
        for m in METRICS:
            mu, hw = mean_ci([r[m] for r in rs])
            row[m] = mu
            row[m + "_ci"] = hw
        out.append(row)
    return out


def paired_change(rows: list[dict], scenario: str, metric: str, policy: str, baseline: str,
                  extra: dict | None = None) -> tuple[float, float, float]:
    """Mean paired difference (policy - baseline), its 95% CI half-width, and
    the relative change of the means. Pairs share a seed."""
    extra = extra or {}

    def pick(p):
        return {r["seed"]: r[metric] for r in rows
                if r["scenario"] == scenario and r["policy"] == p
                and all(r.get(k) == v for k, v in extra.items())}

    a, b = pick(policy), pick(baseline)
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    mu, hw = mean_ci(diffs)
    base_mean = float(np.mean([b[s] for s in seeds])) if seeds else float("nan")
    rel = mu / base_mean if base_mean else float("nan")
    return mu, hw, rel


def config_dict() -> dict:
    return asdict(SimConfig())
