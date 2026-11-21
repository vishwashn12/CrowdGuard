"""Quick paired comparison of engine v1 against engine v2 variants.

Used while developing v2: every scenario plus the stress settings that expose
v1's weaknesses, on identical seeds, with mean and 95% CI.

    python scripts/compare_engines.py --seeds 30
    python scripts/compare_engines.py --v2 '{"exit_score": 0.5}' --v2 '{"exit_score": 0.6}'

Each --v2 is a JSON object of V2Config fields (features as a nested object),
so parameter choices can be compared side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdguard.decision.engine_v2 import V2Config
from crowdguard.simulation.crowd_model import SimConfig, Simulation
from crowdguard.simulation.policies import POLICIES, EngineV2
from crowdguard.simulation.runner import mean_ci
from crowdguard.simulation.scenarios import SCENARIOS, get_scenario

SETTINGS = {
    "default": {},
    "delay 120s": {"actuation_delay_s": 120.0},
    "delay 180s": {"actuation_delay_s": 180.0},
    "comply 40%": {"compliance": 0.4},
    "comply 100%": {"compliance": 1.0},
}
STRESS_SCENARIOS = ["single_surge", "twin_surge", "stadium_entry", "uneven_spacing"]
METRICS = [("danger_person_s", 1 / 60, "danger"), ("p95_wait_s", 1, "p95"),
           ("switches_per_hour", 1, "sw/h"), ("extra_walk_m", 1, "walk")]


def job(args):
    scenario, label, v2json, seed, overrides = args
    cfg = replace(SimConfig(seed=seed), **overrides)
    if v2json is None:
        policy = POLICIES["engine_v1"]()
    else:
        policy = EngineV2(v2_config=V2Config.model_validate(json.loads(v2json)))
    m = Simulation(get_scenario(scenario), cfg).run(policy).metrics
    return scenario, label, seed, json.dumps(overrides), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--v2", action="append", default=None, help="V2Config JSON; repeatable")
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    variants = [("v1", None)] + [(f"v2#{i}" if len(args.v2 or []) > 1 else "v2", v)
                                 for i, v in enumerate(args.v2 or ["{}"])]
    jobs = []
    for sname, overrides in SETTINGS.items():
        scenarios = list(SCENARIOS) if sname == "default" else STRESS_SCENARIOS
        for sc in scenarios:
            for label, v in variants:
                for seed in range(args.seeds):
                    jobs.append((sc, label, v, seed, overrides))

    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        results = list(pool.map(job, jobs, chunksize=8))

    table: dict[tuple, list] = {}
    for sc, label, seed, ov, m in results:
        table.setdefault((ov, sc, label), []).append(m)

    for i, v in enumerate(args.v2 or []):
        print(f"v2#{i}: {v}")
    for sname, overrides in SETTINGS.items():
        ov = json.dumps(overrides)
        print(f"\n=== {sname} ===")
        print(f"{'scenario':17s} {'engine':6s} " + " ".join(f"{h:>15s}" for _, _, h in METRICS))
        scenarios = list(SCENARIOS) if sname == "default" else STRESS_SCENARIOS
        for sc in scenarios:
            for label, _ in variants:
                ms = table[(ov, sc, label)]
                cells = []
                for key, k, _ in METRICS:
                    mu, hw = mean_ci([x[key] * k for x in ms])
                    cells.append(f"{mu:8.0f} ±{hw:5.0f}")
                print(f"{sc:17s} {label:6s} " + " ".join(f"{c:>15s}" for c in cells))


if __name__ == "__main__":
    main()
