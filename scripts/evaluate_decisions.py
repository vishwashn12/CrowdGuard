"""Closed-loop evaluation of congestion-control policies.

    python scripts/evaluate_decisions.py                  # full run, 100 seeds
    python scripts/evaluate_decisions.py --quick          # 20 seeds, for iterating
    python scripts/evaluate_decisions.py --label engine_v2 --policies no_control,nearest_gate,engine_v1,engine_v2,informed

Writes to reports/decision_eval/<label>/:
    main_runs.csv, sens_*.csv     every run (one row per scenario x policy x seed x setting)
    summary.csv, summary.md       means with 95% CIs, paired changes vs. no control
    fig_*.png                     slide-ready figures
    run_info.json                 settings, seeds, versions, for reproducibility

CPU only; no GPU. Uses half the CPU cores by default (--processes to change).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from crowdguard.simulation.crowd_model import DANGER_DENSITY, SimConfig, Simulation
from crowdguard.simulation.policies import POLICIES, POLICY_LABELS
from crowdguard.simulation.runner import (
    METRICS, config_dict, mean_ci, paired_change, run_grid, summarise,
)
from crowdguard.simulation.scenarios import SCENARIOS, get_scenario

# Colour follows the policy, never its rank (validated palette, light surface).
COLORS = {
    "no_control": "#8a8a85",
    "nearest_gate": "#eb6834",
    "engine_v1": "#2a78d6",
    "informed": "#1baf7a",
    "engine_v2": "#6250d6",
}
INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e6e5e0", "#fcfcfb"

METRIC_LABELS = {
    "peak_density": ("Peak density", "people/m²"),
    "danger_person_s": ("Exposure at ≥5 people/m²", "person-minutes"),
    "danger_time_s": ("Time any gate ≥5 people/m²", "minutes"),
    "mean_wait_s": ("Mean time to entry", "s"),
    "p95_wait_s": ("95th-percentile time to entry", "s"),
    "clearance_s": ("Time to clear all arrivals", "min"),
    "diverted_share": ("Arrivals diverted", "%"),
    "extra_walk_m": ("Extra walk per diverted person", "m"),
    "held_share": ("Arrivals held back", "%"),
    "switches_per_hour": ("Instruction changes", "per hour"),
}
SCALE = {"danger_person_s": 1 / 60, "danger_time_s": 1 / 60, "clearance_s": 1 / 60,
         "diverted_share": 100, "held_share": 100}

SWEEPS = {
    "compliance": [0.2, 0.4, 0.6, 0.8, 1.0],
    "actuation_delay_s": [0.0, 30.0, 60.0, 120.0, 180.0],
    "count_noise": [0.0, 0.1, 0.2, 0.3],
}
SWEEP_LABELS = {
    "compliance": "Share of people who follow instructions",
    "actuation_delay_s": "Delay before an instruction reaches the crowd (s)",
    "count_noise": "Crowd-count error (relative s.d.)",
}
SWEEP_SCENARIOS = ["single_surge", "twin_surge", "stadium_entry", "uneven_spacing"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(metric: str, mu: float, hw: float | None = None) -> str:
    k = SCALE.get(metric, 1)
    digits = 2 if metric == "peak_density" else 0 if abs(mu * k) >= 10 else 1
    s = f"{mu * k:.{digits}f}"
    if hw is not None and np.isfinite(hw):
        s += f" ± {hw * k:.{digits}f}"
    return s


# ── figures ──────────────────────────────────────────────────────────


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


def fig_main(summary: list[dict], policies: list[str], scenarios: list[str], metric: str, out: Path) -> None:
    plt = _style()
    label, unit = METRIC_LABELS[metric]
    k = SCALE.get(metric, 1)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    width = 0.8 / len(policies)
    x = np.arange(len(scenarios))
    for i, p in enumerate(policies):
        rows = {r["scenario"]: r for r in summary if r["policy"] == p}
        vals = np.array([rows[s][metric] * k for s in scenarios])
        errs = np.array([rows[s][metric + "_ci"] * k for s in scenarios])
        ax.bar(x + (i - (len(policies) - 1) / 2) * width, vals, width * 0.92, yerr=errs,
               color=COLORS.get(p, "#999"), label=POLICY_LABELS.get(p, p),
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    if metric == "peak_density":
        ax.axhline(DANGER_DENSITY, color="#c62828", lw=1.2, ls="--")
        ax.text(len(scenarios) - 0.5, DANGER_DENSITY, " danger 5/m²", color="#c62828", va="bottom", ha="right", fontsize=9)
    import textwrap
    ax.set_xticks(x, [textwrap.fill(get_scenario(s).title, 16) for s in scenarios], rotation=0, fontsize=9.5)
    ax.set_ylabel(unit)
    ax.set_title(f"{label} by scenario (mean, 95% CI)", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(policies), loc="lower center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_sweep(rows: list[dict], param: str, policies: list[str], metric: str, out: Path) -> None:
    plt = _style()
    k = SCALE.get(metric, 1)
    controlled = [p for p in policies if p != "no_control"]
    fig, axes = plt.subplots(1, len(SWEEP_SCENARIOS), figsize=(4 * len(SWEEP_SCENARIOS), 4.2), sharey=False)
    for ax, sc in zip(axes, SWEEP_SCENARIOS):
        base = [r[metric] for r in rows if r["scenario"] == sc and r["policy"] == "no_control"]
        for p in controlled:
            xs, ys, es = [], [], []
            for v in SWEEPS[param]:
                vals = [r[metric] for r in rows if r["scenario"] == sc and r["policy"] == p and r[param] == v]
                if vals:
                    mu, hw = mean_ci(vals)
                    xs.append(v); ys.append(mu * k); es.append(hw * k)
            ax.errorbar(xs, ys, yerr=es, color=COLORS.get(p, "#999"), lw=2, marker="o", ms=5,
                        capsize=2, label=POLICY_LABELS.get(p, p))
        title = get_scenario(sc).title
        if base:
            title += f"\n(no control: {np.mean(base) * k:,.0f})"
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel(SWEEP_LABELS[param], fontsize=9)
        ax.set_xticks(SWEEPS[param])
        ax.set_ylim(bottom=0)
    label, unit = METRIC_LABELS[metric]
    axes[0].set_ylabel(f"{label}\n({unit})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(controlled), loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_timeseries(policies: list[str], out: Path, scenario: str = "single_surge", seed: int = 0) -> None:
    """Density at the surging gate and its neighbours over time, per policy."""
    plt = _style()
    sc = get_scenario(scenario)
    fig, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 3.6), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), policies):
        res = Simulation(sc, SimConfig(seed=seed)).run(POLICIES[p](), record=True)
        t = res.series["t"] / 60
        area = np.array([g.area_m2 for g in sc.gates])
        dens = np.minimum(res.series["queue"] / area, 9.0)
        # gates are the entities here, so they get their own hues, not policy colours
        for idx, name, col in ((0, "G1 (surge)", INK), (1, "G2", "#6250d6"), (7, "G8", "#eda100")):
            ax.plot(t, dens[:, idx], color=col, lw=1.8, label=name)
        ax.axhline(DANGER_DENSITY, color="#c62828", lw=1, ls="--")
        divert = res.series["action"][:, 0] == 1
        ax.fill_between(t, 0, 9.3, where=divert, color=COLORS["engine_v1"], alpha=0.08, lw=0)
        ax.set_title(POLICY_LABELS.get(p, p), loc="left", fontsize=11)
        ax.set_xlabel("minutes")
        ax.set_xlim(0, 45)
        ax.set_ylim(0, 9.3)
    np.atleast_1d(axes)[0].set_ylabel("density (people/m²)")
    np.atleast_1d(axes)[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"{sc.title}: density over time, seed {seed} (shaded = G1 diversion in force)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# ── report ───────────────────────────────────────────────────────────


def markdown(summary, rows, policies, scenarios, sweeps, info) -> str:
    out = [f"# Decision-policy evaluation: `{info['label']}`", ""]
    out.append(f"{info['seeds']} seeds per cell; values are mean ± 95% CI half-width. "
               f"Generated {info['generated']} by `scripts/evaluate_decisions.py`.")
    out.append("")
    out.append("Policies: " + "; ".join(f"**{POLICY_LABELS.get(p, p)}** (`{p}`)" for p in policies) + ".")
    out.append("")
    key = ["peak_density", "danger_person_s", "p95_wait_s", "switches_per_hour", "diverted_share", "held_share"]
    for s in scenarios:
        sc = get_scenario(s)
        out += [f"## {sc.title} (`{s}`)", "", sc.description, ""]
        head = "| Policy | " + " | ".join(f"{METRIC_LABELS[m][0]} ({METRIC_LABELS[m][1]})" for m in key) + " |"
        out += [head, "|" + " :--- |" + " ---: |" * len(key)]
        for p in policies:
            r = next(x for x in summary if x["scenario"] == s and x["policy"] == p)
            out.append(f"| {POLICY_LABELS.get(p, p)} | " + " | ".join(fmt(m, r[m], r[m + '_ci']) for m in key) + " |")
        out.append("")
        if "no_control" in policies:
            for p in policies:
                if p == "no_control":
                    continue
                bits = []
                for m in ("danger_person_s", "p95_wait_s"):
                    mu, hw, rel = paired_change(rows, s, m, p, "no_control")
                    if np.isfinite(rel):
                        bits.append(f"{METRIC_LABELS[m][0].lower()} {rel * 100:+.0f}%")
                if bits:
                    out.append(f"- {POLICY_LABELS.get(p, p)} vs no control (paired): " + ", ".join(bits))
            out.append("")
    if sweeps:
        out += ["## Robustness sweeps", "",
                "Exposure at ≥5 people/m² (person-minutes) as one setting varies, others at default.", ""]
        for param, srows in sweeps.items():
            out += [f"### {SWEEP_LABELS[param]}", ""]
            vals = SWEEPS[param]
            out += ["| Scenario | Policy | " + " | ".join(str(v) for v in vals) + " |",
                    "| :--- | :--- |" + " ---: |" * len(vals)]
            for s in SWEEP_SCENARIOS:
                for p in policies:
                    cells = []
                    for v in vals:
                        xs = [r["danger_person_s"] for r in srows if r["scenario"] == s and r["policy"] == p and r[param] == v]
                        cells.append(fmt("danger_person_s", *mean_ci(xs)) if xs else "")
                    out.append(f"| {get_scenario(s).title} | {POLICY_LABELS.get(p, p)} | " + " | ".join(cells) + " |")
            out.append("")
    out += ["## Settings", "", "```json", json.dumps(info["sim_config"], indent=2), "```", ""]
    return "\n".join(out)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="engine_v1_baseline")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--sweep-seeds", type=int, default=None, help="seeds per sweep cell (default: seeds // 2)")
    ap.add_argument("--quick", action="store_true", help="20 seeds, 10 per sweep cell")
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--policies", default="no_control,nearest_gate,engine_v1,informed")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "decision_eval"))
    args = ap.parse_args()

    seeds = 20 if args.quick else args.seeds
    sweep_seeds = 10 if args.quick else (args.sweep_seeds or max(10, seeds // 2))
    policies = args.policies.split(",")
    scenarios = args.scenarios.split(",")
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"main grid: {len(scenarios)} scenarios x {len(policies)} policies x {seeds} seeds")
    rows = run_grid(scenarios, policies, seeds, processes=args.processes)
    write_csv(out / "main_runs.csv", rows)
    summary = summarise(rows)
    write_csv(out / "summary.csv", summary)

    sweeps: dict[str, list[dict]] = {}
    if not args.no_sweeps:
        for param, values in SWEEPS.items():
            print(f"sweep {param}: {values}")
            srows: list[dict] = []
            for v in values:
                srows += run_grid(SWEEP_SCENARIOS, policies, sweep_seeds, overrides={param: v}, processes=args.processes)
            sweeps[param] = srows
            write_csv(out / f"sens_{param}.csv", srows)

    print("figures")
    for m in ("peak_density", "danger_person_s", "p95_wait_s", "switches_per_hour"):
        fig_main(summary, policies, scenarios, m, out / f"fig_main_{m}.png")
    for param, srows in sweeps.items():
        fig_sweep(srows, param, policies, "danger_person_s", out / f"fig_sweep_{param}.png")
    fig_timeseries([p for p in policies if p in ("no_control", "nearest_gate", "engine_v1", "engine_v2")],
                   out / "fig_timeseries_single_surge.png")

    info = {
        "label": args.label, "seeds": seeds, "sweep_seeds": sweep_seeds, "policies": policies,
        "scenarios": scenarios, "sim_config": config_dict(), "sweeps": SWEEPS,
        "generated": time.strftime("%Y-%m-%d %H:%M"), "runtime_s": round(time.time() - t0, 1),
        "git_commit": git_commit(), "python": platform.python_version(), "numpy": np.__version__,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary, rows, policies, scenarios, sweeps, info), encoding="utf-8")
    print(f"done in {info['runtime_s']} s -> {out}")


if __name__ == "__main__":
    main()
