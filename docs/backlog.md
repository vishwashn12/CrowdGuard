# Backlog: agreed work, deferred

Work we have agreed to do but have parked. Newest decisions first. Each item says *why* it matters and how we will know it is done.

---

## 0. Done since the plan was written

- **Walking distance between gates** is editable per segment (setup, venue model, map, simulator); uneven-spacing scenario added; the v1 baseline was re-run with it (`reports/decision_eval/engine_v1_baseline`).

## 1. Decision engine v2: DONE 2026-11-21, see [14](14_engine_v2.md)

Kept: steady diversion, coordination, stale-target check, forecast. Rejected: nearest-with-room and three share schemes. **Still open:** escalation to hold people back (needs a combined divert + hold action), walking distance (W7).

### Original plan

**Why.** The v1 baseline ([12](12_congestion_control_evaluation.md)) shows diversion works, but it also has four failing known-weakness tests (K1–K4) and six measured weaknesses (W1–W6).

**Approach.** Build v2 *next to* v1 (v1 stays selectable), with each change behind its own switch, so the paper gets both **v1 vs v2** and an **ablation table** (v2 with one change switched off at a time). After each change: run the test suite plus a `--quick` evaluation and show before/after numbers.

| # | Change | Fixes | Done when |
| ---: | :--- | :--- | :--- |
| 1 | On/off hysteresis: start diverting at HIGH, stop only after ~30 s below MODERATE | K2, W1 | K2 passes; instruction changes/hour fall sharply with no extra danger |
| 2 | Reject targets whose data is stale | K3 | K3 passes; new "frozen camera on a filling gate" scenario |
| 3 | Proportional diversion ("send 40% to G3"), sized to source excess and target headroom | K4, W3, W4, W5 | K4 passes; better at 40% and 100% compliance |
| 4 | Venue-wide allocation that counts people already walking to each gate | K1, W2, W4 | K1 passes; twin surge; danger at 120–180 s delay |
| 5 | Short-horizon forecast (time until a gate becomes critical) | W2 | danger at 120–180 s delay |
| 6 | Escalate to RESTRICT_INFLOW when diversion cannot keep up | W3, W6 | danger at 20% compliance; near saturation |

Knock-on changes: decisions carry a share (the dashboard shows "→ G3 · 40%"). The live system switches to v2 only after it wins the full evaluation.

### Evaluation runs and time estimates

Measured basis: the v1 evaluation (4 policies, 10,800 simulated runs) took **16.4 min on 6 CPU cores**, i.e. ~4 min per policy or ~0.09 s wall-clock per run. One policy = 2,700 runs (6 scenarios × 100 seeds, plus 3 sweep scenarios × 14 settings × 50 seeds). v2 is assumed to cost 1.2–1.5× a v1 run.

| Run | Contents | Runs | Time (6 cores) |
| :--- | :--- | ---: | ---: |
| Per-change quick check | `--quick`, 5 policies, 20 seeds | ~2,700 | ~4–5 min each, ~25–30 min total |
| A. Ablation | v2 + 6 "v2 minus one change" variants, 6 scenarios × 100 seeds, no sweeps | ~4,200 | ~8–10 min |
| B. Final comparison | no control, nearest gate, v1, v2, shortest-wait; all sweeps | ~13,500 | ~20–25 min |
| **A + B (recommended)** | the paper's comparison plus the ablation table | ~17,700 | **~30–35 min** |
| C. Everything | all 11 variants with every sweep | ~29,700 | ~50–70 min |

Levers: 12 processes ≈ 1.4–1.6× faster (more heat; CPU only, no GPU). 50 seeds ≈ half the time with CIs ~1.4× wider. Baseline runs can be reused while the simulator is unchanged, but rerun everything once for the final paper numbers so all figures come from one run.

Commands:

```bash
python scripts/evaluate_decisions.py --quick --label v2_step1 --policies no_control,nearest_gate,engine_v1,engine_v2,informed
python scripts/evaluate_decisions.py --label engine_v2 --policies no_control,nearest_gate,engine_v1,engine_v2,informed
```

(An ablation option still has to be added to the script when v2 exists.)

---

## 2. Remove unwanted features (deferred 2026-11-21)

**Why.** Simplify the system before the paper and demo. Which features count as unwanted is still **to be decided with the user**.

---

## 3. Better inputs (optional, later)

Kept separate from v2 so that the v2 gains can be attributed to the rules alone.

- Line-crossing counts per gate from the existing tracks, giving arrival rate, throughput and queue.
- Per-camera calibration (4 ground points), giving people/m² and m/s instead of counts and px/s.
- Microscopic simulation check of the diversion results (e.g. JuPedSim).
