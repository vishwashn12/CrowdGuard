# 12: Evaluating the congestion-control decisions

This is how CrowdGuard's rule-based diversion logic is tested, and the baseline results for the current engine (**engine v1**). Every later rule change is measured with the same suite, so the paper and slides can show *before vs after* on identical tests.

All three test layers are automatic: they produce pass/fail or numbers, with no manual checking.

| Layer | What it checks | Where | Claim it supports |
| :--- | :--- | :--- | :--- |
| A. Scenario suite | 15 scripted situations with the behaviour an operator would expect, plus 4 known weaknesses | `tests/test_decision_scenarios.py` | "The engine behaves as specified in these situations" |
| B. Invariants | Rules that must hold for **every** input, on 600 random venues (2–24 gates) | `tests/test_decision_invariants.py` | "No input makes the engine do something unsafe or unexplained" |
| C. Closed-loop simulation | Crowd outcomes when the engine controls a simulated venue, vs. baselines, with 95% CIs and stress sweeps | `crowdguard/simulation/`, `scripts/evaluate_decisions.py` | "Diversion reduces dangerous crowding by X% vs. no control, and holds up under noise, delay and partial compliance" |

The simulator is itself tested (`tests/test_simulation.py`): people are conserved, the physics behaves, results are reproducible per seed, and every policy faces identical arrivals.

---

## A. Scenario suite

Each scenario is a timeline of crowd states built through the production fusion path (`build_crowd_state`), fed to the real `DecisionEngine` on a virtual clock (so the 10 s cooldown can be crossed without waiting).

| ID | Situation | Expected behaviour | v1 |
| :--- | :--- | :--- | :---: |
| S01 | Quiet venue | No instructions | ✅ |
| S02 | One gate fills up | Divert in the same sweep it crosses HIGH, to an adjacent free gate, along the perimeter; other gates untouched | ✅ |
| S03 | Surge subsides | Diversion withdrawn | ✅ |
| S04 | Every gate full | RESTRICT_INFLOW everywhere, never DIVERT | ✅ |
| S05 | Target fills mid-diversion | Switch immediately, even inside the cooldown | ✅ |
| S06 | A marginally better gate appears | Keep the target during cooldown; move after | ✅ |
| S07 | Source data 8 s old | MONITOR only (stale) | ✅ |
| S08 | Confidence 0.2 | MONITOR only | ✅ |
| S09 | Motion missing | Still divert on density, confidence × 0.7 | ✅ |
| S10 | Gates without cameras | Never a source; observed gate preferred; unobserved used when it is the only option, and the reason says so | ✅ |
| S11 | Equal targets, different trends | Prefer the one that is clearing | ✅ |
| S12 | Equal targets, different distances | Prefer the nearer one | ✅ |
| S13 | Same input twice | Identical output | ✅ |
| S14 | 24-gate venue | Sweep well inside the 0.5 s decision period (< 50 ms) | ✅ |
| S15 | Any decision | Reason codes and a reason naming source and target | ✅ |

**Known weaknesses**, written as tests that are *expected to fail* (`xfail(strict=True)`). When a later engine fixes one, the suite flags it and the marker is removed deliberately, which records the improvement.

| ID | Weakness | Test | v1 |
| :--- | :--- | :--- | :---: |
| K1 | **Herding.** Gates decide independently, so two congested gates send their crowds to the same small target. | G1, G3 critical; G2 (room for ~80) is the only viable target; v1 sends both. | ❌ |
| K2 | **On/off chatter.** Hysteresis covers *which* target, not *whether* to divert. | Score alternating 0.63/0.67 for 20 s: v1 toggles 39 times; expected ≤ 2. | ❌ |
| K3 | **Stale target.** Only the source's data age is checked. A target whose camera froze 30 s ago still looks empty and wins. | Found while writing this suite. | ❌ |
| K4 | **All-or-nothing.** A diversion names a target but not how much of the crowd to send. | Decision carries no share. | ❌ |

## B. Invariants (600 random venues)

Generated from a fixed seed (`20261121`), covering 2–24 gates, any mix of observed/unobserved gates, counts up to 130% of capacity, any velocity, confidence, data age, trend and missing motion. For every decision:

- one decision per gate that has a state;
- only a congested, fresh, confident, **observed** gate ever diverts or restricts;
- RESTRICT_INFLOW only when no candidate is viable;
- a diversion never targets the source, a rejected gate or an unreachable gate, and (with no hysteresis history) always picks the top-scoring viable gate;
- the route walks the perimeter neighbour to neighbour;
- every decision has reason codes and a reason; confidence stays in [0, 1].

Also: the engine is deterministic; the congestion score never falls when people are added or rises when the crowd speeds up; levels match the thresholds exactly. **v1 passes all of them.**

## C. Closed-loop simulation

### Model

A macroscopic model of a ring venue (`crowdguard/simulation/crowd_model.py`). Each gate is a queue in front of a bottleneck:

| Element | Model | Value (assumption) |
| :--- | :--- | :--- |
| Arrivals | Poisson, time-varying rate per gate (scenario) | — |
| Gate throughput | Nominal rate; **falls when the approach is overcrowded** (faster-is-slower / clogging), linearly from 3.5 to 7 people/m², floor 50% | 10 turnstiles × ~660/h ≈ 1.8 people/s |
| Approach area | Area the camera watches; density = queue / area, capped at 9/m² (longer queues spill outside the view) | 100 m², which holds 500 at 5/m²; the 500 equals CrowdGuard's default capacity |
| Crowd speed | Weidmann fundamental diagram, reported to the engine as px/s (1.34 m/s ↔ 75 px/s, the engine's calibration) | — |
| Cameras | Count refreshed every 2 s (live: CSRNet at 0.5 fps), ±10% count noise, ±10% speed noise, 1 s processing latency | — |
| Engine input | `CrowdState`s built exactly as the live orchestrator does (same fusion, trend tracker, unmonitored-gate placeholder) | confidence 0.9 |
| Decisions | Every 1 s | live: 2 Hz |
| **Actuation delay** | An instruction reaches the crowd 60 s after it is issued (operator reads it, radios staff, signage moves) | 60 s |
| Compliance | Each arriving person follows an instruction with probability *p* | 0.6 |
| Diversion | Compliant arrivals walk along the perimeter at 1.2 m/s and join the target queue (not re-diverted) | — |
| Restriction | 50% of compliant arrivals held in a safe upstream area, released when lifted | — |

Every policy faces the **same arrivals and sensor noise** for a given seed (common random numbers), so comparisons are paired.

**Validity limits** (state in the paper): macroscopic, not agent-based. No geometry inside the approach area, no counter-flow, no panic. Parameters are plausible, not fitted. Danger is a density threshold, not an injury model. A microscopic check (e.g. JuPedSim) is future work.

### Policies compared

| Policy | Description |
| :--- | :--- |
| **No control** | Nobody is redirected. |
| **Nearest free gate** | The obvious manual rule: when a gate reads HIGH, send everyone to the nearest gate that doesn't; otherwise restrict. Same observations as the engine. |
| **CrowdGuard engine v1** | The deployed `DecisionEngine`, unchanged. DIVERT sends all compliant arrivals to the target; RESTRICT_INFLOW holds arrivals back. |
| **Shortest-wait (perfect info)** | Sees true queues, people already walking and true throughput; sends each arrival to the gate with the shortest walk + queue. Greedy, so it is **not** an optimum or upper bound, and not deployable: it shows what better information alone buys. |

### Scenarios

| Scenario | Setting |
| :--- | :--- |
| Single-gate surge | 8 gates at ~67% load; G1 gets +1.9 people/s for 15 min (a train unloading at the nearest gate) |
| Stadium entry peak | 8 gates; 24,000 fans over 45 min with a late peak; transit-side gates get 56% of arrivals |
| Twin surge around one free gate | G1 and G3 surge together; G2 between them is the obvious target for both (herding test) |
| Venue near saturation | Demand 8% above total capacity; diversion alone cannot fix it |
| Festival waves (ghats) | 6 entry points; every 10 min a wave hits the next ghat |
| Surge with degraded sensing | Single surge, but G5 and G6 have no camera and G2's camera freezes for 5 min while G2 is the natural target |
| Surge with uneven gate spacing | 6 gates; G1 surges; the nearly empty G2 is 450 m away, the busier G6 only 70 m (segments 450/120/120/120/120/70 m). Tests whether walking distance is weighed against queueing |

### Metrics

| Metric | Meaning |
| :--- | :--- |
| Peak density | Highest density at any gate (people/m²) |
| **Exposure at ≥5 people/m²** | Person-minutes spent at dangerous density: the main safety outcome |
| Time any gate ≥5/m² | Minutes with at least one gate dangerous |
| Mean / 95th-percentile time to entry | Arrival to passing the gate, including walking for diverted people and time held back |
| Time to clear | When the last arrival is through |
| Arrivals diverted / held | Share of all arrivals |
| Extra walk | Metres walked per diverted person |
| **Instruction changes per hour** | How often the instruction at a gate changes: operator and staff burden; chatter |

Robustness sweeps vary one setting at a time on four scenarios (single surge, twin surge, stadium entry, uneven spacing): compliance 0.2–1.0, actuation delay 0–180 s, count error 0–30%.

Venue distances: each scenario is a ring; walking distance between neighbouring gates is the even arc unless the scenario gives per-segment distances (uneven spacing). Diverted people walk the shorter way round **in metres**.

---

## Baseline results: engine v1

**Run.** 100 seeds per cell (2,800 runs over 7 scenarios), plus 50 seeds per sweep cell (11,200 runs over 4 scenarios); 21 min on 6 CPU cores. Re-run on 2026-11-21 after walking distances became editable: the six evenly spaced scenarios reproduced the previous run's numbers exactly (same seeds), and the uneven-spacing scenario was added. Everything is in `reports/decision_eval/engine_v1_baseline/`: `summary.md` (all tables), CSVs of every run, `run_info.json` (settings, commit) and figures. Values are mean ± 95% CI. Engine under test: v1, unchanged since commit `2ccd247` (the clock hook and the route-direction fix do not change its decisions on even rings).

### Headline (default settings: compliance 0.6, 60 s actuation delay, 10% count noise)

Exposure at ≥5 people/m², in person-minutes:

| Scenario | No control | Nearest free gate | **Engine v1** | Shortest-wait (perfect info) |
| :--- | ---: | ---: | ---: | ---: |
| Single-gate surge | 50,460 ± 298 | 0 | **0** | 0 |
| Stadium entry peak | 26,489 ± 560 | 66 ± 61 | **0** | 0 |
| Twin surge | 95,663 ± 465 | 669 ± 148 | **0** | 0 |
| Near saturation | 77,624 ± 1,767 | 2,903 ± 442 | **74 ± 52** | 2,177 ± 261 |
| Festival waves | 234,788 ± 6,444 | 0 | **0** | 0 |
| Degraded sensing | 50,414 ± 355 | 0 | **0** | 0 |
| Uneven spacing | 50,628 ± 342 | 0 | **0** | 0 |

95th-percentile time to entry, in seconds:

| Scenario | No control | Nearest free gate | **Engine v1** | Shortest-wait |
| :--- | ---: | ---: | ---: | ---: |
| Single-gate surge | 1,991 | 189 | **174** | 132 |
| Stadium entry peak | 882 | 217 | **173** | 192 |
| Twin surge | 1,938 | 210 | **190** | 168 |
| Near saturation | 1,185 | 313 | **219** | 299 |
| Festival waves | 1,236 | 241 | **237** | 234 |
| Degraded sensing | 1,993 | 188 | **175** | 132 |
| Uneven spacing | 2,072 | 195 | **190** | 179 |

Instruction changes per hour (whole venue):

| Scenario | Nearest free gate | **Engine v1** | Shortest-wait |
| :--- | ---: | ---: | ---: |
| Single-gate surge | 297 | **177** | 54 |
| Stadium entry peak | 401 | **254** | 53 |
| Twin surge | 602 | **386** | 88 |
| Near saturation | 1,737 | **1,426** | 194 |
| Festival waves | 129 | **129** | 75 |
| Degraded sensing | 295 | **184** | 58 |
| Uneven spacing | 298 | **180** | 26 |

Figures: `fig_main_danger_person_s.png`, `fig_main_peak_density.png`, `fig_main_p95_wait_s.png`, `fig_main_switches_per_hour.png`, `fig_timeseries_single_surge.png`.

### What the baseline shows

**Claims engine v1 supports (in this model):**

1. **Diversion works.** With 60% compliance and a 60 s delay, v1 removes essentially all exposure to dangerous density in every scenario. That is a 99.9–100% reduction vs. no control, with peak density 3.6–4.3 people/m² against the 9/m² ceiling. The 95th-percentile time to entry falls 80–91%.
2. **It beats the obvious manual rule where it matters.** When several gates are under pressure, v1 has clearly less danger than "send them to the nearest free gate": twin surge 0 vs 669, near saturation 74 vs 2,903, stadium entry 0 vs 66. It also has shorter waits and 18–40% fewer instruction changes. The difference comes from ranking candidates by congestion, headroom and trend instead of distance alone.
3. **It tolerates count error.** Up to 30% count noise leaves its exposure at ~0 in all three sweep scenarios.

**Weaknesses the baseline quantifies (targets for v2):**

| # | Finding | Evidence | Linked to |
| ---: | :--- | :--- | :--- |
| W1 | **Instruction churn.** v1 changes some gate's instruction 129–1,426 times an hour, i.e. every 2.5–28 s across the venue. No operator, radio channel or signage can follow that. The perfect-information rule needs 2–7× fewer changes and is as safe in five of six scenarios (not near saturation), so churn is not the price of safety. | `fig_main_switches_per_hour.png` | K2 (no on/off hysteresis) |
| W2 | **Purely reactive, so it breaks down as instructions take longer.** Exposure appears at a 120 s delay and grows fast at 180 s: single surge 8,609, twin surge 17,318 person-minutes. The perfect-information rule, which counts people already walking, stays near 0 (≤ 299). | `fig_sweep_actuation_delay_s.png` | no forecast; no in-flight accounting |
| W3 | **Needs ≥ 60% compliance.** At 40% compliance v1 leaves 9,317 person-minutes in the single surge, where the perfect-information rule has 0. At 20% every policy fails, and v1 never escalates to RESTRICT_INFLOW, because a viable target still "exists". | `fig_sweep_compliance.png` | K4 (all-or-nothing), no escalation |
| W4 | **Full compliance makes all-or-nothing overshoot.** In the twin surge, exposure rises from 0 at 60% compliance to 47 ± 44 at 100%: sending *everyone* moves the queue to the target. | `sens_compliance.csv` | K1, K4 |
| W5 | **Waits are longer than necessary.** The perfect-information rule reaches a 24% lower 95th-percentile wait in the single surge (132 vs 174 s) with similar diversion volume, so allocation, not information alone, leaves time on the table. | summary tables | K4, routing weights |
| W7 | **Distance is weighed relatively, not as walking time.** With uneven spacing, v1's diverted people walk 124 m on average against 87 m for both the nearest-gate rule and the perfect-information rule: some are sent 450 m to the emptiest gate instead of 70 m to a busier one with room. At a 180 s instruction delay v1 has 8,800 person-minutes of danger here, against 3,830 for the perfect-information rule. | `uneven_spacing` rows, delay sweep | routing weights; no walk-time model |
| W6 | **Restriction is never used.** "Arrivals held back" is 0% for v1 in every scenario, including near saturation, where demand exceeds capacity. | summary tables | RESTRICT only when *no* target is viable |

**Not shown by this model:** the stale-target flaw (K3). The frozen G2 camera in *degraded sensing* made no measurable difference (same outcomes as single surge), because G2 really had spare capacity during the freeze. K3 stays a logic defect, proven by its unit test, that this scenario did not punish. A scenario where the frozen target is *filling* would show it.

### What this means for the rule changes

Each weakness maps to one planned change, and each change has a test that must flip from ❌ to ✅:

| Change | Fixes | Must improve |
| :--- | :--- | :--- |
| On/off hysteresis: enter at HIGH, leave only after a dwell below MODERATE | K2, W1 | switches/hour, with no loss of safety |
| Venue-wide allocation that counts people already walking to each gate | K1, W2, W4 | exposure at 120–180 s delay; twin surge at 100% compliance |
| Proportional diversion ("send 40% to G3") | K4, W3, W4, W5 | exposure at 40% compliance; p95 wait |
| Short-horizon forecast (time-to-critical) | W2 | exposure at long delays |
| Escalate to RESTRICT when diversion is not enough | W3, W6 | exposure at 20% compliance; near saturation |
| Reject targets with stale data | K3 | K3 test; a new "frozen filling target" scenario |
| Choose targets by walk time + expected wait, in seconds (uses the real segment distances) | W7 | extra walk and p95 wait in uneven spacing; danger at long delays |

The comparison is re-run with `python scripts/evaluate_decisions.py --label engine_v2 --policies no_control,nearest_gate,engine_v1,engine_v2,informed`, which puts v1 and v2 side by side on identical seeds.

### Slide-ready statements (engine v1, simulation)

- "Across six simulated scenarios, CrowdGuard's diversion removed **> 99.9%** of exposure to dangerous crowd density (≥5 people/m²) and cut the 95th-percentile time to entry by **80–91%** compared with no control (100 seeds, 95% CI)."
- "Against a 'nearest free gate' rule, it reduced dangerous exposure in the twin-surge and near-saturation scenarios (**669 → 0** and **2,903 → 74** person-minutes) with 18–40% fewer instruction changes."
- "Limitations quantified: up to **1,426 instruction changes per hour**, and protection degrades once instructions take ≥ 2 minutes to reach the crowd or fewer than 60% of people comply. These are the targets for v2."

Always state the model's limits alongside these: a macroscopic simulation with assumed parameters, not field data.
