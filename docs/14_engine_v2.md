# 14: Decision engine v2: what changed, what was rejected, and the results

Engine v2 (`crowdguard/decision/engine_v2.py`) is v1 plus four improvements, each behind its own switch (`V2Features`). Every change was kept only if it improved the weakness it targeted on a paired quick check (`scripts/compare_engines.py`, 30 seeds, seeds 0–29) without a meaningful loss elsewhere. Final numbers come from **different, held-out seeds** (1000–1099), so they are not tuned to the test.

v1 is unchanged and remains selectable (`decision_engine: v1 | v2` in the app config).

## 1. Additional v1 defect found during development

**The target cooldown never expired.** v1 records the time of the *latest* decision as the cooldown start, so while a gate keeps diverting (every 0.5 s), the 10 s cooldown restarts on every tick. A better target can only take over by beating the incumbent by the switching margin. Verified: 60 s of continuous decisions with a slightly better alternative never switched. (Scenario S06 passed only because it advanced the clock 10 s between two decisions.) v2 times the hold from the last *target change*.

## 2. Development record (30-seed paired checks)

Danger = person-minutes at ≥ 5 people/m²; switches = instruction changes per hour (a change of target or of share counts).

| Attempt | Idea | Result | Verdict |
| :--- | :--- | :--- | :--- |
| A | **Latch the diversion on** (start at HIGH, release below 0.55 for 30 s, min 60 s) | switches 10–20× lower; danger at 40 % compliance ~3× lower; **but near saturation 35 → 3,196, twin surge at 100 % compliance 78 → 3,288** | rejected alone |
| B | **Stepped share** (start 50 %, one step per minute) | under-reacted: worse than v1 under every stress setting | rejected |
| C | **Show v1's averaged on/off duty cycle as a share** (v1's flicker, with a minute of actuation delay, acts like a crude proportional control) | lag + rounding down: worse than v1 under stress (e.g. 40 % compliance single surge 9,059 → 15,172) | rejected |
| D | A + **shed share when the target fills** (score ≥ 0.6) | identical to A: protection never changed the outcome | rejected (no effect) |
| E | A + **venue-wide coordination** | diagnosis of A showed danger at gates targeted by several sources at once (herding cascades); with coordination: 2-min delay twin surge 1,502 → 41, 3-min delay stadium 1,125 → 229; small residual costs | **kept** |
| E′ | E with release at score 0.6 after 20 s | removed E's residual costs (near saturation 19 vs v1 35; twin surge at 100 % compliance 2 vs 78) | **kept (defaults)** |
| F | **Ignore targets with stale data** | new `frozen_target` scenario: v1 5,212 → v2 0 (v2 without it: 7,143); no effect elsewhere | **kept** |
| G | **Nearest gate with clear room** (walk less) | walking −20–35 %, but stadium entry with slow instructions 75 → 3,333 (2 min) and 233 → 7,818 (3 min); with "room = not filling" and a stricter threshold still 481–2,417 | rejected: unsafe when all gates fill together |
| H | **Forecast**: start when a MODERATE gate is projected to reach HIGH within 60 s | 3-min delay: single 4,718 → 0, twin 5,788 → 165, uneven 4,700 → 47; 40 % compliance: all → 0; p95 wait −10–25 %; cost: stadium at 3-min delay 233 → 853 (still < v1 1,125), switches somewhat higher | **kept** |
| — | **Escalate to restrict inflow** | needs a gate to divert *and* hold back at once; the action model allows one action per gate | deferred |

**Lesson for the paper:** the obvious fix for "too many instruction changes" (hysteresis) is harmful on its own. The flicker had been doing useful proportional work. Stability only pays off together with coordination between gates.

## 3. Engine v2 as shipped

| Switch | What it does | Fixes |
| :--- | :--- | :--- |
| `steady_diversion` | Start at 100 % exactly when v1 would; hold until the score stays below 0.6 for 20 s (never before 60 s); keep a target 60 s, then switch only for a clear gain | K2, W1 |
| `coordination` | Each round the most crowded gates choose first; a gate already receiving a diversion or diverting itself is not offered; if only shared targets remain, send at most 50 % | K1, W2, W4 |
| `fresh_targets` | Never divert to a monitored gate whose data is older than the 5 s limit | K3 |
| `forecast` | Fit a line to the last 30 s of score; start early when MODERATE and projected ≥ HIGH within 60 s | W2, W3 |
| `nearest_with_room` | (off) tested and rejected | – |

Decisions carry `divert_fraction`. The dashboard shows it ("Divert G1 → G3 · 50 %"), and the radio script says "send half of new arrivals…".

Known-weakness tests: v1 still fails K1–K4 (`xfail strict`); **v2 passes all four** (`tests/test_engine_v2.py`).

## 4. Final results (held-out seeds 1000–1099)

**Run:** 8 scenarios × 5 policies × 100 seeds, robustness sweeps on 4 scenarios (50 seeds per cell), and the ablation (50 seeds); 22,000 simulated runs, 45 min on 6 CPU cores. Everything is in `reports/decision_eval/engine_v2_vs_v1/` (`summary.md`, CSVs, figures, `run_info.json`).

### Default settings (compliance 60 %, 60 s actuation delay, 10 % count noise)

| Scenario | Danger v1 → **v2** (person-min) | Peak density v1 → **v2** (/m²) | p95 wait v1 → **v2** (s) | Instruction changes/h v1 → **v2** |
| :--- | ---: | ---: | ---: | ---: |
| Single-gate surge | 0 → **0** | 3.90 → **3.30** | 174 → **143** | 183 → **21** |
| Stadium entry | 0 → **0** | 3.61 → **2.96** | 173 → **152** | 268 → **32** |
| Twin surge | 0 → **0** | 3.95 → **3.31** | 190 → **160** | 383 → **46** |
| Near saturation | 61 → **0** | 4.15 → **3.77** | 222 → 268 ✗ | 1,458 → **407** |
| Festival waves | 0 → **0** | 4.26 → **3.49** | 236 → **216** | 130 → **31** |
| Degraded sensing | 0 → **0** | 3.91 → **3.33** | 174 → **144** | 185 → **21** |
| Uneven spacing | 0 → **0** | 3.93 → **3.28** | 189 → **157** | 181 → **22** |
| Frozen camera on filling target | **5,557 → 0** | 7.00 → **3.30** | 242 → **150** | 204 → **23** |

(No control: 26,000–235,000 person-minutes in every scenario.)

### Under stress (danger, person-minutes; v1 → **v2**, with the perfect-information reference)

| Stress | Single surge | Twin surge | Stadium entry | Uneven spacing |
| :--- | ---: | ---: | ---: | ---: |
| Instructions take **2 min** | 163 → **0** | 1,650 → **51** | 42 → **0** | 203 → **0** |
| Instructions take **3 min** | 8,544 → **10** (ref 41) | 19,142 → **126** (ref 465) | 1,183 → **672** (ref 521) | 8,492 → **42** (ref 4,298) |
| **40 %** follow instructions | 8,702 → **0** | 7,953 → **0** | 0 → 0 | 8,715 → **6** |
| **100 %** follow instructions | 0 → 0 | 58 → **0** | 0 → 0 | 0 → 0 |
| **20 %** follow instructions | 37,388 → 33,917 | 70,481 → 62,761 | 14,100 → **5,510** | 37,087 → 33,821 |
| **30 %** count error | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |

v2 is close to or better than the **perfect-information** reference under every stress tested (better in 3 of the 4 three-minute-delay cases), despite seeing only noisy, delayed camera counts.

### Ablation (each improvement switched off in turn)

Danger summed over scenarios (person-minutes); instruction changes per hour averaged.

| Variant | Default (8 scen.) | 3-min delay (4 scen.) | 40 % compliance (4 scen.) | Changes/h (default) |
| :--- | ---: | ---: | ---: | ---: |
| **v2, all on** | **0** | **851** | **6** | 75 |
| without steady diversion | 0 | 9,425 | 8,362 | 792 |
| without venue-wide coordination | 1,383 | 7,373 | 6 | 42 |
| without ignoring stale targets | 5,958 | 851 | 6 | 74 |
| without forecast | 63 | 16,238 | 7,375 | 36 |

**Every improvement is necessary somewhere:**
- steady diversion and forecast carry the robustness to slow instructions and low compliance;
- coordination prevents herding;
- stale-target checks prevent the frozen-camera failure.

The costs are visible too: coordination and forecast roughly double instruction changes compared with steady diversion alone, though still about 5× fewer than v1 on average.

### Costs and limits (state them)

- **Near saturation, 95th-percentile wait +21 % (222 → 268 s).** v2 diverts about twice as many people (6.9 → 12.5 %) to keep every gate below danger. Waits are traded for zero danger (v1: 61 person-minutes).
- **More walking in some scenarios** (uneven spacing 122 → 136 m, frozen target 95 → 134 m per diverted person). The walk-less change (G) was rejected as unsafe; walking distance remains an open weakness.
- **At 20 % compliance every policy fails.** Diversion alone cannot help when few people follow it; escalation to holding people back (deferred) is the remaining lever.
- **Stadium entry with a 3-min delay** still has 672 person-minutes (v1 1,183), about the perfect-information level (521).
- All results come from the macroscopic simulator with assumed parameters (docs/12 §C). Parameters were tuned on seeds 0–29 and the **same scenario set**; the held-out seeds remove seed overfitting, not scenario overfitting. A fresh scenario or a microscopic simulator would be the next check.

### Slide-ready statements

- "Engine v2 kept all eight simulated scenarios free of dangerous crowding (≥ 5 people/m²), including a camera freezing on a filling gate, where v1 allowed 5,557 person-minutes. It cut operator instruction changes by 4–9× and the 95th-percentile wait by 8–38 % in seven of eight scenarios (100 held-out runs each)."
- "When instructions take 3 minutes to reach the crowd, v2 reduces dangerous exposure by 43–99.9 % versus v1 (e.g. twin surge 19,142 → 126 person-minutes), close to or better than a policy with perfect information."
- "When only 40 % of people follow instructions, v2 keeps dangerous exposure at ~0 where v1 allowed ~8,000–8,700 person-minutes."
- "Ablation: each of v2's four rule changes is necessary in at least one condition; the intuitive fix for flickering instructions (hysteresis) is harmful on its own and only works combined with venue-wide coordination."
