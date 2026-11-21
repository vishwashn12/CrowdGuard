# 13: Operator instructions and camera calibration

Two changes made on 2026-11-21 after the UI redesign. They change what the operator sees and the units the congestion score is computed in.

## 1. Operator instructions

### Problem

The alert rules produced long, mixed output. A typical action was 30–40 words. Engineering terms ("px/s", "congestion score 67%", "raise anomaly_stride") reached the operator. One diversion appeared three times (gate action, "Diversion active" alert, banner). The crush rule's text ("redirect to the nearest alternative gate") could contradict the engine's chosen target.

### Design (`crowdguard/alerts/commands.py`)

Two channels, a closed vocabulary of five verbs, and a fixed shape for every item:

| Channel | Verbs | Source | Rule |
| :--- | :--- | :--- | :--- |
| **Commands**: what to do with the crowd at a gate | STOP ENTRY, HOLD OUTSIDE, DIVERT, GET READY | engine decisions + congestion alerts | at most **one per gate**; most urgent wins (STOP > HOLD > DIVERT > GET READY) |
| **Incidents**: where to send staff | SEND STAFF | anomaly alerts (person down, walking against the crowd, many people affected) | one per gate and kind |

Every item has:
- a **headline**: verb + gate (+ → target), e.g. *Divert G1 → G3*;
- a **reason** of at most 3 words, e.g. *crush risk*;
- a one-line **detail** with the numbers;
- a **radio script** to read out word for word, e.g. *"G1 team: send new arrivals to G3 by the outer path."*

Further rules:
1. **The engine is the only source of a destination.** A STOP ENTRY at a gate the engine is diverting carries the engine's target, so the operator never sees two destinations for one gate.
2. **System alerts are not shown to the operator at all** (tracking degraded, velocity uncalibrated, motion unavailable). They stay in the alert log and the diagnostics panel.
3. **No acknowledgement step** (decided with the user).
4. **Duplicates are removed:** "Diversion active" and "All clear" no longer appear as items.

Mapping:

| Source | Becomes |
| :--- | :--- |
| Engine DIVERT | DIVERT → target (critical if the source is CRITICAL) |
| Engine RESTRICT_INFLOW | HOLD OUTSIDE · "no free gate" |
| CRUSH_RISK, GATE_OVERLOAD | STOP ENTRY · "crush risk" / "over capacity" |
| ALL_GATES_FULL | HOLD OUTSIDE · "venue full" |
| BOTTLENECK_FORMING, SLOW_FLOW_WARNING, MONITOR_SURGE, IMBALANCED_LOAD | GET READY · "filling fast" / "flow slowing" / "inflow rising" / "uneven load" |
| PERSON_DOWN, COUNTERFLOW_DETECTED, ANOMALY_CLUSTER | SEND STAFF · "person down" / "walking against crowd" / "many people affected" |

**Dashboard.**
- The right panel holds Commands, then Incidents, plus a Log tab in the same vocabulary. Tapping a row shows its radio script and detail.
- The banner shows only the top **critical** item, with its radio script.
- A new critical item chimes. There are no toast pop-ups for instructions, because they duplicated the panel.

Tests: `tests/test_commands.py`. They cover the vocabulary, length limits (reason ≤ 3 words, headline ≤ 5, radio ≤ 20), one command per gate, a single destination, hidden system alerts, and stable "since" times.

## 2. Camera calibration (real units)

### Problem

"Capacity" was the denominator of the whole congestion score, yet nobody can know it. It also isn't a gate property: CSRNet counts the whole camera view. Speed was in px/s, which depends on zoom and distance.

### Method (`crowdguard/utils/calibration.py`)

The operator clicks the four corners of a rectangle **on the ground** in the gate's first frame, in order around it, and enters its real width (1 → 2) and length (2 → 3) in metres. From that:

| Quantity | How |
| :--- | :--- |
| Homography | 4-point DLT, image pixels → metres on the ground plane |
| Counting zone | the marked quadrilateral |
| People in zone | **max**(CSRNet density map summed over the zone, people tracked with feet in the zone); see the finding below |
| Density | people in zone ÷ area (people/m²) |
| Walking speed | median over tracked people in the zone of ‖Δ(foot point on the ground)‖ × analysed fps (m/s); a person's foot point is the bottom-centre of their box |
| Capacity | area × 5 people/m² (`calibrated_capacity_density`) |
| Free-flow speed | 1.34 m/s (Weidmann), replacing the px/s `max_expected_velocity` |

The congestion formula is unchanged; only its inputs change units. A calibrated gate is therefore scored on **exactly the scale the closed-loop simulator uses** ([12](12_congestion_control_evaluation.md)): capacity = area × 5/m², and free walking = 1.34 m/s maps to the speed term's ceiling. The live system and the evaluation now agree.

Alert-rule speed thresholds, written in px/s, are applied to calibrated gates on the same scale (1.34 m/s ↔ 75 px/s).

**Uncalibrated gates** keep the old behaviour, with a default capacity of **300 people** (was 500). Gates without a camera also assume 300.

Calibrations are stored per camera in `uploads/calibrations.json` and survive restarts. Replacing a gate's footage starts it uncalibrated.

### Finding: CSRNet under-counts near the camera

Measured on 4 frames of the G1 demo clip, with the zone over the near half of the corridor:

| | Whole frame | Inside the near zone |
| :--- | ---: | ---: |
| CSRNet (ShanghaiTech A) | 133–148 | **9–14** |
| YOLOv8m-pose detections | 60–73 | **27–38** |

CSRNet places ~93% of its count on the dense far end of the corridor and under-counts large people near the camera by about 3×. The detector has the opposite weakness (it misses small distant people). Inside a near-camera zone the detector is the better count. The zone count therefore takes the larger of the two, the conservative choice. **Decision (2026-11-21):** keep the larger-of-the-two rule; a hand-counted validation was judged unnecessary for now. *Paper note:* the evidence is one camera and four frames. A proper evaluation needs annotated zone counts (see [10](10_limitations_future_work.md), item 7).

### Limits

- **The size is operator-entered.** An over-estimated rectangle inflates speed and deflates density. In the demo test, a guessed 12 × 20 m rectangle gave 1.6–3.3 m/s for a walking crowd, which suggests the guess was too large. Use measured or well-estimated dimensions.
- **Flat ground is assumed** (a single homography). Stairs or ramps need their own zone.
- Speed uses box bottoms as foot points, so occlusion jitter adds noise; the median over ≥ 3 people damps it.

## 3. Other changes

- **At most 12 gates** (was 24).
- **Setup no longer asks for capacity or radius.** Capacity comes from calibration, or the default 300. The radius was removed from the UI because it cannot affect any decision: routing uses distance ÷ the venue's largest distance, and both scale with the radius. It still sets the metres quoted in reasons and in the simulator's walking times.
