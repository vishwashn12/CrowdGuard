# 07: Congestion scoring, diversion decisions and alerts

Nothing in this layer is learned. Every weight and threshold is a **hand-set heuristic**, inherited from `crowd_management` unless marked otherwise. None has been fitted or validated against ground-truth crowd incidents. State this plainly in the paper and present the layer as a transparent, explainable rule system.

Sources:
- `crowdguard/decision/{config,fusion,routing,trend,engine}.py`
- `crowdguard/alerts/rules.py`
- `crowdguard/venue/*`

## Venue model

- **Gates on a ring.** N gates (user-chosen, 2 to 24) sit on a circle of radius R (default 120 m). Neighbouring gates are connected with edge length equal to the arc $2\pi R / N$. All-pairs shortest paths are computed with Floyd–Warshall.
- **Capacity per gate.** Default 500. The demo layout `circular_venue.yaml` uses 500/600/550/500/450/650. **These are placeholders** and must be set from observed occupancy.
- **Monitoring is optional.** A gate may have a camera bound or none. Unmonitored gates stay in the graph as diversion *targets* only.

## Per-gate state (fusion)

Inputs: the CSRNet `crowd_count` and the motion quantities from [05](05_pretrained_models.md).

$$\text{density\_score} = \operatorname{clamp}_{[0,1]}\!\left(\frac{\text{crowd\_count}}{\text{capacity}}\right), \qquad \text{capacity\_utilization} = \text{density\_score}$$

### Congestion score

$$S = \frac{w_d\,\text{density} + w_v\left(1 - \operatorname{clamp}\!\left(\frac{v}{v_{\max}}\right)\right) + w_c\,\text{utilization}}{w_d + w_v + w_c}$$

| Parameter | Value |
| :--- | :--- |
| $w_d$ (density) | 0.5 |
| $w_v$ (inverse velocity) | 0.2 |
| $w_c$ (capacity) | 0.3 |
| $v_{\max}$ (`max_expected_velocity`) | **75 px/s**. CrowdGuard recalibrated it; the prototype used 15. |

**When motion is missing or stale**, the velocity term is dropped and the weights renormalised to $(w_d\,\text{density} + w_c\,\text{util}) / (w_d + w_c)$. Substituting zero velocity would invent congestion; substituting maximum velocity would invent free flow.

> **Note for the paper:** in the current implementation `capacity_utilization` equals `density_score`, so density and capacity together weight count/capacity by 0.8 and the formula reduces to $S = 0.8\,u + 0.2\,(1 - v/v_{\max})$. Either describe it that way, or separate the two terms (for example, density as people per m² once areas are known).

**Recalibrating $v_{\max}$ (measured).** On 1440×1080 footage, a freely flowing crowd moved at **70–78 px/s**. With the prototype's 15 px/s the inverse-velocity term was pinned at 0 for every observation: every gate read LOW, and a stalling crowd would not register until it was already dense. The `VELOCITY_UNCALIBRATED` advisory fires when the term saturates. $v_{\max}$ is a pixel rate and must be calibrated per camera.

### Levels

| Level | Score |
| :--- | :--- |
| LOW | S < 0.40 |
| MODERATE | 0.40 ≤ S < 0.65 |
| HIGH | 0.65 ≤ S < 0.80 |
| CRITICAL | S ≥ 0.80 |

### Confidence

Base confidence comes from the observation. It is multiplied by **0.5** if density is missing and by **0.7** if motion is missing or stale.

### Trend

The engine keeps the last 5 congestion scores per gate and compares the mean of the newer half with the older half. The trend is INCREASING if the difference exceeds +0.05, DECREASING if below −0.05, and otherwise STABLE.

### Unmonitored gates

An unmonitored gate is assigned utilisation **0.3**, deliberately non-zero: assuming an unseen gate is empty would make it the most attractive target. It has no density and no motion, so its confidence is reduced, and it can never be a diversion source.

## Decision engine (runs at 2 Hz while a session is running)

Per gate, rules are evaluated in order. The first match wins, and every result carries reason codes and a text explanation.

| # | Condition | Action |
| :--- | :--- | :--- |
| 1 | observation older than **5 s** | MONITOR (stale data must not move people) |
| 2 | confidence < **0.3** | MONITOR |
| 3 | level below **HIGH** | MONITOR if MODERATE, else NO_ACTION |
| 4 | HIGH or CRITICAL, but no viable target | **RESTRICT_INFLOW** (hold arrivals upstream) |
| 5 | HIGH or CRITICAL with a viable target | **DIVERT** to the best-scoring target, subject to hysteresis |

### Candidate viability

- The target must be reachable.
- Its congestion score must be ≤ **0.75**.
- Its remaining capacity (1 − utilisation) must be ≥ **0.15**.
- Unmonitored targets are allowed by default (`allow_unmonitored_targets`).

### Routing score (higher is better)

$$R = 0.35\,(1 - S_{\text{cand}}) + 0.30\,(1 - u_{\text{cand}}) + 0.20\,(1 - d/d_{\max}) + 0.15\,b_{\text{trend}} \;-\; 0.15\cdot[\text{unmonitored}]$$

The weights sum to 1; the code divides by their sum, so they stay normalised if you change them.

where $d$ is the shortest-path distance, $d_{\max}$ is the largest distance in the venue, and $b_{\text{trend}}$ is 1.0 for DECREASING, 0.5 for STABLE and 0.0 for INCREASING. The result is clamped at 0. Candidates are ranked viable-first, then by $R$.

### Hysteresis

Once a source gate has a target, it keeps that target for at least **10 s** (cooldown). During the cooldown, a challenger replaces it only if it beats the incumbent's routing score by **0.15**. This prevents signage and staff from flip-flopping.

### Action types

The engine can return `NO_ACTION`, `MONITOR`, `DIVERT`, `RESTRICT_INFLOW` and `HOLD`.

## Alert rules

Congestion rules are mutually exclusive per gate, evaluated most-severe first. Anomaly and system rules are evaluated independently, so a gate can raise one congestion alert and one anomaly alert at the same time. Alerts are de-duplicated by `(rule_id, gate_id)`: a lasting condition is one alert with an occurrence count. Escalations are logged as new incidents in `logs/alerts.jsonl`. `ALL_CLEAR` is suppressed while any real alert is active, and `IMBALANCED_LOAD` is suppressed on gates with a hard congestion alert.

### Congestion rules

| Rule | Severity | Condition |
| :--- | :--- | :--- |
| `CRUSH_RISK` | EMERGENCY | level HIGH or CRITICAL **and** velocity < 5 px/s **and** motion available **and** count > 20 |
| `GATE_OVERLOAD` | EMERGENCY | utilisation ≥ 1.0 |
| `ALL_GATES_FULL` | EMERGENCY | the engine chose RESTRICT_INFLOW (congested, no viable target) |
| `BOTTLENECK_FORMING` | ALERT | trend INCREASING **and** S > 0.65 **and** level HIGH or CRITICAL |
| `DIVERT_IN_PROGRESS` | ALERT | a diversion is active (includes staffing instructions) |
| `SLOW_FLOW_WARNING` | WARNING | level MODERATE or HIGH **and** velocity < 10 px/s **and** count > 15 |
| `MONITOR_SURGE` | WARNING | action MONITOR **and** trend INCREASING |
| `IMBALANCED_LOAD` | WARNING | the gate carries ≥ 2.5× the venue average **and** count > 30 (not the lightest gate) |

### Anomaly rules (fed by the classifier and tracker, never by congestion)

| Rule | Severity | Condition |
| :--- | :--- | :--- |
| `PERSON_DOWN` | EMERGENCY if the surrounding crowd is moving (velocity ≥ 6 px/s) or the gate is congested; otherwise WARNING | ≥ 1 confirmed SITTING person |
| `COUNTERFLOW_DETECTED` | WARNING at ≥ 2 people; ALERT at ≥ 4 | confirmed OPPOSITE people |
| `ANOMALY_CLUSTER` | ALERT | ≥ 25% of tracked people anomalous, and at least 3 people |

### System rules

| Rule | Severity | Condition |
| :--- | :--- | :--- |
| `TRACKING_DEGRADED` | WARNING | pose continuity-break ratio ≥ 20% over a rolling 10 s window |
| `MOTION_UNAVAILABLE` | INFO | a monitored gate is scored from density alone |
| `VELOCITY_UNCALIBRATED` | INFO | measured velocity ≥ $v_{\max}$, so the term is saturated (added in CrowdGuard) |
| `ALL_CLEAR` | INFO | nothing else is active |

## Design choice: anomalies are a separate channel

Anomalies deliberately do **not** enter the congestion score or routing. The two call for different responses:
- **Congestion** is answered by **moving crowd flow**: divert, or restrict inflow.
- **An anomaly** is answered by **sending staff to one person**.

Folding a collapsed individual into a gate's congestion score would blur both. The one deliberate coupling is in the opposite direction: congestion *context* escalates `PERSON_DOWN`, because a seated person in a moving or packed crowd is far more dangerous than one in a calm area.

## Changes from `crowd_management`

| Change | Reason |
| :--- | :--- |
| $v_{\max}$ raised from 15 to 75 px/s, plus the `VELOCITY_UNCALIBRATED` rule | measured saturation |
| Unmonitored gates publish an assumed state (utilisation 0.3) rather than 0 | 0 made unseen gates look empty and attractive |
| The venue builder rejects bindings or capacities for unknown gates | they were previously dropped silently |
| `ALL_CLEAR` suppressed when real alerts are active | contradictory UI |
| Anomaly and system rules added | integration of the anomaly branch |
| The gate count is set at runtime (2–24) with generated compass names | user requirement |
