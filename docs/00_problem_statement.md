# 00: Problem statement, motivation and use cases

## One-line problem statement

At large venues with several entry or exit gates, crowds tend to pile up at one or two gates while others stay under-used. Operators usually see this too late, and too little is visible to act on it. **CrowdGuard watches every gate's CCTV at once, detects congestion and dangerous individual behaviour as it forms, and tells operators where to redirect incoming crowd, and why, before density reaches crush levels.**

## Motivation

### Crowd disasters are mostly flow-management failures

Many fatal crowd crushes have no single trigger such as a fire or an attack. They happen because too many people converge on too little space, often while space nearby stays free. Recurring patterns:

- **Uneven distribution across entry points.** People head for the nearest or most familiar gate, the one their ticket names, or the one others are heading to.
- **Bottlenecks:** a single ramp, footbridge, stairway, gate or token counter.
- **Late detection.** By the time density is visible to the naked eye from ground level, people inside the crowd can no longer move.
- **Information gaps.** Control rooms watch dozens of camera feeds; gate staff see only their own gate.

### Incidents that illustrate the problem

> **Verify every figure and date against official inquiry reports or reputable sources before citing.** The values below are approximate, from general knowledge.

| Incident | Year | Setting | Approx. deaths | Relevant pattern |
| :--- | :--- | :--- | ---: | :--- |
| Hillsborough, Sheffield (UK) | 1989 | football stadium | 97 | central pens overfilled while side pens had space: **a distribution failure** |
| Love Parade, Duisburg (Germany) | 2010 | music festival | 21 | single entry/exit ramp; inflow and outflow met at a bottleneck |
| Mina, Hajj (Saudi Arabia) | 2015 | religious gathering | 769 official (higher by other counts) | two crowd streams converging; counter-flow |
| Itaewon, Seoul (South Korea) | 2022 | street festival | 159 | extreme density in a narrow alley; no crowd monitoring |
| Kanjuruhan Stadium (Indonesia) | 2022 | football stadium | ~135 | panic toward limited, partly locked exits |
| Allahabad (Prayagraj) railway station, Kumbh Mela (India) | 2013 | pilgrim transit | ~36 | footbridge and platform overcrowding |
| Elphinstone Road station, Mumbai (India) | 2017 | commuter footbridge | 23 | narrow footbridge at peak |
| Hathras satsang, Uttar Pradesh (India) | 2024 | religious gathering | ~121 | crowd surge at exit |
| Maha Kumbh, Prayagraj (India) | 2025 | religious gathering | 30 (official) | surge at bathing ghats |
| New Delhi railway station (India) | 2025 | pilgrim transit | ~18 | platform and footbridge surge |
| M. Chinnaswamy Stadium, Bengaluru (India) | 2025 | stadium celebration | ~11 | crowds massing at stadium gates |

India is especially exposed: religious gatherings (melas, temple festivals, yatras) and public events draw some of the largest crowds in the world, often at sites with fixed, narrow entry points and limited monitoring.

### Crowd science behind the approach

- **Density thresholds.** Above roughly 4–5 people/m², involuntary body contact begins and movement becomes constrained. Around 6–7 people/m² and above, crowds can show **turbulence**: shock waves that push people off their feet. These ranges come from Fruin's levels of service, G. K. Still's crowd-risk work, and Helbing et al.'s analysis of the 2006 Jamarat (Hajj) footage. *TODO: cite exact values from the sources.*
- **Throughput falls when a gate is overloaded.** Pedestrian flow (people per metre per second) rises with density only up to a point, then falls ("faster-is-slower", Helbing et al. 2000; the pedestrian fundamental diagram, e.g. Weidmann 1993). **Pushing more people into an already congested gate lowers its throughput and raises the risk.** Moving extra arrivals to a gate below its peak flow increases total throughput *and* safety. This is the theoretical case for diversion.
- **Counter-flow and fallen people seed collapses.** People moving against the flow create turbulence, and a fallen or seated person in a moving crowd is a trip point that can cascade. This is why CrowdGuard detects these two behaviours per person (see [03](03_anomaly_classifier.md)).

## What CrowdGuard does

1. **Sees every gate at once.** It ingests one CCTV feed or uploaded video per gate and processes them in parallel on a single GPU.
2. **Measures crowd state per gate:** crowd count (CSRNet), crowd speed and flow coherence (pose tracking), and trend.
3. **Decides per gate:** no action, monitor, **divert incoming crowd to gate X along route Y**, or restrict inflow when every gate is full. Every decision comes with an explanation.
4. **Flags dangerous individuals:** a person sitting or collapsed in the flow, or moving against it, as a separate alert channel for staff dispatch.
5. **Alerts and logs:** de-duplicated, prioritised alerts and an incident log.

**Human in the loop.** CrowdGuard is **decision support**, not an autonomous controller. It recommends and explains, and an operator decides. This is deliberate, for accountability (someone must own a decision to redirect thousands of people), for trust (operators reject black boxes), and because the score has not yet been validated against real incidents (see [10](10_limitations_future_work.md)).

## Use cases

| Use case | Typical setting | What CrowdGuard provides |
| :--- | :--- | :--- |
| **Stadium entry before a match** | 4–24 turnstile gates around a ring | detects the gate whose queue is building; recommends diverting arriving fans to under-used gates along the perimeter |
| **Stadium exit / dispersal** | all gates at once after the final whistle | spots gates where outflow has stalled (high density, near-zero speed means crush risk) |
| **Religious gatherings** (Kumbh Mela, temple festivals, Hajj, yatras) | multiple ghats, darshan queues, entry points | balances pilgrims across access points; flags fallen people and counter-flow inside dense crowds |
| **Railway stations during festivals** | platforms, footbridges, exits | early warning when a footbridge or exit approaches critical density |
| **Concerts and festivals** | entry gates, stage barriers | gate balancing; anomaly alerts near barriers |
| **Rallies and public celebrations** | open venues with improvised gates | quick setup: choose the number of gates, upload or connect feeds, start |

### Primary users

- **Control-room operator.** Watches all gates, accepts or overrides recommendations, dispatches staff.
- **Gate or zone staff and police.** Receive instructions ("redirect arrivals at North-East to East, via the outer concourse").
- **Event safety planner.** Reviews incident logs after the event and calibrates capacities and thresholds.

## Scope and non-goals

- **In scope:** density, flow and individual-behaviour monitoring; gate-level diversion recommendations; alerting.
- **Not in scope:**
  - autonomous control of gates or signage;
  - fire or medical evacuation planning;
  - identifying individuals: **no face recognition and no re-identification across cameras**. Tracks are anonymous and exist only within one camera feed.
- **Deployment assumption:** fixed CCTV cameras, one or more per gate, with a view of the approach area.

## Research questions (for the paper)

1. **RQ1: Perception.** Can a single pose-estimation pass per frame provide both crowd-level motion and per-person anomaly features in real time for several cameras on commodity hardware?
2. **RQ2: Anomaly detection.** How well do trajectory and pose features detect sitting/fallen and counter-flow individuals in dense crowds, and what limits them: detection or classification?
3. **RQ3: Decision quality.** Does gate-level diversion based on measured crowd state reduce peak density and waiting time compared with no control and with naive policies, and how robust is it to sensor noise, missing cameras and partial compliance? (Evaluation plan: closed-loop simulation; see the congestion-control review.)
4. **RQ4: Usability.** Can operators understand and act on the recommendations quickly?

## Success criteria

| Criterion | Target |
| :--- | :--- |
| Real-time operation | ≥ 3 gates at the source frame rate on one consumer GPU (achieved: 3 gates on an RTX 3050 6 GB) |
| Anomaly detection | report per-class precision/recall plus incident-level recall and false alarms per minute |
| Congestion control (simulation) | lower peak density and 95th-percentile wait than no-control and nearest-gate baselines, across a range of compliance levels, with confidence intervals |
| Stability | diversion targets do not oscillate (switches per hour, measured) |
| Explainability | every recommendation carries a human-readable reason |
