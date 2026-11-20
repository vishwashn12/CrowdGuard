# Decision-policy evaluation: `engine_v1_baseline`

100 seeds per cell; values are mean ± 95% CI half-width. Generated 2026-11-21 03:30 by `scripts/evaluate_decisions.py`.

Policies: **No control** (`no_control`); **Nearest free gate** (`nearest_gate`); **CrowdGuard engine v1** (`engine_v1`); **Shortest-wait (perfect info)** (`informed`).

## Single-gate surge (`single_surge`)

8 gates at ~67% load; G1 receives +1.9 people/s for 15 min (e.g. a train or bus convoy unloading at the nearest gate).

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 50460 ± 298 | 1991 ± 14 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 4.00 ± 0.04 | 0.0 ± 0.0 | 189 ± 1 | 297 ± 9 | 4.4 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.92 ± 0.03 | 0.0 ± 0.0 | 174 ± 1 | 177 ± 5 | 3.9 ± 0.0 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 3.46 ± 0.04 | 0.0 ± 0.0 | 132 ± 1 | 54 ± 2 | 4.1 ± 0.0 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -93%

## Stadium entry peak (`stadium_entry`)

8 gates of 2.2 people/s; 24,000 fans arrive over 45 min with a late peak; transit-side gates G1, G2, G8 get 56% of arrivals.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 26489 ± 560 | 882 ± 17 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 4.53 ± 0.06 | 66 ± 61 | 217 ± 2 | 401 ± 12 | 6.8 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.64 ± 0.02 | 0.0 ± 0.0 | 173 ± 1 | 254 ± 8 | 3.4 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 4.28 ± 0.03 | 0.0 ± 0.0 | 192 ± 1 | 53 ± 2 | 5.6 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -75%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -80%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -78%

## Twin surge around one free gate (`twin_surge`)

G1 and G3 surge together; G2 between them is the obvious free gate for both; the far side of the venue is already busy.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 95663 ± 465 | 1938 ± 11 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 5.22 ± 0.11 | 669 ± 148 | 210 ± 2 | 602 ± 12 | 7.5 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.93 ± 0.03 | 0.0 ± 0.0 | 190 ± 1 | 386 ± 8 | 6.0 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 4.01 ± 0.03 | 0.0 ± 0.0 | 168 ± 1 | 88 ± 4 | 6.4 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -99%, 95th-percentile time to entry -89%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -90%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%

## Venue near saturation (`near_saturation`)

Gates loaded at 83-128% of throughput for 30 min; total demand exceeds total capacity by ~8%.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 77624 ± 1767 | 1185 ± 21 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 5.89 ± 0.14 | 2903 ± 442 | 313 ± 5 | 1737 ± 43 | 14 ± 0 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 4.18 ± 0.11 | 74 ± 52 | 219 ± 5 | 1426 ± 25 | 6.7 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 5.44 ± 0.05 | 2177 ± 261 | 299 ± 3 | 194 ± 5 | 9.1 ± 0.2 | 0.4 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -96%, 95th-percentile time to entry -74%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -81%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -97%, 95th-percentile time to entry -75%

## Festival waves (ghats) (`festival_waves`)

6 entry points of 2.0 people/s around a 200 m site at 60% load; every 10 min a 6-minute wave of +2.6 people/s hits the next ghat.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 234788 ± 6444 | 1236 ± 16 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 4.26 ± 0.02 | 0.0 ± 0.0 | 241 ± 1 | 129 ± 3 | 6.4 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 4.26 ± 0.02 | 0.0 ± 0.0 | 237 ± 1 | 129 ± 3 | 6.4 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 4.42 ± 0.02 | 0.0 ± 0.0 | 234 ± 1 | 75 ± 2 | 5.4 ± 0.0 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -80%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -81%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -81%

## Surge with degraded sensing (`degraded_sensing`)

As single_surge, but G5 and G6 have no camera and G2's camera freezes from 10 to 15 min, while G2 is the natural target.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 50414 ± 355 | 1993 ± 16 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 3.98 ± 0.04 | 0.0 ± 0.0 | 188 ± 1 | 295 ± 9 | 4.4 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.91 ± 0.03 | 0.0 ± 0.0 | 175 ± 1 | 184 ± 5 | 3.9 ± 0.0 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 3.46 ± 0.04 | 0.0 ± 0.0 | 132 ± 1 | 58 ± 2 | 4.1 ± 0.0 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -93%

## Robustness sweeps

Exposure at ≥5 people/m² (person-minutes) as one setting varies, others at default.

### Share of people who follow instructions

| Scenario | Policy | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Single-gate surge | No control | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 |
| Single-gate surge | Nearest free gate | 37363 ± 458 | 9320 ± 1905 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.0 ± 2.0 |
| Single-gate surge | CrowdGuard engine v1 | 37363 ± 458 | 9317 ± 1882 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Single-gate surge | Shortest-wait (perfect info) | 33555 ± 576 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | No control | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 |
| Twin surge around one free gate | Nearest free gate | 69763 ± 735 | 6668 ± 1510 | 600 ± 215 | 1397 ± 336 | 2040 ± 379 |
| Twin surge around one free gate | CrowdGuard engine v1 | 69766 ± 736 | 6275 ± 1619 | 0.0 ± 0.0 | 12 ± 20 | 47 ± 44 |
| Twin surge around one free gate | Shortest-wait (perfect info) | 61122 ± 822 | 20 ± 40 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | No control | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 |
| Stadium entry peak | Nearest free gate | 13928 ± 748 | 35 ± 54 | 82 ± 99 | 66 ± 53 | 582 ± 211 |
| Stadium entry peak | CrowdGuard engine v1 | 13993 ± 759 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | Shortest-wait (perfect info) | 11305 ± 689 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### Delay before an instruction reaches the crowd (s)

| Scenario | Policy | 0.0 | 30.0 | 60.0 | 120.0 | 180.0 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Single-gate surge | No control | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 |
| Single-gate surge | Nearest free gate | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1174 ± 291 | 12598 ± 855 |
| Single-gate surge | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 287 ± 217 | 8609 ± 773 |
| Single-gate surge | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.3 ± 2.7 |
| Twin surge around one free gate | No control | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 |
| Twin surge around one free gate | Nearest free gate | 0.0 ± 0.0 | 62 ± 51 | 600 ± 215 | 11419 ± 417 | 31705 ± 1516 |
| Twin surge around one free gate | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1540 ± 506 | 17318 ± 2112 |
| Twin surge around one free gate | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 25 ± 32 | 299 ± 133 |
| Stadium entry peak | No control | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 |
| Stadium entry peak | Nearest free gate | 0.0 ± 0.0 | 0.0 ± 0.0 | 82 ± 99 | 10007 ± 723 | 10216 ± 471 |
| Stadium entry peak | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 34 ± 68 | 1107 ± 381 |
| Stadium entry peak | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 36 ± 38 | 231 ± 139 |

### Crowd-count error (relative s.d.)

| Scenario | Policy | 0.0 | 0.1 | 0.2 | 0.3 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Single-gate surge | No control | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 | 50538 ± 455 |
| Single-gate surge | Nearest free gate | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 30 ± 61 |
| Single-gate surge | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 35 ± 69 |
| Single-gate surge | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | No control | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 | 95872 ± 688 |
| Twin surge around one free gate | Nearest free gate | 823 ± 317 | 600 ± 215 | 617 ± 156 | 416 ± 180 |
| Twin surge around one free gate | CrowdGuard engine v1 | 51 ± 45 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | No control | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 | 26609 ± 765 |
| Stadium entry peak | Nearest free gate | 333 ± 161 | 82 ± 99 | 0.0 ± 0.0 | 0.4 ± 0.8 |
| Stadium entry peak | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

## Settings

```json
{
  "dt": 1.0,
  "decision_period": 1.0,
  "obs_period": 2.0,
  "obs_latency_s": 1.0,
  "count_noise": 0.1,
  "speed_noise": 0.1,
  "compliance": 0.6,
  "restrict_strength": 0.5,
  "actuation_delay_s": 60.0,
  "seed": 0
}
```
