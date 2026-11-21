# Decision-policy evaluation: `engine_v2_vs_v1`

100 seeds per cell; values are mean ± 95% CI half-width. Generated 2026-11-21 18:28 by `scripts/evaluate_decisions.py`.

Policies: **No control** (`no_control`); **Nearest free gate** (`nearest_gate`); **CrowdGuard engine v1** (`engine_v1`); **CrowdGuard engine v2** (`engine_v2`); **Shortest-wait (perfect info)** (`informed`).

## Single-gate surge (`single_surge`)

8 gates at ~67% load; G1 receives +1.9 people/s for 15 min (e.g. a train or bus convoy unloading at the nearest gate).

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 50562 ± 344 | 1999 ± 17 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 3.98 ± 0.04 | 0.0 ± 0.0 | 188 ± 1 | 307 ± 9 | 4.5 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.90 ± 0.03 | 0.0 ± 0.0 | 174 ± 1 | 183 ± 5 | 3.9 ± 0.0 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 3.30 ± 0.03 | 0.0 ± 0.0 | 143 ± 1 | 21 ± 1 | 4.5 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 3.46 ± 0.05 | 0.0 ± 0.0 | 132 ± 1 | 57 ± 2 | 4.1 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -93%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -93%

## Stadium entry peak (`stadium_entry`)

8 gates of 2.2 people/s; 24,000 fans arrive over 45 min with a late peak; transit-side gates G1, G2, G8 get 56% of arrivals.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 26428 ± 554 | 881 ± 15 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 4.53 ± 0.07 | 116 ± 93 | 217 ± 3 | 402 ± 12 | 6.8 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.61 ± 0.02 | 0.0 ± 0.0 | 173 ± 1 | 268 ± 8 | 3.5 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 2.96 ± 0.03 | 0.0 ± 0.0 | 152 ± 2 | 32 ± 1 | 5.5 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 4.32 ± 0.04 | 0.0 ± 0.0 | 193 ± 2 | 53 ± 2 | 5.7 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -75%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -80%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -83%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -78%

## Twin surge around one free gate (`twin_surge`)

G1 and G3 surge together; G2 between them is the obvious free gate for both; the far side of the venue is already busy.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 96455 ± 457 | 1956 ± 12 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 5.38 ± 0.13 | 876 ± 182 | 213 ± 2 | 608 ± 14 | 7.7 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.95 ± 0.03 | 0.0 ± 0.0 | 190 ± 1 | 383 ± 8 | 6.1 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 3.31 ± 0.02 | 0.0 ± 0.0 | 160 ± 1 | 46 ± 1 | 7.3 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 3.97 ± 0.03 | 0.0 ± 0.0 | 169 ± 1 | 86 ± 3 | 6.5 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -99%, 95th-percentile time to entry -89%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -90%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -92%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%

## Venue near saturation (`near_saturation`)

Gates loaded at 83-128% of throughput for 30 min; total demand exceeds total capacity by ~8%.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 79205 ± 1603 | 1203 ± 19 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 5.90 ± 0.14 | 2940 ± 432 | 316 ± 5 | 1762 ± 48 | 14 ± 0 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 4.15 ± 0.11 | 61 ± 46 | 222 ± 4 | 1458 ± 27 | 6.9 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 3.77 ± 0.07 | 0.0 ± 0.0 | 268 ± 4 | 407 ± 23 | 12 ± 0 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 5.47 ± 0.05 | 2213 ± 235 | 300 ± 3 | 197 ± 5 | 9.1 ± 0.2 | 0.4 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -96%, 95th-percentile time to entry -74%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -82%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -78%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -97%, 95th-percentile time to entry -75%

## Festival waves (ghats) (`festival_waves`)

6 entry points of 2.0 people/s around a 200 m site at 60% load; every 10 min a 6-minute wave of +2.6 people/s hits the next ghat.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 235048 ± 6469 | 1241 ± 16 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 4.26 ± 0.02 | 0.0 ± 0.0 | 241 ± 1 | 130 ± 3 | 6.4 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 4.26 ± 0.02 | 0.0 ± 0.0 | 236 ± 1 | 130 ± 3 | 6.4 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 3.49 ± 0.03 | 0.0 ± 0.0 | 216 ± 3 | 31 ± 1 | 8.3 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 4.41 ± 0.02 | 0.0 ± 0.0 | 234 ± 1 | 76 ± 2 | 5.4 ± 0.0 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -81%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -81%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -83%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -81%

## Surge with degraded sensing (`degraded_sensing`)

As single_surge, but G5 and G6 have no camera and G2's camera freezes from 10 to 15 min, while G2 is the natural target.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 50208 ± 342 | 1978 ± 17 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 3.97 ± 0.04 | 0.0 ± 0.0 | 187 ± 1 | 303 ± 8 | 4.4 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.91 ± 0.03 | 0.0 ± 0.0 | 174 ± 1 | 185 ± 5 | 3.9 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 3.33 ± 0.03 | 0.0 ± 0.0 | 144 ± 1 | 21 ± 1 | 4.4 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 3.44 ± 0.04 | 0.0 ± 0.0 | 131 ± 1 | 55 ± 2 | 4.1 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -93%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -93%

## Surge with uneven gate spacing (`uneven_spacing`)

6 gates; G1 surges; the nearly empty G2 is 450 m away, the busier G6 only 70 m. Segments G1->G2..G6->G1: 450, 120, 120, 120, 120, 70 m.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 50517 ± 275 | 2068 ± 14 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 3.96 ± 0.03 | 0.0 ± 0.0 | 196 ± 1 | 289 ± 8 | 6.1 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 3.93 ± 0.03 | 0.0 ± 0.0 | 189 ± 1 | 181 ± 5 | 5.5 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 3.28 ± 0.03 | 0.0 ± 0.0 | 157 ± 2 | 22 ± 1 | 6.4 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 4.18 ± 0.04 | 0.0 ± 0.0 | 179 ± 2 | 25 ± 1 | 5.7 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -92%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -91%

## Frozen camera on a filling target (`frozen_target`)

G1 surges; G2, the obvious target, has its own surge from min 10 while its camera is frozen from min 9 to 19.

| Policy | Peak density (people/m²) | Exposure at ≥5 people/m² (person-minutes) | 95th-percentile time to entry (s) | Instruction changes (per hour) | Arrivals diverted (%) | Arrivals held back (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No control | 9.00 ± 0.00 | 50557 ± 336 | 2012 ± 16 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Nearest free gate | 8.93 ± 0.05 | 12972 ± 538 | 377 ± 18 | 238 ± 7 | 5.5 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v1 | 7.00 ± 0.32 | 5557 ± 999 | 242 ± 17 | 204 ± 5 | 4.8 ± 0.1 | 0.0 ± 0.0 |
| CrowdGuard engine v2 | 3.30 ± 0.03 | 0.0 ± 0.0 | 150 ± 1 | 23 ± 1 | 4.7 ± 0.1 | 0.0 ± 0.0 |
| Shortest-wait (perfect info) | 3.95 ± 0.04 | 0.0 ± 0.0 | 152 ± 1 | 61 ± 3 | 4.8 ± 0.1 | 0.0 ± 0.0 |

- Nearest free gate vs no control (paired): exposure at ≥5 people/m² -74%, 95th-percentile time to entry -81%
- CrowdGuard engine v1 vs no control (paired): exposure at ≥5 people/m² -89%, 95th-percentile time to entry -88%
- CrowdGuard engine v2 vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -93%
- Shortest-wait (perfect info) vs no control (paired): exposure at ≥5 people/m² -100%, 95th-percentile time to entry -92%

## Robustness sweeps

Exposure at ≥5 people/m² (person-minutes) as one setting varies, others at default.

### Share of people who follow instructions

| Scenario | Policy | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Single-gate surge | No control | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 |
| Single-gate surge | Nearest free gate | 37388 ± 495 | 8738 ± 1965 | 0.0 ± 0.0 | 0.0 ± 0.0 | 7.1 ± 14.2 |
| Single-gate surge | CrowdGuard engine v1 | 37388 ± 495 | 8702 ± 1954 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Single-gate surge | CrowdGuard engine v2 | 33917 ± 684 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Single-gate surge | Shortest-wait (perfect info) | 33627 ± 638 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | No control | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 |
| Twin surge around one free gate | Nearest free gate | 70485 ± 707 | 8564 ± 2054 | 879 ± 220 | 1465 ± 322 | 1983 ± 342 |
| Twin surge around one free gate | CrowdGuard engine v1 | 70481 ± 718 | 7953 ± 1975 | 0.0 ± 0.0 | 9.1 ± 18.2 | 58 ± 53 |
| Twin surge around one free gate | CrowdGuard engine v2 | 62761 ± 981 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | Shortest-wait (perfect info) | 61694 ± 956 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | No control | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 |
| Stadium entry peak | Nearest free gate | 14001 ± 619 | 28 ± 39 | 148 ± 158 | 55 ± 66 | 549 ± 219 |
| Stadium entry peak | CrowdGuard engine v1 | 14100 ± 619 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | CrowdGuard engine v2 | 5510 ± 1032 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | Shortest-wait (perfect info) | 11119 ± 754 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | No control | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 |
| Surge with uneven gate spacing | Nearest free gate | 37087 ± 347 | 8690 ± 1447 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | CrowdGuard engine v1 | 37087 ± 347 | 8715 ± 1449 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | CrowdGuard engine v2 | 33821 ± 438 | 6.4 ± 10.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | Shortest-wait (perfect info) | 32210 ± 453 | 107 ± 100 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### Delay before an instruction reaches the crowd (s)

| Scenario | Policy | 0.0 | 30.0 | 60.0 | 120.0 | 180.0 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Single-gate surge | No control | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 |
| Single-gate surge | Nearest free gate | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1159 ± 248 | 12449 ± 890 |
| Single-gate surge | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 163 ± 143 | 8544 ± 835 |
| Single-gate surge | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 10 ± 19 |
| Single-gate surge | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 41 ± 57 |
| Twin surge around one free gate | No control | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 |
| Twin surge around one free gate | Nearest free gate | 0.0 ± 0.0 | 42 ± 42 | 879 ± 220 | 11650 ± 418 | 32879 ± 1675 |
| Twin surge around one free gate | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1650 ± 533 | 19142 ± 2262 |
| Twin surge around one free gate | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 51 ± 75 | 126 ± 61 |
| Twin surge around one free gate | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 59 ± 82 | 465 ± 213 |
| Stadium entry peak | No control | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 |
| Stadium entry peak | Nearest free gate | 0.0 ± 0.0 | 0.0 ± 0.0 | 148 ± 158 | 9680 ± 715 | 10276 ± 425 |
| Stadium entry peak | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 42 ± 67 | 1183 ± 447 |
| Stadium entry peak | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 672 ± 380 |
| Stadium entry peak | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 18 ± 23 | 521 ± 228 |
| Surge with uneven gate spacing | No control | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 |
| Surge with uneven gate spacing | Nearest free gate | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 562 ± 176 | 11142 ± 869 |
| Surge with uneven gate spacing | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 203 ± 149 | 8492 ± 790 |
| Surge with uneven gate spacing | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 42 ± 56 |
| Surge with uneven gate spacing | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 395 ± 263 | 4298 ± 460 |

### Crowd-count error (relative s.d.)

| Scenario | Policy | 0.0 | 0.1 | 0.2 | 0.3 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Single-gate surge | No control | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 | 50561 ± 495 |
| Single-gate surge | Nearest free gate | 2.3 ± 4.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Single-gate surge | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Single-gate surge | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Single-gate surge | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | No control | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 | 96611 ± 689 |
| Twin surge around one free gate | Nearest free gate | 964 ± 297 | 879 ± 220 | 922 ± 212 | 810 ± 237 |
| Twin surge around one free gate | CrowdGuard engine v1 | 91 ± 54 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Twin surge around one free gate | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | No control | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 | 26643 ± 758 |
| Stadium entry peak | Nearest free gate | 603 ± 313 | 148 ± 158 | 21 ± 36 | 0.0 ± 0.0 |
| Stadium entry peak | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Stadium entry peak | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | No control | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 | 50180 ± 336 |
| Surge with uneven gate spacing | Nearest free gate | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | CrowdGuard engine v1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | CrowdGuard engine v2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Surge with uneven gate spacing | Shortest-wait (perfect info) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

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

## Ablation: engine v2 with one improvement switched off

Totals over the scenarios in each setting (mean per run, summed over scenarios); danger in person-minutes at >= 5 people/m^2. Higher than 'all on' means the improvement helps.

### default (8 scenarios)

| Variant | Danger (person-min) | 95th-pct wait (s, mean over scenarios) | Instruction changes/h (mean) |
| :--- | ---: | ---: | ---: |
| **v2 (all on)** | 0 | 174 | 75 |
| without steady diversion | 0 | 183 | 792 |
| without venue-wide coordination | 1,383 | 177 | 42 |
| without ignore stale targets | 5,958 | 185 | 74 |
| without forecast | 63 | 193 | 36 |

### delay 180 s (4 scenarios)

| Variant | Danger (person-min) | 95th-pct wait (s, mean over scenarios) | Instruction changes/h (mean) |
| :--- | ---: | ---: | ---: |
| **v2 (all on)** | 851 | 208 | 26 |
| without steady diversion | 9,425 | 262 | 457 |
| without venue-wide coordination | 7,373 | 227 | 25 |
| without ignore stale targets | 851 | 208 | 26 |
| without forecast | 16,238 | 289 | 19 |

### compliance 40% (4 scenarios)

| Variant | Danger (person-min) | 95th-pct wait (s, mean over scenarios) | Instruction changes/h (mean) |
| :--- | ---: | ---: | ---: |
| **v2 (all on)** | 6 | 168 | 24 |
| without steady diversion | 8,362 | 239 | 308 |
| without venue-wide coordination | 6 | 168 | 24 |
| without ignore stale targets | 6 | 168 | 24 |
| without forecast | 7,375 | 232 | 11 |
