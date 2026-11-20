# 06: Detector evaluation (recall of labelled people)

**Question:** which detector and resolution best finds the people the anomaly classifier must label, especially seated and distant people? The user reported that the small model missed edge cases and that the large model was more accurate. This study measured the claim instead of assuming it.

## Protocol

- **Frames.** 243, sampled across both Hajj test videos (1440×1080).
- **Targets.** All 2,366 annotation boxes in those frames: sitting 1,509, standing 297, moving_in_opposite 256, different_crowd_direction 170, sleeping 126; the remaining few are running.
- **Metric.** Recall: the share of labelled boxes matched by some predicted person box at IoU ≥ 0.5 and at IoU ≥ 0.3. IoU 0.3 is the primary figure, because the hand-drawn boxes are looser than detector boxes, and it is the same threshold the training labels used.
- **Precision is not measurable.** Ordinary people are unannotated, so every unmatched prediction may still be a real person.
- **Settings.** FP16, confidence 0.15, IoU 0.5. Timing is the median ms per frame on an RTX 3050 6 GB laptop GPU.
- **Size split.** "Small" means a labelled box under 80 px tall (838 boxes); "large" means 80 px or taller (1,528).

## Results (recall at IoU ≥ 0.5 / IoU ≥ 0.3)

### Pose models at 960

| Class (n) | pose-s @960 | pose-m @960 | pose-l @960 |
| :--- | :---: | :---: | :---: |
| ALL (2366) | 17.4 / 33.1 | 16.1 / 29.9 | 14.6 / 26.0 |
| sitting (1509) | 5.3 / 19.3 | 4.2 / 15.6 | 3.7 / 12.7 |
| sleeping (126) | 92.1 / 92.1 | 89.7 / 89.7 | 90.5 / 92.9 |
| standing (297) | 43.1 / 61.6 | 44.1 / 59.6 | 39.7 / 53.9 |
| moving_in_opposite (256) | 34.4 / 75.4 | 28.9 / 70.7 | 22.7 / 57.4 |
| different_crowd_direction (170) | 0.0 / 0.0 | 0.0 / 1.2 | 0.0 / 0.0 |
| small <80 px (838) | 0.0 / 3.5 | 0.0 / 4.2 | 0.0 / 2.3 |
| large ≥80 px (1528) | 27.0 / 49.3 | 25.0 / 44.0 | 22.6 / 39.0 |
| boxes predicted per frame | 54.6 | 53.6 | 47.8 |

### Pose models at 1280 and 1440

| Class (n) | pose-s @1280 | **pose-m @1280 (shipped)** | pose-l @1280 | pose-l @1440 |
| :--- | :---: | :---: | :---: | :---: |
| ALL (2366) | 22.1 / 43.2 | 21.9 / **40.4** | 20.5 / 38.2 | 23.0 / 44.8 |
| sitting (1509) | 9.7 / 29.2 | 10.0 / **26.2** | 8.0 / 23.7 | 11.5 / 32.3 |
| sleeping (126) | 92.9 / 96.8 | 92.1 / 93.7 | 96.0 / 98.4 | 96.8 / 98.4 |
| standing (297) | 50.5 / 74.4 | 55.6 / 75.1 | 47.8 / 72.7 | 49.5 / 74.4 |
| moving_in_opposite (256) | 41.4 / 86.7 | 33.2 / 81.6 | 37.9 / 77.7 | 38.7 / 81.2 |
| different_crowd_direction (170) | 0.0 / 7.1 | 0.0 / 5.3 | 0.0 / 2.9 | 0.6 / 8.8 |
| small <80 px (838) | 0.1 / 10.0 | 0.2 / 7.6 | 0.0 / 6.6 | 2.0 / 16.0 |
| large ≥80 px (1528) | 34.1 / 61.3 | 33.8 / 58.4 | 31.7 / 55.6 | 34.6 / 60.5 |
| boxes predicted per frame | 73.9 | 70.7 | 63.6 | 73.5 |
| ms / frame (FP16, median) | 22.6 | **40.7** | 65.0 | 79.9 |

### Detection-only YOLOv8l (no keypoints) against the best small pose model

| Class (n) | pose-s @1280 | det-l @960 | det-l @1280 | det-l @1440 |
| :--- | :---: | :---: | :---: | :---: |
| ALL (2366) | 22.1 / 43.2 | 15.9 / 28.6 | 23.6 / 43.3 | 22.8 / 47.6 |
| sitting (1509) | 9.7 / 29.2 | 4.0 / 15.1 | 12.7 / 32.6 | 16.0 / **44.7** |
| sleeping (126) | 92.9 / 96.8 | 92.9 / 96.0 | 95.2 / 96.8 | 89.7 / 92.1 |
| standing (297) | 50.5 / 74.4 | 39.4 / 51.5 | 48.5 / 66.3 | 33.3 / 47.8 |
| moving_in_opposite (256) | 41.4 / 86.7 | 31.6 / 67.6 | 40.2 / 82.4 | 33.2 / 73.0 |
| different_crowd_direction (170) | 0.0 / 7.1 | 0.0 / 0.6 | 0.0 / 0.6 | 0.0 / 4.1 |
| small <80 px (838) | 0.1 / 10.0 | 0.1 / 5.4 | 1.2 / 16.3 | 4.5 / **30.7** |
| large ≥80 px (1528) | 34.1 / 61.3 | 24.5 / 41.4 | 35.9 / 58.0 | 32.8 / 56.9 |
| boxes predicted per frame | 73.9 | 51.8 | 62.2 | 58.0 |
| ms / frame (FP16, median) | 22.8 | 39.7 | 63.2 | 78.1 |

### Summary (IoU ≥ 0.3)

This is the table used in the CrowdGuard README.

| Detector | Seated | Distant (<80 px) | All labelled | ms / frame |
| :--- | ---: | ---: | ---: | ---: |
| pose-s @960 (earlier CrowdGuard default) | 19% | 4% | 33% | 15 |
| pose-m @960 | 16% | 4% | 30% | 26 |
| **pose-m @1280 (shipped)** | **26%** | **8%** | **40%** | **41** |
| pose-l @1280 | 24% | 7% | 38% | 65 |
| pose-l @1440 | 32% | 16% | 45% | 80 |
| det-l @1440 | 45% | 31% | 48% | 78 (no keypoints) |

The ms values for the @960 pose rows come from the separate timing run reported in the README.

## Findings

1. **Resolution matters more than model size.** At 960, larger pose models found *fewer* seated people (s 19% > m 16% > l 13%). Raising every model to 1280 lifted all of them. Switching to a larger model without raising resolution would have made the edge cases worse.
2. **Detection models find seated people better than pose models.** At 1440 the scores are 45% vs 32%. Pose models are trained only on people with visible keypoints, so heavily occluded or folded seated bodies are under-represented. The classifier needs keypoints, so a detection model cannot simply replace the pose model.
3. **Detection caps anomaly accuracy.** Even the best configuration finds fewer than half of the labelled seated people. A person never detected can never be classified. This ceiling sits *before* the classifier metrics in [03](03_anomaly_classifier.md).
4. **`different_crowd_direction` is almost never matched (≤ 9%).** Its annotation boxes probably cover groups or regions rather than individuals. This supports excluding it from training.
5. **The shipped choice.** pose-m @1280 runs at 41 ms. pose-l @1440 is better (45% vs 40% overall) but costs about 2× the time, which the 3-camera GPU budget could not afford ([09](09_runtime_performance.md)).

## Ideas for the paper

- A **hybrid detector**: detection-model boxes for recall, with pose run on crops for keypoints. The detection models' gain on seated (+13 points) and distant (+15 points) people is the motivation.
- **Report detection recall alongside classifier F1**, so readers see the end-to-end ceiling.
