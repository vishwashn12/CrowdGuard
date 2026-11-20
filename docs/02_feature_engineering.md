# 02: Feature engineering

Sources:
- `realtime/extract_features.py`: the training-time extractor.
- `crowdguard/crowdguard/models/posetrack/features.py`: a line-for-line port, so inference inputs match training.
- `crowdguard/crowdguard/models/posetrack/tracker.py`: tracking state and crowd direction.

## Pipeline per frame

1. **Detect and estimate pose.** YOLOv8m-pose gives one box and 17 COCO keypoints (x, y, confidence) per person. Training used imgsz 960 and confidence 0.15; CrowdGuard uses imgsz 1280 in FP16 with confidence 0.15 and IoU 0.5.
2. **Track.** ByteTrack (Ultralytics `bytetrack.yaml`, `persist=True`) gives each person a persistent id.
3. **Keep trajectory state per track.** This includes the centre-point history (≤ 60 positions), per-frame step lengths, a stationary-frame count, and the current box and keypoints.
4. **Estimate crowd direction** from all moving tracks (see below).
5. **Build features** for every track with ≥ 10 frames of history, then classify.

## Constants (training rate: 27 fps, stride 1)

| Constant | Value | Meaning |
| :--- | ---: | :--- |
| `DIRECTION_WINDOW` | 15 frames (0.56 s) | look-back for heading and speed |
| `MIN_SPEED` | 2.0 px/frame | below this a step counts as stationary; a track must move at least this fast to vote on crowd direction |
| `HISTORY_LENGTH` | 60 frames (2.2 s) | trajectory memory |
| `MIN_TRACK_FRAMES` | 10 frames | minimum track age before features are produced |
| `MIN_CROWD_PEOPLE` | 3 | minimum moving tracks needed to define a crowd direction |
| crowd-direction history | 30 frames | exponential smoothing window |
| Torso keypoint confidence gate | > 0.5 | shoulders, hips, nose |
| Leg keypoint confidence gate | > 0.3 | hip, knee, ankle |

## Per-track heading and speed

Let $p_t$ be the box centre at frame $t$ and $W = \min(\text{DIRECTION\_WINDOW}, |\text{history}|)$:

$$\theta = \operatorname{atan2}(p_t^y - p_{t-W}^y,\; p_t^x - p_{t-W}^x), \qquad v = \frac{\lVert p_t - p_{t-W} \rVert}{W}\ \text{px/frame}$$

Measuring over a window rather than between adjacent frames suppresses detector jitter.

## Crowd direction

Moving tracks qualify when $v \ge$ MIN_SPEED and they are at least 3 frames old. If at least `MIN_CROWD_PEOPLE` qualify:

1. Take a speed-weighted circular mean: $\phi_t = \operatorname{atan2}\left(\sum_i v_i \sin\theta_i,\ \sum_i v_i \cos\theta_i\right)$.
2. Smooth it with exponential circular smoothing over the last 30 values, with weights $w_k \propto e^{\,\text{linspace}(-1,0)}$ (newest weighted most):
   $\Phi = \operatorname{atan2}\left(\sum_k w_k \sin\phi_k,\ \sum_k w_k \cos\phi_k\right)$.

Otherwise the crowd direction is marked invalid.

## The 10 features (order fixed: index = column in the model)

| # | Name | Group | Formula | Intuition |
| ---: | :--- | :--- | :--- | :--- |
| 0 | `displacement` | motion | $\lVert p_{\text{last}} - p_{\text{first}} \rVert$ over the stored history (px) | stationary vs moving |
| 1 | `direction_sin` | direction | $\sin\theta$ | absolute heading |
| 2 | `direction_cos` | direction | $\cos\theta$ | absolute heading |
| 3 | `cos_sim` | direction | $\cos(\theta - \Phi)$ if the crowd direction is valid, else **0** | +1 with the flow, −1 against it. 0 is used when no flow exists, so "unknown" is never read as "aligned". |
| 4 | `straightness` | motion | displacement / Σ step lengths (0 if path ≈ 0) | purposeful walking ≈ 1; jitter or loitering is low |
| 5 | `aspect_ratio` | appearance | box width / box height | seated people are wider relative to height |
| 6 | `pos_y_norm` | appearance | box-centre y / frame height | pseudo-depth: lower in frame means nearer, larger and faster in pixels |
| 7 | `torso_height_ratio` | pose | max(0, hip_y − shoulder_y) / box height | upper-body proportion |
| 8 | `head_to_hip_ratio` | pose | max(0, hip_y − nose_y) / box height | the head is closer to the hips when seated |
| 9 | `knee_bend_angle` | pose | mean hip–knee–ankle angle (°) over the visible legs | ≈180° standing, toward 90° seated |

Keypoint handling: for shoulders and hips, the mean of left and right is used if both pass the confidence gate, else the visible side, else the feature is unavailable. Fallbacks are `(torso, head_to_hip, knee) = (0.0, 0.0, 180.0)`, i.e. "not sitting" when the body is not visible. A measured knee angle (≠ 180°) is present in **75.0% of training rows and 83.5% of test rows**; `torso_height_ratio > 0` holds in 92.7% and 95.3%. This confirms that the pose model, not plain detection, produced the datasets.

> Note on `extract_features.py`: its `--model` default is `yolov8m.pt` (detection only), but the docstring usage and the knee-angle coverage show the dataset was extracted with `--model yolov8m-pose.pt`. Report the pose model.

## Version 1 feature set (for the comparison in [03](03_anomaly_classifier.md))

The V1 baseline (commit `07baf6a`, 27 Aug 2026) used **13 trajectory/appearance features and no pose**:
`speed_avg, speed_max, speed_var, displacement, direction_sin, direction_cos, cos_sim, straightness, aspect_ratio, area_norm, frac_stationary, pos_y_norm, density_local`.
V2 dropped the speed statistics, `area_norm`, `frac_stationary` and `density_local`, and added the three pose features.

## Post-classification logic (in `tracker.py`, not learned)

| Mechanism | Rule | Purpose |
| :--- | :--- | :--- |
| Confirmation | an anomaly label must repeat for **5 consecutive** predictions; a return to normal applies immediately | asymmetric smoothing: alarms are raised slowly and cleared fast |
| Sitting speed veto | if confirmed SITTING and recent step speed > **3.0 px/frame**, reset to NORMAL | a moving person cannot be sitting; stops ID theft from carrying a false alarm |
| Lost-track buffer | a vanished track is kept **30 frames** and re-stitched by IoU ≥ 0.3 | survives brief occlusion |
| Stationary guard | a lost track that is >50% stationary may only be re-adopted by a detection within **30 px** of its last centre | stops a passer-by from "stealing" a seated person's id |
| Rule fallback (no classifier) | opposite if `cos_sim < −0.5`, speed ≥ MIN_SPEED and age ≥ 3, confirmed over 5 frames | graceful degradation |
| `nearby` count | tracks within a **150 px** radius | context for alerts |

## Stride compensation (CrowdGuard only)

The classifier was trained on consecutive frames at 27 fps. When CrowdGuard analyses only every *s*-th frame (`anomaly_stride`), `AnomalyConfig.scaled_for_stride(s)` rescales every frame-counted and per-frame-pixel constant so that it covers the same *time*:

| Constant | Rule | s = 1 | s = 7 (shipped) |
| :--- | :--- | ---: | ---: |
| direction window | max(2, round(15/s)) | 15 | 2 |
| history length | max(window+2, round(60/s)) | 60 | 9 |
| min track frames | max(3, round(10/s)) | 10 | 3 |
| confirm frames | max(2, round(5/s)) | 5 | 2 |
| min speed (px/step) | 2.0·s | 2.0 | 14.0 |
| sitting veto (px/step) | 3.0·s | 3.0 | 21.0 |
| stationary re-stitch distance | 30·s | 30 | 210 |
| lost-track buffer | max(1, round(30/s)) | 30 | 4 |

**Limitation:** this corrects the *linear* quantities only. `straightness` and `cos_sim` depend on path curvature, which is lost when frames are skipped, so they do not rescale exactly. Anomaly predictions at s > 1 are an approximation of the trained model; to our knowledge they have not been evaluated against labels at s = 7 (see [10](10_limitations_future_work.md)).
