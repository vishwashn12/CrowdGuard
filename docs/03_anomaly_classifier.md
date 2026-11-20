# 03: Anomaly classifier

The CatBoost classifier is **the only model trained in either prototype**. Everything else is pretrained ([05](05_pretrained_models.md)) or hand-written rules ([07](07_congestion_decision_alerts.md)).

Sources:
- `realtime/extract_features.py` (labelling) and `realtime/train_model.py` (training);
- `realtime/best_model.pkl`, copied to `crowdguard/assets/best_model.pkl`;
- `realtime/results/version_{1,2}/`, `realtime/catboost_info/learn_error.tsv`;
- re-scoring the shipped model on `test_features_dataset.csv`.

## Task

Three-class, per person, per frame: **0 normal · 1 sitting · 2 opposite** (moving against the dominant crowd flow).

**Motivation.** Two localised breakdowns tend to precede crowd crush incidents: a person **sitting or collapsed** inside a moving flow (a trip hazard and stampede nucleus) and people **moving against the flow** (counter-flow turbulence). A 2-D box cannot tell a seated person from someone standing still, which is why skeletal features were added in V2.

## Labelling procedure

For every tracked person with ≥ 10 frames of history, in every frame:

1. Compute the IoU between the track box and every annotation box in that frame.
2. Take the best match. **If IoU ≥ 0.3:**
   - `sitting` → label 1
   - `moving_in_opposite` → label 2
   - any other class (`standing`, `sleeping`, `different_crowd_direction`, `running`, `moving_non_human_object`) → **row dropped**, so these are not mislabelled as normal
3. **If IoU < 0.3 for every box, or the frame has none:** label 0 (normal).

Rationale for IoU 0.3 rather than 0.5: the hand-drawn boxes are looser than detector boxes. A correctly found seated person scored IoU 0.48 in our checks.

Consequence: a seated person whose detector box overlaps the label by less than 0.3 becomes a **normal** row. This is a plausible source of label noise ([04](04_ablation_and_validity.md)).

## Class balancing (training set only)

The `balance_training_set` pipeline from `imbalanced-learn`:

| Step | Setting | Effect |
| :--- | :--- | :--- |
| SMOTE | `sampling_strategy={1: 25000, 2: 25000}`, `k_neighbors=5`, `random_state=42` | sitting 3,535 → 25,000; opposite 2,313 → 25,000 |
| RandomUnderSampler | `{0: 100000}`, `random_state=42` | normal 160,127 → 100,000 |
| Result | **150,000 rows** | 2:1:1 split (normal : sitting : opposite) |

The **test set was left unbalanced** (97.9% normal), matching deployment conditions. No feature scaling was used (tree models).

Caveat for the paper: SMOTE ran on per-frame rows, and consecutive rows of one person are near-duplicates. Synthetic samples are therefore interpolations within a few dozen individuals ([01](01_dataset.md): 198 annotated objects in train).

## Models compared

All use `random_state=42`.

| Model | Hyperparameters |
| :--- | :--- |
| Random Forest | 200 trees, max_depth 15, min_samples_leaf 5, class_weight balanced |
| XGBoost | 300 trees, max_depth 6, lr 0.05, tree_method hist, device cuda, eval mlogloss |
| **CatBoost (selected)** | **300 iterations, depth 6, lr 0.05, task_type GPU, MultiClass loss** |
| LightGBM | 300 trees, max_depth 6, lr 0.05, class_weight balanced |

**Selection rule:** the highest **macro F1** on the held-out test videos. The winner was then refit on the full balanced training set and pickled as `{classifier, scaler=None, feature_cols, label_names, model_name}`.

No hyperparameter search was performed. The values above are hand-set defaults.

## Results: model comparison (test videos 1–2, 49,799 rows, unbalanced)

**Reported** by `realtime/docs/project_summary.md`. Only the CatBoost row was re-scored by us, and it reproduced exactly.

| Model | Normal F1 | Sitting F1 | Opposite F1 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **CatBoost** | **0.985** | 0.223 | **0.793** | **0.667** |
| XGBoost | 0.985 | **0.225** | 0.791 | 0.667 |
| Random Forest | 0.962 | 0.143 | 0.633 | 0.579 |
| LightGBM | 0.962 | 0.142 | 0.591 | 0.565 |

CatBoost and XGBoost tie at three decimals. The selection between them is not meaningful, so say so, or break the tie with AUC or seeds.

## Results: shipped CatBoost, re-scored

Re-scored from `best_model.pkl` on `test_features_dataset.csv`.

| Class | Precision | Recall | F1 | ROC-AUC (OvR) | Support |
| :--- | ---: | ---: | ---: | ---: | ---: |
| normal | 0.992 | 0.978 | 0.985 | 0.932 | 48,752 |
| sitting | 0.158 | 0.379 | 0.223 | 0.908 | 456 |
| opposite | 0.760 | 0.829 | 0.793 | 0.954 | 591 |
| **macro** | 0.637 | 0.729 | **0.667** | 0.931 | 49,799 |

Accuracy is 0.971, which is not meaningful given 97.9% normal; report macro F1 and per-class values instead.

**Confusion matrix** (rows = truth, columns = predicted):

| | normal | sitting | opposite |
| :--- | ---: | ---: | ---: |
| **normal** | 47,692 | 919 | 141 |
| **sitting** | 269 | 173 | 14 |
| **opposite** | 100 | 1 | 490 |

What the matrix shows:
- **False-alarm rate per 1,000 normal rows:** sitting **18.9**, opposite **2.9**.
- **Sitting:** 919 false positives against 173 true positives, so roughly 1 in 6 sitting calls is correct.
- **Opposite:** errors are almost all confusion with normal; only 1 opposite row was called sitting.

**Training curve** (`catboost_info/learn_error.tsv`): MultiClass loss went from 1.038 at iteration 0 to 0.216 at iteration 299. No validation curve was logged, so early stopping or overfitting cannot be assessed from the logs.

The `results/version_2/confusion_matrix.png` plot shows exactly this matrix. Its title says "(LOVO)" but the run used the explicit train/test split; the title string is hard-coded in `train_model.py`. Do not reproduce that label.

## Version history: V1 to V2

| | V1 (commit `07baf6a`, 27 Aug 2026) | V2 (commit `38cb59a`, 29 Aug 2026, shipped) |
| :--- | :--- | :--- |
| Detector during extraction | YOLOv8m (detection) @960 | YOLOv8m-pose @960 |
| Features | 13, no pose ([02](02_feature_engineering.md)) | 10, including 3 pose |
| Classifier | Random Forest | CatBoost (best of 4) |
| Evaluation | plot titled LOVO (leave-one-video-out) | explicit split, test videos 1–2 |

V1 Random Forest confusion matrix, from `results/version_1/confusion_matrix.png` (149,306 rows):

| | normal | sitting | opposite |
| :--- | ---: | ---: | ---: |
| **normal** | 138,077 | 3,799 | 1,707 |
| **sitting** | 1,381 | 1,623 | 213 |
| **opposite** | 643 | 192 | 1,671 |

Metrics we derived from that matrix:

| Class | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: |
| normal | 0.986 | 0.962 | 0.973 |
| sitting | 0.289 | 0.505 | 0.368 |
| opposite | 0.465 | 0.667 | 0.548 |
| macro | | | 0.630 |

The V1 false-sitting rate is 26.5 per 1,000 normal rows.

**V1 and V2 are not directly comparable. Do not present V1 → V2 as an improvement or a regression without re-running.** The reasons:
- **Different evaluation protocol.** V1 is LOVO inside the training videos; V2 uses held-out test videos.
- **Different data.** The V1 matrix totals 149,306 rows. The `features_dataset.csv` committed with V1 has **103,095** rows over 5 videos (ids 2, 3, 5, 7, 8), so the plot came from a dataset version that was never committed. Its provenance is unknown.
- V1's apparently higher sitting F1 (0.37 vs 0.22) could come from LOVO on training-distribution videos rather than from a better model.

A fair comparison means running both feature sets with both protocols. The controlled ablation in [04](04_ablation_and_validity.md) is the defensible substitute.

## Use in CrowdGuard

- Loaded by `crowdguard/models/posetrack/classifier.py`. The stored `feature_cols` must equal `FEATURE_NAMES` in `features.py`, in order, or startup fails. This prevents silent input misalignment.
- Predictions are smoothed, vetoed and stitched by the tracker logic in [02](02_feature_engineering.md).
- Confirmed labels feed the anomaly alert rules `PERSON_DOWN`, `COUNTERFLOW_DETECTED` and `ANOMALY_CLUSTER` ([07](07_congestion_decision_alerts.md)). They are **not** an input to congestion scoring.
- If the pickle is missing, CrowdGuard falls back to the geometric opposite-direction rule only.
- Inference at 1280 px rather than the training resolution of 960 changes detection, not feature scales: boxes and keypoints are returned in original-frame pixels.

## Metric caveats to state in the paper

1. **Row level, not incident level.** One seated person seen for 300 frames contributes about 300 rows. A detector that catches them in 40% of frames scores 40% recall, yet an operator would get an alarm. Incident-level (per-person-episode) recall and false alarms per minute would be more meaningful operationally; they have not been computed yet.
2. **Conditional on detection.** People the detector misses never become rows ([01](01_dataset.md), [06](06_detector_evaluation.md)).
3. **Single seed, no confidence intervals.** Report this, or add bootstrap CIs over persons or videos.
4. **Only 2 test videos and 33 annotated objects.** Generalisation claims must be modest.
