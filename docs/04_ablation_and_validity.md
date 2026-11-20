# 04: Ablation, feature importance and validity checks

This file records what we **verified** about claims in the prototype documentation and corrects the ones that do not hold. Use these numbers, not the prototype's, in the paper.

## Controlled feature ablation

**Setup.** Only the feature set changes. The model is CatBoost (300 iterations, depth 6, lr 0.05, seed 42). Balancing is the same SMOTE + undersampling (25k / 25k / 100k). Train is `features_dataset.csv`; test is `test_features_dataset.csv`. The script was run from a scratch directory and wrote nothing to the prototype.

| Feature set | Sitting P | Sitting R | Sitting F1 | Opposite P | Opposite R | Opposite F1 | Macro F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All 10 (shipped set) | 0.141 | 0.349 | **0.201** | 0.769 | 0.832 | **0.799** | **0.662** |
| Without pose (7) | 0.128 | 0.384 | 0.191 | 0.663 | 0.822 | 0.734 | 0.636 |
| Without `pos_y_norm` (9) | 0.135 | 0.375 | 0.198 | 0.703 | 0.846 | 0.768 | 0.650 |
| Without pose and `pos_y_norm` (6) | 0.115 | 0.436 | 0.182 | 0.562 | 0.853 | 0.677 | 0.612 |

The "All 10" row gives 0.662 rather than the shipped model's 0.667 because the ablation trained CatBoost on CPU, while the shipped model was trained with `task_type='GPU'`. GPU and CPU CatBoost use different histogram and random-number implementations. Compare rows *within* this table only.

Findings:
- **The pose features help, modestly.** Sitting F1 goes 0.191 → 0.201 (+0.010) and opposite F1 goes 0.734 → 0.799 (+0.065). Most of the gain is on **opposite**, probably because keypoints sharpen the box and centre, which stabilises the trajectory.
- **Pose mostly buys precision.** Without pose, the sitting and opposite classes get *higher recall but lower precision*.
- **`pos_y_norm` is not a hidden crutch.** Removing it costs 0.012 macro F1. That matters for transfer to new camera angles, where vertical position means something different.

## Correction 1: the original "pose ablation" claim

`realtime/docs/project_summary.md` and the docstring of `crowdguard/models/posetrack/features.py` state: *"Without pose features, Sitting F1 dropped from ~0.14 down to 0.03."* This was **not a controlled comparison**:
- The 0.143 is the Random Forest with all 10 features (from the model-comparison table).
- The 0.033 comes from the Random Forest ablation subsets in `train_model.py` ("Motion only", "Appearance only", and so on). Those subsets remove much more than pose.

With everything else held fixed, the drop is 0.201 → 0.191. **Cite the controlled table above.** The `features.py` docstring in CrowdGuard repeats the old claim and should be updated to match.

## Correction 2: feature-importance ranking

The prototype README ranks `cos_sim` first. That ranking comes from a **Random Forest** (Gini) refit, not from the shipped CatBoost model.

Random Forest Gini importance, **reported**, top 7 only:

| Rank | Feature | Importance |
| ---: | :--- | ---: |
| 1 | cos_sim | 0.1664 |
| 2 | direction_sin | 0.1632 |
| 3 | pos_y_norm | 0.1462 |
| 4 | aspect_ratio | 0.1007 |
| 5 | displacement | 0.0813 |
| 6 | head_to_hip_ratio | 0.0794 |
| 7 | knee_bend_angle | 0.0790 |

Shipped CatBoost importance, **measured** with `get_feature_importance()` (PredictionValuesChange, which sums to 100):

| Rank | Feature | Importance |
| ---: | :--- | ---: |
| 1 | pos_y_norm | 17.86 |
| 2 | direction_sin | 13.48 |
| 3 | straightness | 11.80 |
| 4 | aspect_ratio | 11.07 |
| 5 | displacement | 11.00 |
| 6 | direction_cos | 9.37 |
| 7 | cos_sim | 8.11 |
| 8 | head_to_hip_ratio | 7.28 |
| 9 | torso_height_ratio | 5.68 |
| 10 | knee_bend_angle | 4.35 |

How to read these:
- **`pos_y_norm` ranks first, yet removing it costs little.** Importance reflects how much the model *uses* a feature, not what it *needs*: correlated features can substitute for it. Report importance and ablation together.
- **Absolute heading outranks `cos_sim`.** `direction_sin` (13.5) is above `cos_sim` (8.1). The model may be partly learning the *scene-specific* flow direction of these fixed cameras rather than a relative "against the crowd" concept. This is a generalisation risk for new cameras and should be tested ([10](10_limitations_future_work.md)).

## Correction 3: evaluation protocol label

Both result plots are titled "(LOVO)", because the title string is hard-coded in `train_model.py`. The V2 numbers come from the **explicit train/test split**. Describe it as "held-out test videos". For V1, the protocol and dataset cannot be confirmed ([03](03_anomaly_classifier.md)).

## Correction 4: CSRNet was never validated on Hajj footage

`crowd_management/examples/validate_csrnet.py` reports an MAE (mean absolute error) between the CSRNet count and a "ground-truth count". That count comes from `utils/gt_loader.py`, which returns the **number of annotation boxes in the frame**.

Measured on test video 2: 12 to 15 boxes per frame (median 13). CSRNet counts about 40 to 190 people in the same frames. The annotations cover only notable or abnormal individuals, not the crowd. **The MAE therefore measures nothing about counting accuracy.** No valid Hajj counting result exists. The only counting evidence is the published ShanghaiTech Part-A figure ([05](05_pretrained_models.md)).

## Threats to validity (hypotheses we could not test)

1. **Label noise for sitting (untested hypothesis).** Seated people are often partly occluded, and the detector draws a tight box around the visible upper body. Against a loose hand-drawn box, the IoU can fall below 0.3, so the row is labelled *normal*. Some of the 919 "false" sitting predictions may therefore be real seated people with wrong labels. **This could not be verified:** the feature CSVs do not store boxes, so rows cannot be traced back to annotations. Testing it requires re-extracting with boxes saved, then inspecting a sample of false positives by eye.
2. **Correlated samples.** Rows are frames, not people. The effective sample size is closer to the number of tracked individuals (198 annotated objects in train, 33 in test) than to 165,975.
3. **Two test videos, same site.** Camera viewpoints, crowd type and lighting are shared with the training data. Cross-site performance is unknown.
4. **No repeated runs.** One seed; no variance or confidence intervals.
5. **Detection ceiling.** See [06](06_detector_evaluation.md): the classifier can only label people the detector finds.
