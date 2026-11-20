# 10: Limitations and future work

## Limitations (state these in the paper)

### Data
1. **Small and correlated.** There are 8 training and 2 test videos from one event and site: 175 s of training footage and 44.5 s of test footage. Only 198 annotated objects in train and 33 in test. Rows are per frame, so the effective sample size is the number of people, not the number of rows.
2. **Normal is inferred, not annotated.** Ordinary people carry no labels, and "normal" means "matched no annotation at IoU ≥ 0.3". Missed or loosely matched anomalous people therefore become normal rows.
3. **Low label yield.** Only 12.5% of training sitting boxes (6.8% of test) become labelled rows ([01](01_dataset.md)).

### Anomaly classifier
4. **Sitting is weak.** F1 0.223, precision 0.158, about 19 false sitting rows per 1,000 normal rows. The asymmetric confirmation (5 consecutive predictions) and the speed veto reduce, but do not remove, false alarms at alert level.
5. **Row-level metrics only.** No incident-level recall and no false-alarms-per-hour figure.
6. **Absolute-heading reliance.** `direction_sin` outranks `cos_sim` in importance, so the model may encode these cameras' flow direction. Transfer to new viewpoints is untested.
7. **The stride approximation.** The deployed system analyses 1 frame in 7. Linear constants are rescaled, but curvature features are not. Anomaly accuracy at stride 7 **has not been measured against labels.**
8. **Single seed, no confidence intervals, no hyperparameter search.** CatBoost and XGBoost tie, so the choice between them is arbitrary.

### Detection
9. **A detection ceiling.** The shipped detector finds 26% of labelled seated people and 8% of distant (< 80 px) people at IoU 0.3. The best configuration tested finds 45%.

### Counting and congestion
10. **CSRNet has no valid evaluation on this footage.** The prototype's validation compared against annotation-box counts, not crowd counts. The model is used off the shelf from ShanghaiTech Part A, and the shipped single-scale setting under-counts by about 10% relative to 3-scale.
11. **Congestion and routing are unvalidated heuristics.** All weights and thresholds are hand-set. In the current code, `capacity_utilization` equals `density_score`, so the formula weights one quantity at 0.8.
12. **Calibration is per camera and per venue.** `max_expected_velocity` is in px/s and depends on resolution, zoom and distance. Gate capacities are placeholders. Distances assume an idealised ring.
13. **No perspective correction.** Pixel speeds and sizes vary with depth. Only `pos_y_norm` addresses this.

### System
14. **Single consumer GPU.** Three cameras need stride 7 on an RTX 3050 6 GB. Scaling beyond that needs more GPUs or smaller models.
15. **Evaluated on looped recorded clips, not live CCTV.** RTSP is supported but was not field-tested.
16. **No user study.** Operator usefulness, and the effect of diversion on actual crowd flow, are not evaluated.

## Future work (ranked by expected value per effort)

| # | Experiment | Why it matters | Effort |
| ---: | :--- | :--- | :--- |
| 1 | **Incident-level evaluation:** group rows into per-person episodes; report episode recall, detection delay and false alarms per minute, with and without confirmation and veto | the operational metric reviewers will ask for | low |
| 2 | **Evaluate the classifier at stride 7** (re-extract features at stride 7, score against labels) | validates the deployed configuration | low |
| 3 | **Bootstrap CIs** over persons and videos; 5 seeds | statistical credibility | low |
| 4 | **Re-extract with boxes saved;** inspect sitting false positives | tests the label-noise hypothesis ([04](04_ablation_and_validity.md)) | medium |
| 5 | **Detection-then-pose hybrid:** detection-model boxes, pose on crops | +13 points seated and +15 points distant recall available ([06](06_detector_evaluation.md)) | medium |
| 6 | **Remove absolute heading** (or rotate-augment) and re-evaluate | viewpoint generalisation | low |
| 7 | **Annotate crowd counts** on a few dozen Hajj frames (point annotations); report CSRNet MAE and RMSE at 1 and 3 scales; optionally fine-tune | the first valid counting result | medium |
| 8 | **Fix the congestion formula** (separate density per area from utilisation) and calibrate thresholds against expert-labelled clips | turns the heuristic into a validated score | high |
| 9 | **Leave-one-video-out** over all 10 videos as a second protocol | robustness of the reported numbers | low |
| 10 | **Temporal model** over feature sequences (a GRU or a small transformer on a track window) | uses temporal context the per-frame model ignores | high |
| 11 | **Simulation of diversion policies** (agent-based, using the venue graph) | evaluates the decision layer without real incidents | high |
