# CrowdGuard research notes

These are reference notes for writing a paper on CrowdGuard. CrowdGuard is a multi-gate crowd-management system with per-person anomaly detection. It combines two earlier prototypes:

| Prototype | What it did | Location |
| :--- | :--- | :--- |
| `realtime` | Single-camera anomaly detection: YOLOv8m-pose + ByteTrack + 10 hand-built features + CatBoost (classes normal / sitting / opposite-direction) | `../../realtime/` |
| `crowd_management` | Multi-camera congestion scoring and rule-based diversion: CSRNet density + YOLOv8l/ByteTrack motion + decision engine | `../../crowd_management/` |

Every number here was either read from the code or artifacts, or measured on the development machine. Each section names its source. A figure is labelled **reported** when it comes from a prototype's own documentation and was not re-run. **TODO** marks anything that must be checked before you cite it.

## Files

| File | Contents | Paper section it feeds |
| :--- | :--- | :--- |
| [00_problem_statement.md](00_problem_statement.md) | Problem statement, motivation (incidents, crowd science), use cases, users, scope, research questions | Introduction, Motivation |
| [01_dataset.md](01_dataset.md) | Hajj videos, annotation classes, per-video counts, train/test split, feature-row yield | Dataset |
| [02_feature_engineering.md](02_feature_engineering.md) | The 10 features with formulas, tracking constants, crowd-direction estimate | Method: features |
| [03_anomaly_classifier.md](03_anomaly_classifier.md) | Labelling, balancing, 4 models + hyperparameters, selection, results, V1 vs V2 | Method + Results |
| [04_ablation_and_validity.md](04_ablation_and_validity.md) | Controlled ablation, feature importance, corrections to earlier claims, threats to validity | Ablation, Discussion |
| [05_pretrained_models.md](05_pretrained_models.md) | YOLOv8m-pose, CSRNet, ByteTrack: provenance, checkpoint metadata, what was *not* trained | Method: components |
| [06_detector_evaluation.md](06_detector_evaluation.md) | Recall of labelled anomalous people for 8 detector configurations | Experiments |
| [07_congestion_decision_alerts.md](07_congestion_decision_alerts.md) | Congestion score, routing score, hysteresis, safety refusals, all alert rules and thresholds | Method: decision layer |
| [08_system_architecture.md](08_system_architecture.md) | Layers, data flow, dual-rate ingestion, session clock, playback, web stack | System design |
| [09_runtime_performance.md](09_runtime_performance.md) | GPU benchmarks, stride selection, live measurements, bugs found and their measured effect | Experiments: real-time |
| [10_limitations_future_work.md](10_limitations_future_work.md) | What is unvalidated, known weaknesses, next experiments | Limitations, Future work |
| [11_reproducibility.md](11_reproducibility.md) | Versions, hardware, commands, seeds, file inventory, commit history | Reproducibility / appendix |
| [12_congestion_control_evaluation.md](12_congestion_control_evaluation.md) | How the rule-based decisions are tested: scenario suite, invariants, closed-loop simulation; baseline results for engine v1 and its known weaknesses | Method: evaluation, Results |
| [13_instructions_and_calibration.md](13_instructions_and_calibration.md) | Operator commands/incidents vocabulary; 4-point ground calibration (people/m², m/s); CSRNet near-camera under-count finding | Method: decision support UI, calibration; Discussion |
| [backlog.md](backlog.md) | Agreed but deferred work: engine v2 plan, ablation run-time estimates, feature removal, better inputs | (planning, not a paper section) |

## Key numbers

| Item | Value | Source |
| :--- | :--- | :--- |
| Training data | 8 videos, 4,732 frames, 1440×1080 @ 27 fps, 75,106 annotation boxes | label CSVs |
| Test data | 2 videos, 1,201 frames, 10,803 annotation boxes | label CSVs |
| Feature rows | 165,975 train / 49,799 test (per person, per frame) | feature CSVs |
| Classifier | CatBoost, 300 trees, depth 6, lr 0.05, 10 features | `best_model.pkl` |
| Test macro F1 | **0.667** (normal 0.985, sitting 0.223, opposite 0.793) | re-scored from shipped model |
| Test ROC-AUC (one-vs-rest) | normal 0.932, sitting 0.908, opposite 0.954 | re-scored |
| Opposite-direction | precision 0.760, recall 0.829 | re-scored |
| Sitting | precision 0.158, recall 0.379 | re-scored |
| Detector recall on labelled seated people | 26% with the shipped detector (pose-m @1280); 45% best observed (det-l @1440) | own study, IoU ≥ 0.3 |
| Real-time | 3 cameras, 27 fps playback per gate, 3.9 analysed fps per gate, on an RTX 3050 6 GB laptop GPU | live measurement |
| Tests | 954 tests + 4 expected failures (known engine weaknesses), no GPU needed | `pytest` |
| Decision evaluation | engine v1 removes > 99.9% of exposure at ≥5 people/m² vs no control in 6 simulated scenarios; up to 1,426 instruction changes/h | [12](12_congestion_control_evaluation.md) |

## Contributions you can defensibly claim

1. **A single pose pass feeds two consumers.** One YOLOv8m-pose + ByteTrack pass per frame gives both crowd motion (track displacement → velocity, flow consistency) and per-person anomaly features (keypoints, trajectory). The prototypes ran two separate YOLO models over the same frames.
2. **Dual-rate ingestion from a single decode.** The anomaly path runs at a frame stride and density at ≈0.5–5 fps. Both share one frame-id clock, with a causality-checked temporal synchroniser.
3. **Stride compensation.** Frame-counted windows and pixel thresholds of a classifier trained at 27 fps are rescaled for sparse analysis. The limitation is stated: curvature features do not rescale exactly.
4. **A model-agnostic decision layer**, enforced by an AST test. Congestion routing uses no vision imports, and anomalies travel on a separate alert channel instead of entering the congestion score.
5. **Honest handling of gates without cameras.** An unobserved gate is assumed non-empty, routing to it is penalised, and it can never be a diversion source.
6. **Real-time playback decoupled from analysis.** A session clock keeps all gates in lockstep, the display is delayed by 0.5 s, boxes are interpolated between keyframes, and stale overlays are withdrawn.
7. **A measured critique of the prototypes.** See [04](04_ablation_and_validity.md): an uncontrolled ablation, an invalid CSRNet validation, and a mislabelled evaluation protocol.

## Citation TODOs

- **HAJJv2 dataset.** Find the original paper or dataset page and cite it. Neither repo records the source.
- **CSRNet.** Y. Li, X. Zhang, D. Chen, "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes", CVPR 2018 (arXiv:1802.10062).
- **YOLOv8 / Ultralytics.** Cite the software (Jocher et al., Ultralytics YOLOv8, 2023) as Ultralytics recommends.
- **ByteTrack.** Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
- **CatBoost.** L. Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", NeurIPS 2018.
- **SMOTE.** N. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", JAIR 2002.
- **COCO keypoints.** T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
- **ShanghaiTech.** Y. Zhang et al., "Single-Image Crowd Counting via Multi-Column CNN", CVPR 2016.

Check every bibliographic detail above against the publisher before you submit. They are given from memory.
