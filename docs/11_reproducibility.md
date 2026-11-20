# 11: Reproducibility

## Environment

| Item | Value |
| :--- | :--- |
| OS | Windows 11 Home (build 26300) |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop GPU, driver 581.86 |
| CPU | Intel 12th generation (family 6, model 151) |
| Python | 3.11.9 |
| CrowdGuard venv | torch 2.5.1+cu121, ultralytics 8.4.215, opencv 5.1, numpy 2.4.6, catboost 1.2.10, scikit-learn 1.9.1, fastapi 0.164, uvicorn 0.54, pydantic 2.14, pytest 9.2 |
| `realtime` venv (training) | imbalanced-learn 0.14.2, xgboost 3.2.0, lightgbm 4.7.0, catboost (GPU) |

## Seeds and determinism

- SMOTE, RandomUnderSampler and all four classifiers use `random_state=42`.
- The shipped CatBoost was trained with `task_type='GPU'`. GPU training is not bit-identical to CPU training: a CPU retrain of the same configuration gave macro F1 0.662 against the shipped 0.667. Re-score the pickled model to reproduce the reported numbers exactly.

## Reproduce the classifier (in `realtime/`)

```bash
# 1. Features (one row per tracked person per frame)
python extract_features.py --model yolov8m-pose.pt                  # -> features_dataset.csv
python extract_features.py --model yolov8m-pose.pt --test-mode      # -> test_features_dataset.csv
#    defaults: --imgsz 960 --conf 0.15 (IoU 0.5); pass the pose model explicitly,
#    the script's own default is the detection-only yolov8m.pt

# 2. Train 4 models, select by macro F1 on the held-out test CSV, save best_model.pkl
python train_model.py --dataset features_dataset.csv --test_dataset test_features_dataset.csv
```

## Re-score the shipped model (no training)

```python
import pickle, numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
b = pickle.load(open("crowdguard/assets/best_model.pkl", "rb"))
df = pd.read_csv("realtime/test_features_dataset.csv")
X, y = df[b["feature_cols"]].values.astype(np.float32), df["label"].values
pred = np.asarray(b["classifier"].predict(X)).reshape(-1).astype(int)
proba = b["classifier"].predict_proba(X)
print(precision_recall_fscore_support(y, pred, labels=[0, 1, 2]))
print(confusion_matrix(y, pred, labels=[0, 1, 2]))
print([roc_auc_score((y == k).astype(int), proba[:, k]) for k in range(3)])
# expected: macro F1 0.667; AUC 0.932 / 0.908 / 0.954
```

The controlled ablation script used for [04](04_ablation_and_validity.md) is short. It is the same CatBoost and balancing with only the feature columns changed, so it can be rebuilt from the description there. Consider committing a copy to `crowdguard/scripts/` if the paper needs it.

## Run CrowdGuard

```bash
cd crowdguard
venv\Scripts\activate
python -m pytest tests -q                                  # ~920 tests incl. 600 random invariant cases; 4 xfail = known engine weaknesses
python scripts/evaluate_decisions.py                       # closed-loop policy evaluation, CPU only, ~15-20 min on 6 cores
python scripts/evaluate_decisions.py --quick               # 20 seeds, ~4 min
python scripts/benchmark_gpu.py --video "path/to/clip.mp4" --cameras 3
python scripts/run_server.py                               # http://localhost:8000
python scripts/run_server.py --config config/app/demo_samples.yaml   # 3 demo gates staged
curl -s localhost:8000/api/stats                           # buffers, latency, synchroniser health
```

## Shipped configuration (`config/app/default.yaml`)

| Key | Value |
| :--- | :--- |
| `ingestion.anomaly_stride` | 7 |
| `ingestion.density_fps` | 0.5 |
| `models.posetrack.model_path` | `assets/yolov8m-pose.pt` |
| `models.posetrack.imgsz` / `fp16` | 1280 / true |
| `models.posetrack.confidence_threshold` / `iou_threshold` | 0.15 / 0.5 |
| `models.posetrack.tracker_config` | `bytetrack.yaml` |
| `models.csrnet.inference_resolution` | [720, 540] |
| `models.csrnet.scales` | [1.0] |
| `display.delay_s` / `width` / `jpeg_quality` | 0.5 / 960 / 75 |
| `DecisionConfig` | defaults in [07](07_congestion_decision_alerts.md) |

## Artifact inventory

| File | Size | Role |
| :--- | ---: | :--- |
| `crowdguard/assets/best_model.pkl` | 651 KB | CatBoost classifier (trained) |
| `crowdguard/assets/yolov8m-pose.pt` | 53.3 MB | pose detector (pretrained, COCO-Pose) |
| `crowdguard/checkpoints/PartAmodel_best.pth` | 130 MB | CSRNet (pretrained, ShanghaiTech A) |
| `crowdguard/assets/yolov8{s,l}-pose.pt` | 23.5 / 89.4 MB | used only in the detector study |
| `realtime/features_dataset.csv`, `test_features_dataset.csv` | | feature datasets |
| `realtime/results/version_{1,2}/*.png` | | prototype plots (titles say "LOVO", see [04](04_ablation_and_validity.md)) |
| `realtime/catboost_info/learn_error.tsv` | | CatBoost training loss |

## Commit history of the prototypes

| Repository | Commit | Date | Change |
| :--- | :--- | :--- | :--- |
| realtime | `07baf6a` | 2026-08-27 | V1: Random Forest, 13 trajectory features |
| realtime | `38cb59a` | 2026-08-29 | V2: CatBoost with pose features; test split added |
| realtime | `6054327` | 2026-11-11 | documentation |
| crowd_management | `252d4fa` | 2026-08-28 | ingestion and models (CSRNet, RAFT) |
| crowd_management | `546f196` | 2026-08-30 | decision layer |
| crowd_management | `74aa4b9` | 2026-08-31 | RAFT replaced with YOLOv8 + ByteTrack; env config |
| crowd_management | `780daba` | 2026-10-15 | 3-gate simulation and advisory engine |

CrowdGuard is a new codebase. It reads the prototypes' assets and logic but imports neither and modifies neither (`realtime/` is clean; `crowd_management/requirements.txt` has a pre-existing local modification that is not ours).
