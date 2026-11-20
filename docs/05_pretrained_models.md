# 05: Pretrained models and what was not trained

| Component | File | Trained by | Trained on | Fine-tuned here? |
| :--- | :--- | :--- | :--- | :--- |
| Pose detector | `assets/yolov8m-pose.pt` | Ultralytics | COCO-Pose (COCO 2017 person keypoints) | **No** |
| Tracker | Ultralytics `bytetrack.yaml` | algorithmic, no weights | — | — |
| Density / counting | `checkpoints/PartAmodel_best.pth` | external (CSRNet) | ShanghaiTech Part A | **No** |
| Anomaly classifier | `assets/best_model.pkl` | **this project** | Hajj training videos 1–8 | trained from scratch ([03](03_anomaly_classifier.md)) |
| Decision engine, alert rules | code | hand-written | — | not learned |

## YOLOv8m-pose

Checkpoint metadata, read from the `.pt` file:

| Field | Value |
| :--- | :--- |
| Ultralytics version at training | 8.0.77 |
| Date | 2023-04-19 |
| Model config | `yolov8m-pose.yaml` |
| Data | `coco-pose.yaml` |
| Training image size | 640 |
| Epochs | 1000 (schedule; best checkpoint shipped) |
| Batch | 128 |
| Optimiser | SGD |
| `pretrained` | False (trained from scratch on COCO-Pose) |
| Parameters | 26,464,462 (≈26.5 M) |
| Keypoint shape | 17 × 3 (x, y, visibility/confidence), COCO order |
| Classes | 1 (person) |

How it is used:
- **Prototype (training-feature extraction).** imgsz 960, confidence 0.15, IoU 0.5.
- **CrowdGuard (inference).** imgsz **1280**, **FP16**, confidence 0.15, IoU 0.5, `persist=True` ByteTrack.
- CrowdGuard keeps **one model instance per camera**, because Ultralytics stores ByteTrack state inside the predictor. A shared instance would mix tracks across scenes.
- ByteTrack is reset whenever a clip loops or a session restarts.

Why 1280 rather than 960 or a larger model: see [06](06_detector_evaluation.md). Resolution mattered more than model size. Note that 1280 is twice the training resolution of 640, which helps the small, distant people in 1440×1080 CCTV.

Alternatives on disk but unused: `yolov8s-pose.pt` (23.5 MB) and `yolov8l-pose.pt` (89.4 MB). Shipped: `yolov8m-pose.pt` (53.3 MB).

## CSRNet

Architecture: VGG-16 front-end (first 10 conv layers) plus a dilated-convolution back-end, with a density-map output that is summed to give the count (Li et al., CVPR 2018).

Checkpoint metadata, read from the `.pth` file:

| Field | Value |
| :--- | :--- |
| Keys | `state_dict, epoch, arch, optimizer, best_prec1` |
| Epoch | 376 |
| `best_prec1` | **65.90**: by the convention of the common PyTorch CSRNet re-implementation this field stores the best test **MAE** on ShanghaiTech Part A |
| File size | 130 MB |

> **TODO:** confirm where the checkpoint came from (the repo does not record it) and that `best_prec1` is the Part-A test MAE. For comparison, the CSRNet paper reports Part-A MAE 68.2 / MSE 115.0. Check that against the paper before citing.

How it is used in CrowdGuard:
- Input: frame resized to **720×540**, ImageNet mean/std normalisation.
- `scales`: configurable, and must include 1.0. The prototype used **[0.7, 1.0, 1.3]** averaged; CrowdGuard ships **[1.0]** (single scale) for speed.
- Sampling: `density_fps` 0.5 per camera in the shipped 3-camera config (5.0 is the schema default).
- Output used: `crowd_count` only. The density heat-map is no longer shown in the UI.

Scale and resolution trade-off, measured on one demo frame on an RTX 3050 ([09](09_runtime_performance.md)):

| Input | Scales | ms / frame | Count | vs. 720×540 ×3 |
| :--- | :---: | ---: | ---: | ---: |
| 720×540 | 3 | 203.0 | 157.3 | baseline |
| **720×540** | **1 (shipped)** | **74.1** | **140.9** | **−10.4%** |
| 512×384 | 3 | 118.0 | 121.7 | −22.6% |
| 512×384 | 1 | 41.0 | 114.0 | −27.5% |
| 384×288 | 3 | 69.4 | 86.5 | −45.0% |
| 384×288 | 1 | 28.0 | 83.4 | −47.0% |

These counts are relative only; there is no ground truth. The conclusion still holds: **dropping scales is a cheaper loss than dropping pixels.** Lowering resolution loses far more count than dropping to single scale. Under-counting makes the system under-react, so gate capacities must be calibrated against the scale set in use. CrowdGuard logs a warning at startup when running single-scale.

**Validation status:** there is no valid counting evaluation on Hajj footage ([04](04_ablation_and_validity.md), correction 4).

## Motion estimation history (from `crowd_management`)

| Date (commit) | Motion source |
| :--- | :--- |
| 28 Aug 2026 (`252d4fa`) | RAFT optical flow |
| 31 Aug 2026 (`74aa4b9`) | replaced by YOLOv8l detection + ByteTrack (track-centre displacement) |
| CrowdGuard | derived from the **same YOLOv8m-pose + ByteTrack pass** as the anomaly branch; no separate motion model |

Crowd motion in CrowdGuard, per analysed frame:
- **velocity** (px/s): the median over tracks of $\lVert c_t - c_{t-1} \rVert \cdot f_{\text{anomaly}}$, where $c$ is the track centre and $f_{\text{anomaly}} = f_{\text{source}} / \text{stride}$. Using px/s keeps the value comparable across strides.
- **flow consistency**: $\lVert \sum_i d_i \rVert / \sum_i \lVert d_i \rVert$ over track displacements $d_i$, in [0, 1]. 1 means everyone moves the same way.
- **dominant direction**: $\operatorname{atan2}(\sum d^y, \sum d^x)$.
- **active ratio**: the share of tracks with velocity above `motion_threshold_px` (2.0 px/s).
- **motion confidence**: $\min(1, 0.5\,\text{flow} + 0.3\,\text{active} + 0.2\,\min(1, n_{\text{tracks}}/20))$.
