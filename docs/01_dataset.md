# 01: Dataset

## Source

The **HAJJv2** dataset, as named by `crowd_management/examples/validate_csrnet.py`. It contains fixed-camera CCTV of Hajj pilgrim crowds with per-object bounding-box annotations of abnormal behaviour.

> **TODO:** cite the dataset's original publication and licence. Neither prototype repository records them.

Local copies (read-only):
- `realtime/Training/Videos/{1..8}.mp4`, `realtime/Training/Labels/HajjDataset_V{n}_Train.csv`
- `realtime/Testing/Videos/{1,2}.mp4`, `realtime/Testing/Labels/HajjDataset_V{n}_Test.csv`

## Videos

All 10 videos are **1440×1080 at 27.0 fps**. Measured with OpenCV.

| Split | Video | Frames | Duration |
| :--- | :--- | ---: | ---: |
| Train | 1 | 700 | 25.9 s |
| Train | 2 | 700 | 25.9 s |
| Train | 3 | 700 | 25.9 s |
| Train | 4 | 700 | 25.9 s |
| Train | 5 | 501 | 18.6 s |
| Train | 6 | 492 | 18.2 s |
| Train | 7 | 466 | 17.3 s |
| Train | 8 | 473 | 17.5 s |
| **Train total** | 8 | **4,732** | **175.3 s** |
| Test | 1 | 700 | 25.9 s |
| Test | 2 | 501 | 18.6 s |
| **Test total** | 2 | **1,201** | **44.5 s** |

The split is the dataset's own: train and test are separate videos, so no frames leak between them.

## Annotation format

These are CSV files with columns including `Frame_No` (1-indexed), `Obj_Name` (a persistent object id), `Classes`, `X`, `Y`, `Width`, `Height`. Only **abnormal or notable people** are annotated, typically 12–15 boxes per frame. Ordinary walking pilgrims are **not** annotated, although a frame contains roughly 40–190 people (from the CSRNet estimate). This matters in two places:

1. "Normal" training labels must be inferred: any detected person who matches no annotation counts as normal. See [03](03_anomaly_classifier.md).
2. The per-frame annotation count **is not a crowd count**. See [04](04_ablation_and_validity.md) on the invalid CSRNet validation.

### Annotation classes

| Class | Train boxes | Test boxes | Used by the classifier as |
| :--- | ---: | ---: | :--- |
| `sitting` | 28,203 | 6,704 | **sitting** (label 1) |
| `moving_in_opposite` | 20,569 | 1,296 | **opposite** (label 2) |
| `standing` | 18,168 | 1,574 | excluded |
| `different_crowd_direction` | 4,723 | 677 | excluded |
| `sleeping` | 1,873 | 501 | excluded |
| `moving_non_human_object` | 1,380 | 0 | excluded |
| `running` | 190 | 51 | excluded |
| **Total** | **75,106** | **10,803** | |

### Per-video annotation breakdown

| Video | Boxes | Annotated frames | Distinct objects | sitting | opposite | standing | diff. direction | sleeping | non-human | running |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train 1 | 15,352 | 700 | 41 | 3,619 | 4,055 | 6,047 | 1,449 | 0 | 182 | 0 |
| Train 2 | 12,166 | 700 | 25 | 7,164 | 2,899 | 1,086 | 317 | 700 | 0 | 0 |
| Train 3 | 15,105 | 700 | 33 | 5,977 | 4,883 | 2,523 | 1,022 | 0 | 700 | 0 |
| Train 4 | 8,117 | 700 | 19 | 2,077 | 2,408 | 2,781 | 151 | 700 | 0 | 0 |
| Train 5 | 2,663 | 501 | 7 | 1,503 | 240 | 839 | 0 | 0 | 0 | 81 |
| Train 6 | 5,415 | 492 | 25 | 2,188 | 874 | 1,975 | 346 | 0 | 32 | 0 |
| Train 7 | 9,570 | 466 | 30 | 3,182 | 3,262 | 1,222 | 1,438 | 0 | 466 | 0 |
| Train 8 | 6,718 | 473 | 18 | 2,493 | 1,948 | 1,695 | 0 | 473 | 0 | 109 |
| Test 1 | 4,160 | 700 | 13 | 2,100 | 838 | 1,171 | 0 | 0 | 0 | 51 |
| Test 2 | 6,643 | 501 | 20 | 4,604 | 458 | 403 | 677 | 501 | 0 | 0 |

Every frame of every video carries at least one annotation.

**Distinct objects are few.** There are 198 annotated objects in training and 33 in test. The number of independent anomalous *people* is therefore small, even though there are tens of thousands of boxes. Report it in the paper: per-frame rows of the same person are highly correlated.

## Derived feature dataset

`realtime/extract_features.py` turns the videos into one row per tracked person per frame (method in [03](03_anomaly_classifier.md)). Files: `features_dataset.csv` (train) and `test_features_dataset.csv` (test).

| Video | normal | sitting | opposite | total |
| :--- | ---: | ---: | ---: | ---: |
| Train 1 | 21,413 | 587 | 105 | 22,105 |
| Train 2 | 24,799 | 335 | 204 | 25,338 |
| Train 3 | 22,137 | 311 | 812 | 23,260 |
| Train 4 | 22,754 | 598 | 278 | 23,630 |
| Train 5 | 19,769 | 411 | 172 | 20,352 |
| Train 6 | 19,035 | 469 | 169 | 19,673 |
| Train 7 | 13,861 | 490 | 150 | 14,501 |
| Train 8 | 16,359 | 334 | 423 | 17,116 |
| **Train** | **160,127 (96.5%)** | **3,535 (2.1%)** | **2,313 (1.4%)** | **165,975** |
| Test 1 | 30,407 | 358 | 393 | 31,158 |
| Test 2 | 18,345 | 98 | 198 | 18,641 |
| **Test** | **48,752 (97.9%)** | **456 (0.9%)** | **591 (1.2%)** | **49,799** |

### Label yield: annotation boxes → labelled feature rows

| Class | Train boxes | Train rows | Yield | Test boxes | Test rows | Yield |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| sitting | 28,203 | 3,535 | **12.5%** | 6,704 | 456 | **6.8%** |
| opposite | 20,569 | 2,313 | 11.2% | 1,296 | 591 | 45.6% |

Test video 2 has 4,604 sitting boxes but only **98** sitting rows (2.1%). Only about one annotated seated person in 8 to 15 becomes a training or test example. A row is produced only when all three conditions below hold:

1. the detector finds the person (see [06](06_detector_evaluation.md): about 16–26% recall for seated people at IoU 0.3);
2. ByteTrack keeps the track for at least 10 frames;
3. the track box overlaps the hand-drawn box with IoU ≥ 0.3.

This is a central fact for the paper. The classifier is trained and evaluated only on the seated people the detector can see, which are a biased subset (larger, less occluded people). Test-set metrics are therefore **conditional on detection**, not end-to-end.
