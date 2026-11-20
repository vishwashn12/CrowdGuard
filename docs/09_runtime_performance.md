# 09: Runtime performance and engineering findings

**Hardware.** NVIDIA GeForce RTX 3050 6 GB Laptop GPU (driver 581.86) and an Intel 12th-generation CPU (family 6, model 151), on Windows 11. All timings are medians over 30 frames after warm-up, on 1440×1080 @ 27 fps footage, unless noted otherwise.

## Single-stream model latency

| Model / setting | Median | p95 | Single-stream fps |
| :--- | ---: | ---: | ---: |
| YOLOv8m-pose @960, FP32 | 48.4 ms | 49.9 ms | 20.7 |
| YOLOv8m-pose @960, FP16 | 26.8 ms | 29.8 ms | 37.4 |
| **YOLOv8m-pose @1280, FP16 (shipped)** | **40.7 ms** | | 24.6 |
| CSRNet 720×540, 3 scales | 204.4 ms | 207.7 ms | 4.9 |
| **CSRNet 720×540, 1 scale (shipped)** | **74.6 ms** | 76.7 ms | 13.4 |

**FP16** is about 1.8× faster for the pose model (48.4 → 26.8 ms), with identical detections in our checks.

## Budget and stride selection

`scripts/benchmark_gpu.py` measures both branches and solves for the smallest sustainable `anomaly_stride`. The inputs are:
- a device budget of 80%;
- a contention factor of 1.15 for more than one camera;
- density cost = cameras × density_fps × CSRNet ms;
- pose cost = cameras × (27 / stride) × pose ms (with the classifier on).

**Example of the first, failing configuration.** CSRNet 3-scale at 5 fps per camera on 3 cameras needed 3,066 ms of GPU per second. The density branch alone saturated the device, leaving 0 ms for pose. This forced the move to single-scale CSRNet and a lower `density_fps`.

**Shipped configurations (measured):**

| Cameras | `density_fps` | `anomaly_stride` | Analysed fps per gate | Playback fps |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 2 | 13.5 | 27 |
| 2 | 1.0 | 5 | 5.4 | 27 |
| 3 | 0.5 | **7** (shipped) | 3.9 | 27 |

The stride sets only how often boxes and labels refresh; playback always runs at the footage's full frame rate.

## Live system measurements (3 cameras, shipped config)

| Metric | Value |
| :--- | :--- |
| Frames read per camera | 27.0 fps |
| Frames rendered per camera | 26.5–27.0 fps |
| Distinct frames received by the browser per gate | 27.0 fps (27.1 drawn in an automated browser test) |
| Frames skipped / refused | 0 / 0 |
| Motion observations accepted by the synchroniser | 97% |
| Producer start skew (frame 0, 3 gates) | < 30 ms |
| Start request latency, real browser flow | 12 ms |
| Browser sockets to the server | 2, regardless of gate count |
| With a browser drawing all 3 streams | 26.7 fps playback per gate; 11.4 analysed frames/s in total (3.8 per gate) |

## Engineering findings: bugs found by measurement

Each of these is a measurable effect, useful for an "implementation lessons" or "system validation" subsection.

| # | Symptom | Root cause | Fix | Measured effect |
| ---: | :--- | :--- | :--- | :--- |
| 1 | The synchroniser rejected 100% of motion as stale | the producer paced only *emitted* frames, so files were read at 161 fps instead of 27 and the pose queue backed up | pace every frame read | read rate 161 → 27.0 fps; valid motion 0% → 86% |
| 2 | Still 14% stale at stride 7 | the stale gap (10 frames) was below the natural gap at stride 7 | `stale_gap = max(10, 2 × stride)` | 86% → 97% valid |
| 3 | The benchmark recommended too-low strides | a 1.6 contention factor hid bug 1 | corrected to 1.15 after the fix | projections matched the live load |
| 4 | Every gate read LOW | `max_expected_velocity` = 15 px/s against a real 70–78 px/s, so the velocity term saturated | 75 px/s plus the `VELOCITY_UNCALIBRATED` rule | the velocity term contributes again |
| 5 | Anomaly windows were 7× too long | `scaled_for_stride` existed but was never called | applied per packet stride; tests added | heading window 3.9 s → 0.56 s at stride 7; the sitting veto is no longer 7× too strict |
| 6 | Playback fell to 9–14 fps with a browser open | one shared render thread made about 50 OpenCV calls per frame, each releasing and re-acquiring the GIL (a convoy effect) | one thread per camera and batched `polylines` | 26.7 fps playback; analysis 11.4 frames/s |
| 7 | Analysis drifted 7 s behind the video under load | the pose queue had no latency bound | skip frames that waited > 0.8 × the display delay | boxes stay aligned with the video; skips are reported |
| 8 | The UI was stuck on "READY" after Start | Chrome's 6-connections-per-host limit was used up by MJPEG streams | all video over a single WebSocket | Start answered in 12 ms; 2 sockets total |
| 9 | Buttons did nothing after an update | the browser cached an old `app.js` | `no-cache` plus versioned asset URLs | — |

Lesson worth stating: **several of these failures were silent.** Bugs 1, 4 and 5 produced plausible numbers rather than errors. Instrumentation (per-stage fps, synchroniser validity ratio, continuity-break ratio, saturation advisories) is what exposed them. The system now reports each of these at `/api/stats` and as alerts.

## Measured trade-offs (summary)

| Lever | Cost | Gain |
| :--- | :--- | :--- |
| CSRNet 3 → 1 scale | −10% count | about 2.7× faster |
| CSRNet 720×540 → 512×384 | −23% count | about 1.7× faster: **the wrong lever** |
| Pose @960 → @1280 (m) | about 1.5× slower | seated recall 16% → 26% |
| Pose m → l @1280 | about 1.6× slower | **no gain** (26% → 24%) |
| FP32 → FP16 | none observed | about 1.8× faster |
| Stride 1 → 7 | anomaly features approximate | 3 cameras fit on a 6 GB laptop GPU |
