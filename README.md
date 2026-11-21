# CrowdGuard

Multi-gate crowd management with per-person anomaly detection, for venues where
footage from several gates is monitored at once.

Two things happen on every camera, from a single pass of one model:

1. **Crowd management.** Density, flow and capacity per gate feed a rule-based
   decision engine that redirects incoming crowd from a congested gate to a
   viable one, along the perimeter ring, with hysteresis so the target does not
   oscillate and with an explanation attached to every decision.
2. **Anomaly detection.** Per-person pose and trajectory features classify
   individuals who are sitting or collapsed on the ground, or moving against the
   dominant flow — the two localised breakdowns that precede most crush events.

This project combines two earlier prototypes: a single-camera anomaly detector
(`../realtime`) and a multi-camera crowd decision system (`../crowd_management`).
Neither of those folders is modified or imported by this one.

---

## Quick start

```bash
cd crowdguard

py -m venv venv
venv\Scripts\activate                 # Windows;  source venv/bin/activate elsewhere

# PyTorch first, matched to your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Dashboard (React + Vite). Needs Node 18+. Without this build the server
# falls back to the original page in frontend/.
cd web-ui && npm install && npm run build && cd ..

python scripts/run_server.py
```

Model weights are not in the repository: put `yolov8m-pose.pt` in `assets/` and
the CSRNet checkpoint `PartAmodel_best.pth` in `checkpoints/`.

Open <http://localhost:8000> and use the **Setup** tab:

1. **How many gates?** Pick 2–12. Gates are laid out on a perimeter ring.
2. **Add footage and calibrate.** Upload a video per gate, then press
   **Calibrate**: click the four corners of a rectangle on the ground in the
   frame and enter its real size in metres. Calibrated gates are measured in
   people/m² and m/s; uncalibrated gates assume they are full at 300 people.
   See `docs/13_instructions_and_calibration.md`.
3. **Start monitoring.** Every gate with footage begins at the same instant and
   plays at the footage's real frame rate. Clips loop when they end.

While monitoring runs, setup is locked. **Stop** returns you to setup with all
footage still in place. Gates left without footage still take part in routing as
diversion targets, assuming 300 people.

To start with feeds already staged, use the demo config, which reads (and never
writes) the sample clips in the sibling project. You still press Start; pass
`--autostart` to skip that for headless use:

```bash
python scripts/run_server.py --config config/app/demo_samples.yaml
```

### Before you trust the numbers, measure your hardware

```bash
# run with nothing else on the GPU, or the timings are contaminated
python scripts/benchmark_gpu.py --video "path/to/clip.mp4" --cameras 3
```

This prints measured latency for both branches and the smallest
`anomaly_stride` your GPU can sustain. Put that number in your config, then
**verify it against reality**, because the projection is only an estimate:

```bash
curl -s localhost:8000/api/stats
```

`buffers.anomaly.dropped` should stay near zero and
`posetrack.continuity_break_ratio` low. If frames are being refused, raise
`anomaly_stride` by one and re-check. Why this matters is explained under
[The frame-rate constraint](#the-frame-rate-constraint).

### Measured on the development machine

RTX 3050 6GB laptop GPU, 1440×1080 @ 27 fps footage, `yolov8m-pose` @ 1280 in
FP16 with the anomaly classifier, single-scale CSRNet:

| Cameras | `density_fps` | `anomaly_stride` | Analysed per gate | Playback |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1.0 | 2 | 13.5 fps | 27 fps |
| 2 | 1.0 | 5 | 5.4 fps | 27 fps |
| 3 | 0.5 | **7** (shipped) | 3.9 fps | 27 fps |

The stride only sets how often boxes and labels refresh; playback always runs at
the footage's full frame rate. Verified live with three cameras: 27.0 fps read,
27.0 fps displayed and 27.0 distinct frames/s received by the browser on every
gate, 0 frames skipped, 0 frames refused, 97% of motion observations accepted.

### Choosing the detector: measured, not assumed

Recall on the **labelled anomalous people** in the Hajj test set (2,366 boxes
over 243 frames), FP16. IoU ≥ 0.3, because the hand-drawn labels are looser than
detector boxes — a correctly found seated person scored IoU 0.48.

| Detector | Seated | Distant (<80 px) | All labelled | ms/frame |
| :--- | :--- | :--- | :--- | :--- |
| pose-small @960 (previous) | 19% | 4% | 33% | 15 |
| pose-medium @960 | 16% | 4% | 30% | 26 |
| **pose-medium @1280 (shipped)** | **26%** | **8%** | **40%** | **41** |
| pose-large @1280 | 24% | 7% | 38% | 65 |
| pose-large @1440 | 32% | 16% | 45% | 80 |
| detection-large @1440 | 45% | 31% | 48% | 78 (no keypoints) |

What the measurements showed:

- **Resolution mattered more than model size.** Medium at 960 found *fewer*
  seated people than small at 960. Changing the model without raising the
  resolution would have made the edge cases worse.
- **Detection models find seated people better than pose models** (45% vs 32% at
  the same size and resolution), because pose models are trained only on people
  with visible keypoints. The classifier needs keypoints, so a detection model
  alone cannot replace the pose model.
- **Detection caps anomaly accuracy.** Even the best setup finds under half the
  labelled seated people; a person never detected can never be classified.

Further traps:

- **Lowering CSRNet resolution is the wrong lever.** Measured, 512×384 cost 23%
  of the crowd count against 10% for a single scale, and under-counting makes the
  system under-react. Drop scales, not pixels.
- **FP16 is free.** ~1.7× faster with identical detections.

---

## Architecture

```
            CameraRegistry          one read per camera, cameras added at runtime
                  |
      +-----------+-----------+     dual-rate fork of a single decode
      v                       v
 anomaly buffer          density buffer
 (every Nth frame)       (decimated to density_fps)
      |                       |
      v                       v
 PoseTrackWorker          CSRNetWorker
 in order, never skips    newest frame per camera
      |         |              |
      v         v              v
 AnomalyResult  MotionResult   CSRNetResult
      |         |              |
      |         +--> TemporalSynchronizer <--+     causality: motion <= density
      |                      |
      |                      v
      |                 FusionAdapter  ....... the only module importing both halves
      |                      |
      |                      v
      |                  CrowdState  ......... model-agnostic: no torch, no cv2
      |                      |
      |                      v
      |            DecisionEngine + VenueTopology
      |                      |
      +--------> RuleEngine <+
                      |
                      v
            AlertStore / DashboardState --> FastAPI + WebSocket + MJPEG
```

Imports only ever point downward. The layering is load-bearing, not decorative:

- **`decision/` and `venue/` import no model code at all** — not `torch`, not
  `cv2`, not `ultralytics`. They are reasoning over numbers, and they are tested
  without a GPU. [`tests/test_adapters.py`](tests/test_adapters.py) asserts this
  with an AST scan, so the invariant fails loudly rather than eroding.
- **`adapters/fusion_adapter.py` is the single boundary** where model result
  types meet decision types. Swapping CSRNet or the pose model is a change to
  that one file.

### One model pass, two consumers

The pose pass yields box centres *and* 17 keypoints per person. Crowd motion
needs the former; anomaly features need the latter. Running one
`yolov8m-pose` per camera and deriving both is what keeps a 6 GB GPU viable
alongside CSRNet — the earlier prototypes ran two separate YOLO models over the
same frames.

A separate model instance is held per camera, because Ultralytics stores
ByteTrack state inside the model object when `persist=True`; sharing one would
let tracks from different scenes contaminate each other.

### Anomalies are a separate channel

Anomalies deliberately do **not** enter `CrowdState` or routing. Congestion
routing stays a pure function of density, flow and capacity; per-person
behaviour travels to `alerts/` on its own path.

The reason is that the two call for different responses. A congestion alert is
answered by moving crowd flow. An anomaly alert is answered by sending staff to
one person. Folding a collapsed individual into a congestion score would blur
both. A gate can raise one of each simultaneously, and the dashboard shows both.

---

## Real-time playback

Playback is deliberately separate from analysis:

```
producer  ──every frame, stamped t0 + k/fps──>  display buffer ──> renderer ──> /stream
    │                                                                 ▲
    └──1 frame in anomaly_stride──> pose model ──keyframes────────────┘
```

- **Paced by the footage.** Each frame read is stamped with its scheduled time
  on the session clock. The renderer shows whichever frame is due, so a fast GPU
  cannot make playback run ahead and a slow one cannot make it crawl.
- **All gates in lockstep.** Start gives every camera the same `t0`, so frame
  *k* of every gate falls due at the same instant. Measured: three producers
  released frame 0 within 30 ms of each other.
- **Boxes aligned, not trailing.** Playback runs `display.delay_s` (0.5 s) behind
  live, so the analysis on either side of each displayed frame is usually done.
  Boxes are interpolated between those keyframes and glide with the video.
  Labels come from the earlier keyframe, so an alarm never shows before the
  analysis that raised it.
- **Honest when analysis lags.** A keyframe's boxes are held for at most one
  second. After that they are withdrawn instead of left frozen over a moving
  crowd. Interpolation never crosses a clip loop, where tracking restarts.
- **All video over one WebSocket.** Every gate's frames travel over a single
  `/ws/video` socket as tagged binary messages. The browser decodes them off the
  main thread and draws them on canvases, keeping only the newest frame per gate,
  so a slow decode skips stale frames instead of lagging. Measured: 27.1 frames/s
  drawn on every gate.

  This replaced one never-ending MJPEG request per gate, which broke the app:
  browsers allow only **six HTTP connections per server**. Video panes, plus
  replaced panes the browser had not yet released, used up every slot, so Start
  and every other request queued inside the browser and never reached the
  server. The dashboard looked frozen on "READY" while the server sat idle. With
  six or more gates holding footage it would have failed outright. WebSockets
  are outside that limit: the dashboard now holds two sockets, telemetry and
  video, whatever the number of gates.
- **One render thread per camera, boxes drawn in batches.** Drawing a frame
  costs ~3.8 ms on an idle machine, but every OpenCV call releases and re-takes
  Python's GIL. A single shared render thread making ~50 calls per frame slowed
  to ~15 ms per frame once a browser was playing the streams, dropped playback
  to 10 fps, and starved the model threads. With per-camera threads and one draw
  call per colour group, measured with a browser drawing all three streams:
  26.7 fps playback and 11.4 frames/s analysed.
- **Analysis can't drift behind.** The pose worker skips any frame that has
  waited longer than 80% of the display delay, so the results it does produce
  are always current enough to line up with the video. Skips count as
  continuity breaks over a rolling 10 s window and raise `TRACKING_DEGRADED`.

Display frames are scaled to `display.width` (960) before overlay and encoding.
Click a gate's video to enlarge it to full width.

The page and its assets are served with `Cache-Control: no-cache`, and asset
URLs carry a version tag. Without that, a browser can keep running an old
`app.js` against a new page after an update, and buttons that the old script
never wired up silently do nothing.

### Stride compensation

The anomaly constants are defined per frame at training rate. At
`anomaly_stride: 7` the tracker rescales every frame-counted window to span the
same time as at training: heading window 15 → 2, position history 60 → 9,
minimum track age 10 → 3, confirmation run 5 → 2. Pixel thresholds go up 7×.
The scaling is taken from each frame's own stride.

*Fixed bug:* until this release, `scaled_for_stride` was defined but never
called. At stride 7 the heading window covered 3.9 s instead of 0.56 s, and the
sitting veto was 7× too strict.

## The frame-rate constraint

This is the single most important thing to understand before tuning anything.

The anomaly classifier's features are **per-frame pixel quantities** —
`displacement`, `straightness`, `cos_sim`, a `direction_window` of 15 frames, a
`min_speed` of 2.0 px/frame. It was trained on *consecutive* frames at native
video rate. CSRNet, by contrast, only needs a few samples per second: crowd
density does not change meaningfully between adjacent frames.

So the two branches want different frame rates, and the ingestion layer gives
them exactly that. One decode per camera forks into two independently decimated
paths, sharing a single immutable `FramePacket` and a single `frame_id` clock
(which is what makes the temporal synchroniser's comparison meaningful).

**`anomaly_stride: 1` is the correct setting.** It feeds the classifier the rate
it was trained on. If your GPU cannot sustain it, raising the stride does not
produce an error — it produces quietly worse predictions, because every motion
feature is then computed over a longer interval than the model ever saw.
`AnomalyConfig.scaled_for_stride` compensates for the linear part of that
(distances scale up, windows scale down), but path curvature is genuinely lost
when frames are skipped, so `straightness` and `cos_sim` do not rescale exactly.

The system is explicit about this rather than silent:

- `scripts/benchmark_gpu.py` measures your hardware and recommends a stride.
- The startup banner warns whenever the stride exceeds 1.
- The anomaly buffer uses a **newest-drop** policy, so a saturated pose branch
  refuses new frames instead of tearing holes in the middle of a trajectory.
- Every `AnomalyResult` carries `continuity_ok` and `frames_skipped`, the
  dashboard shows "frame gaps", and the `TRACKING_DEGRADED` alert fires when the
  break ratio passes 20%.

If you need to get back to stride 1, in rough order of value: use
`yolov8s-pose.pt` (~2x faster), lower `imgsz` to 640, lower `density_fps`, or run
fewer cameras per GPU.

---

## Tracking safeguards

In dense crowds the dominant failure mode is **ID theft through occlusion**: a
pedestrian walks in front of a seated person, inherits their track, and carries
the `SITTING` label across the scene as a false alarm. Three independent guards,
carried over from the original detector and now unit-tested:

| Guard | What it does |
| :--- | :--- |
| **Lost-track buffer** | A vanished track is retained for 30 frames and re-stitched on reappearance by IoU, so brief occlusion does not fragment identity. |
| **Stationary guard** | A *mostly stationary* lost track — a candidate seated person — may only be re-adopted by a detection whose centre is within 30 px, however well the boxes overlap. |
| **Speed veto** | Any track labelled `SITTING` while its recent step speed exceeds 3 px/frame is reset to `NORMAL` outright. A person who is moving is not sitting, whatever the model says. |

Label smoothing is deliberately asymmetric: an anomaly must repeat for 5
consecutive predictions to be confirmed, but a return to `NORMAL` takes effect
immediately. Raising an alarm slowly is acceptable; clearing a stale one slowly
is not.

---

## Gates without cameras

A gate with no camera is still part of the venue. Its capacity is known even
though its occupancy is not, so it stays in the topology and remains a legal
diversion target — which matters, because the venue you most need to divert into
is often the one you are not watching.

Such a gate is handled honestly rather than optimistically:

- it is assigned a neutral, deliberately **non-zero** assumed occupancy
  (`assumed_unmonitored_utilization`, default 0.3). Assuming an unseen gate is
  empty would make it the most attractive target in the venue;
- both `has_density` and `has_motion` are false, which degrades confidence;
- its routing score carries an `unmonitored_target_penalty`, so an observed gate
  of equal standing is preferred;
- the decision reason says so in words, and the dashboard marks it `NO CAMERA`;
- it cannot be a diversion *source*: congestion you cannot see is congestion you
  cannot detect, and the engine refuses to act on low confidence.

Set `allow_unmonitored_targets: false` to require live observation of every
target.

---

## Safety behaviour

The decision engine refuses to act, with an explanation, rather than guessing:

| Condition | Action | Why |
| :--- | :--- | :--- |
| Observation older than `max_observation_age_s` | `MONITOR` | A diversion based on stale data can send people into a gate that has since filled. |
| Confidence below `min_confidence` | `MONITOR` | Half an observation is not grounds for moving a crowd. |
| Congested, but every reachable gate is full | `RESTRICT_INFLOW` | There is nowhere to divert to; hold arrivals upstream instead of redistributing a problem. |
| Motion unavailable or stale | velocity term dropped | Substituting zero would invent congestion; substituting maximum would invent free flow. The remaining weights are renormalised. |

Missing CSRNet weights are a **startup failure**, not a warning: an untrained
density head produces a plausible-looking heat map and a number that means
nothing. A missing anomaly classifier degrades gracefully instead, since pose
tracking and the geometric counter-flow rule still work without it.

---

## Alert rules

Congestion rules are mutually exclusive per gate and evaluated most-severe
first — an operator facing a crush does not also need to be told the gate is
busy. Anomaly rules are evaluated independently.

| Rule | Severity | Fires when |
| :--- | :--- | :--- |
| `CRUSH_RISK` | EMERGENCY | High density with near-zero movement |
| `GATE_OVERLOAD` | EMERGENCY | Count exceeds stated capacity |
| `ALL_GATES_FULL` | EMERGENCY | Congested with no viable diversion target |
| `PERSON_DOWN` | EMERGENCY / WARNING | Someone on the ground; emergency only when the surrounding crowd is moving or congested |
| `COUNTERFLOW_DETECTED` | ALERT / WARNING | 2+ people against the flow, escalating at 4+ |
| `ANOMALY_CLUSTER` | ALERT | 25%+ of tracked people behaving abnormally |
| `BOTTLENECK_FORMING` | ALERT | High congestion and still rising |
| `DIVERT_IN_PROGRESS` | ALERT | A diversion is active, with staffing instructions |
| `SLOW_FLOW_WARNING` | WARNING | Speed falling at steady density |
| `MONITOR_SURGE` | WARNING | On the watchlist and inflow growing |
| `IMBALANCED_LOAD` | WARNING | One gate carrying 2.5x the venue average |
| `TRACKING_DEGRADED` | WARNING | Pose branch dropping frames |
| `MOTION_UNAVAILABLE` | INFO | Congestion scored from density alone |
| `VELOCITY_UNCALIBRATED` | INFO | Velocity term pinned at its ceiling |
| `ALL_CLEAR` | INFO | Nothing else fired |

`PERSON_DOWN` escalating on context is deliberate: someone seated in a calm,
sparse area is noteworthy, while the same person in a moving flow is a trip
hazard and a stampede nucleus.

Alerts are deduplicated by `(rule_id, gate_id)`, so a condition lasting a minute
is one alert with an occurrence count rather than a hundred. Escalations are
logged as fresh incidents; conditions easing off update in place. Incidents
append to `logs/alerts.jsonl`.

---

## HTTP API

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Operator dashboard |
| `GET` | `/stream/{gate_id}` | MJPEG of one gate, for external viewers such as VLC; the dashboard does not use it |
| `GET` | `/api/status` | Full system snapshot |
| `GET` | `/api/alerts` | Active alerts, counts, incident log |
| `GET` | `/api/decisions` | Latest decision per gate, with candidate reasoning |
| `GET` | `/api/venue` | Topology, including unmonitored gates |
| `GET` | `/api/venue/config` | Editable venue parameters for the setup screen |
| `POST` | `/api/venue/configure` | Rebuild the venue with a different gate count |
| `GET` | `/api/stats` | Buffers, worker latency, synchroniser health |
| `GET` | `/api/health` | Liveness and readiness |
| `GET` | `/api/cameras` | Registered cameras and their gates |
| `POST` | `/api/gates/{gate_id}/upload` | Stage footage on a gate (setup only) |
| `POST` | `/api/cameras/{id}/reassign` | Move a staged camera to another gate (setup only) |
| `DELETE` | `/api/cameras/{id}` | Remove staged footage (setup only) |
| `GET` | `/api/session` | Setup or running, and how much footage is staged |
| `POST` | `/api/session/start` | Start every staged gate at the same instant |
| `POST` | `/api/session/stop` | Stop monitoring and return to setup |
| `WS` | `/ws` | Full snapshot pushed twice a second |
| `WS` | `/ws/video` | Every gate's frames as binary `[len][gate_id][jpeg]` messages |

If the server ever dies without a Python error, look in `logs/crash.log`: a
native crash in OpenCV, PyTorch or the CUDA driver writes every thread's stack
there.

Setup endpoints return **409** while monitoring runs. Per-camera pause and
resume were removed: pausing one gate would break the shared session clock.

Uploads are streamed to disk in chunks with the size cap enforced as they
arrive, filenames are reduced to a safe character set, and extensions are
allow-listed.

---

## Layout

```
crowdguard/
├── config/
│   ├── app/default.yaml           no cameras; upload to bind gates
│   ├── app/demo_samples.yaml      three gates bound to sibling sample clips
│   └── venue/circular_venue.yaml  starting layout; gate count is editable live
├── assets/
│   ├── best_model.pkl             trained CatBoost anomaly classifier
│   └── yolov8m-pose.pt            pose weights
├── checkpoints/PartAmodel_best.pth  CSRNet ShanghaiTech-A
├── crowdguard/
│   ├── config/                    pydantic schemas, YAML + .env loading
│   ├── utils/                     geometry, logging, exceptions
│   ├── ingestion/                 sources, dual-rate producer, camera registry
│   ├── models/
│   │   ├── contracts.py           CSRNetResult, MotionResult, AnomalyResult
│   │   ├── csrnet/                architecture, weights, multi-scale inference
│   │   └── posetrack/             features, tracker, classifier, inference
│   ├── adapters/                  synchroniser + the fusion boundary
│   ├── decision/                  contracts, fusion, routing, trend, engine
│   ├── venue/                     gate graph, distances, perimeter routes, builder
│   ├── alerts/                    rules, contracts, dedup store
│   ├── pipeline/                  workers and the orchestrator
│   └── web/                       state, overlays, FastAPI app
├── web-ui/                        dashboard: React + TypeScript + Vite (served at /)
├── frontend/                      original dashboard, no build step (served at /classic)
├── docs/                          research notes for the paper
├── scripts/
│   ├── run_server.py              entry point
│   └── benchmark_gpu.py           measure throughput, recommend a stride
├── crowdguard/simulation/         closed-loop crowd model + policies for evaluating decisions
├── reports/decision_eval/         evaluation results (tables, CSVs, figures)
└── tests/                         ~920 tests, no GPU required
```

---

## Tests

```bash
python -m pytest tests -q
```

954 tests (plus 4 expected failures that document known engine weaknesses), ~20 seconds, no GPU and no model weights needed — a direct consequence
of keeping the decision layer model-agnostic and the tracker free of inference.
They cover the venue graph and perimeter routing, congestion scoring and the
missing-motion path, every engine safety refusal, hysteresis, routing with and
without cameras, the 10-D feature vector and pose geometry, the ID-theft guards
and speed veto, synchroniser causality, every alert rule, dedup and escalation,
buffer drop policies, the dual-rate producer, and the layering invariant itself.
The decision engine also has a scenario suite, 600 random-venue invariant checks
and a tested closed-loop simulator; see `docs/12_congestion_control_evaluation.md`.

```bash
python scripts/evaluate_decisions.py          # policy evaluation, CPU only, ~16 min on 6 cores
```

---

## Configuration reference

Everything is in `config/app/default.yaml`, overridable by environment variable
(see `.env.example`). The settings worth knowing:

| Setting | Default | Notes |
| :--- | :--- | :--- |
| `ingestion.anomaly_stride` | `1` | Pose decimation. **Leave at 1 unless the benchmark says otherwise.** |
| `ingestion.density_fps` | `5` | CSRNet sampling rate per camera |
| `ingestion.loop_sources` | `true` | Restart short clips so demo gates stay live |
| `models.posetrack.model_path` | `yolov8m-pose.pt` | Must be a `*-pose` model; keypoints are required |
| `models.posetrack.imgsz` | `960` | Drop to 640 for roughly 2x throughput |
| `models.anomaly.*` | — | Training-time constants. Changing them shifts the classifier's input distribution silently. |

`DecisionConfig` holds the congestion weights, routing weights, hysteresis and
viability thresholds. The congestion formula is a transparent, tunable
heuristic — it has **not** been validated against ground-truth crush incidents,
and should be calibrated per venue before being relied on operationally.

### Two things you must calibrate

Both of these are silent when wrong, which is why each has an advisory attached.

**`congestion_weights.max_expected_velocity`** (default 75 px/s) is the speed at
which a crowd counts as freely flowing. It is a *pixel* rate, so it scales with
resolution, zoom and camera distance, and must be set per camera. The upstream
default of 15 px/s was measured as badly wrong for 1440×1080 footage, where a
flowing crowd runs at 70–78 px/s: the inverse-velocity term sat pinned at zero,
every gate read LOW, and a stalling crowd would not have registered until it was
already dense. The `VELOCITY_UNCALIBRATED` advisory fires when the term
saturates. Set it to roughly what your camera shows for a crowd moving freely.

**Gate capacities** in `config/venue/circular_venue.yaml` are placeholders.
`density_score = crowd_count / capacity` drives every congestion threshold, and
`crowd_count` depends on the CSRNet scale set in use — single-scale under-counts
by about 10%. Capacities must be set from real observed occupancy, against
whichever scale set you run, before operational use.

---

## Credits

- Density estimation: [CSRNet](https://arxiv.org/abs/1802.10062) (Li et al., CVPR 2018), ShanghaiTech Part-A weights
- Detection, pose and tracking: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) + [ByteTrack](https://github.com/ifzhang/ByteTrack)
- Anomaly classifier: [CatBoost](https://catboost.ai/), trained on Hajj crowd anomaly annotations
