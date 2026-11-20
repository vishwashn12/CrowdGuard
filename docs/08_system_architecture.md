# 08: System architecture

## Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.11 |
| Vision / ML | PyTorch 2.5.1 (CUDA 12.1), Ultralytics 8.4 (YOLOv8-pose + ByteTrack), OpenCV, NumPy, CatBoost |
| Configuration | Pydantic v2 schemas, YAML files, `.env` overrides |
| Server | FastAPI + Uvicorn (ASGI): REST, 2 WebSockets, upload handling |
| Frontend | plain HTML, CSS and JavaScript with no framework or build step; canvas video, SVG venue map |
| Tests | pytest, 269 tests, no GPU or weights needed |

The whole system is **one process**. Pipeline threads and the web server share one address space and one CUDA context.

## Package layout and the layering rule

```
crowdguard/
  config/      pydantic schemas + YAML/.env loading
  ingestion/   video/RTSP/webcam sources, dual-rate producer, camera registry, frame buffers
  models/      csrnet/ (density), posetrack/ (pose, tracker, features, classifier), result contracts
  adapters/    temporal synchroniser + fusion adapter      <- the ONLY bridge
  decision/    congestion fusion, routing, trend, engine  <- no torch / cv2 / ultralytics
  venue/       gate graph, distances, ring builder        <- no torch / cv2 / ultralytics
  alerts/      rules, alert contracts, de-dup store
  display/     playback timeline + per-camera renderer
  pipeline/    workers + orchestrator (session state machine)
  web/         dashboard state, overlays, FastAPI app
```

**Invariant:** imports only point downward. `decision/` and `venue/` import no model code, and `adapters/fusion_adapter.py` is the only module that sees both model result types and decision types. An AST-scanning unit test (`tests/test_adapters.py`) enforces this, so the rule cannot erode silently.

Why it matters:
- the decision logic can be tested without a GPU;
- replacing CSRNet or the pose model touches one file;
- the decision layer can be presented in the paper as an independent, model-agnostic component.

## Data flow

```
             CameraRegistry  (one decode per camera; cameras added at runtime)
                   |
     +-------------+--------------+----------------+
     v             v              v                |
 display sink   anomaly buffer  density buffer     |   one FramePacket, one frame_id clock
 (every frame)  (1 in stride)   (~density_fps)     |
     |             |              |                |
     |        PoseTrackWorker   CSRNetWorker       |
     |         |        |          |               |
     |   AnomalyResult MotionResult CSRNetResult   |
     |         |        +--> TemporalSynchronizer <+   causality: motion frame_id <= density frame_id
     |         |                   |
     |         |              FusionAdapter --> CrowdState (model-free)
     |         |                   |
     |         |          DecisionEngine + VenueTopology   (2 Hz)
     |         |                   |
     |         +-------> RuleEngine <+--> AlertStore (de-dup, JSONL log)
     |                         |
     v                         v
 PlaybackRenderer -----> DashboardState --> /ws (telemetry, 2 Hz) + /ws/video (frames)
```

## Ingestion: dual rate from a single decode

- **One read per camera**, wrapped in an immutable `FramePacket` (camera id, frame id, source index, video time, wall time, loop index, anomaly stride, density stride).
- **Fan-out:**
  - display: every frame;
  - anomaly: every `anomaly_stride`-th frame. The buffer uses a **newest-drop** policy, so a saturated pose worker refuses new frames instead of tearing holes in the middle of a trajectory;
  - density: decimated to `density_fps`, drained to the newest frame.
- **Shared frame-id clock.** It is what makes the synchroniser's comparison meaningful.
- **Stride-aware tracking.** Each packet carries its stride, so the tracker rescales its constants ([02](02_feature_engineering.md)).

## Session clock and lockstep playback

- **Two states, setup → running.** In *setup*, footage is uploaded and assigned to gates; each gate shows a dimmed first-frame preview. Setup endpoints return HTTP 409 while running.
- **Start** resets all state, then sets $t_0 = \text{now} + 1\text{ s}$ and starts every producer on that same $t_0$. Producer frame $k$ is released at $t_0 + k/f_{\text{source}}$, so every gate plays in lockstep. Measured: three producers released frame 0 within 30 ms of each other.
- **Pacing.** Pacing applies to every frame read, not just emitted frames. If a producer falls more than 1 s behind schedule, it resynchronises instead of bursting.
- **Clips loop.** Each loop increments `loop_index` and resets ByteTrack.

## Playback decoupled from analysis

The display runs **0.5 s behind live** (`display.delay_s`), so the analysis keyframes on either side of each displayed frame are usually finished.

- **Interpolation.** For display frame $k$ between analysis keyframes $a \le k \le b$, each person's box is linearly interpolated between their boxes at $a$ and $b$. **Labels come from the earlier keyframe**, so an alarm never appears before the analysis that raised it.
- **Holding.** Boxes are held for at most **1 s** without a newer keyframe, then withdrawn rather than left frozen over a moving crowd.
- **Loop boundaries.** Interpolation never crosses a loop boundary.
- **Bounded analysis latency.** The pose worker skips any frame that waited longer than $\max(0.25, 0.8 \times \text{delay})$ = 0.4 s. Skips count as continuity breaks over a rolling 10 s window, and that ratio drives `TRACKING_DEGRADED`.
- **One render thread per camera**, with overlays drawn in batches (one `cv2.polylines` call per colour group) to avoid contention on Python's GIL ([09](09_runtime_performance.md)).
- **Encoding.** Display frames are scaled to 960 px wide and JPEG-encoded at quality 75.

## Web delivery

| Channel | Content |
| :--- | :--- |
| `GET /` | dashboard. Assets are served with `Cache-Control: no-cache` and `?v=<mtime>` version tags, so a stale `app.js` cannot run against a new page. |
| `WS /ws` | full JSON snapshot twice a second: session, gates, decisions, alerts, stats |
| `WS /ws/video` | every gate's frames as binary messages `[1-byte tag length][gate id][JPEG]` |
| REST | session start/stop, venue configure (gate count), upload / reassign / delete footage, status, alerts, decisions, stats, health |
| `GET /stream/{gate}` | MJPEG, kept for external viewers only |

**Why video goes over one WebSocket.** The first design used one never-ending MJPEG HTTP request per gate. Browsers allow only **6 HTTP/1.1 connections per host**. Once video panes (plus replaced panes not yet released) used all six, every other request, including *Start*, queued inside the browser and never reached the server. The UI looked frozen on "READY" while the server sat idle; we confirmed this with 7 established sockets and a profiler showing an idle server. WebSockets do not count against that limit. The dashboard now holds exactly **two sockets regardless of gate count**. The browser decodes frames off the main thread (`createImageBitmap`) and keeps only the newest pending frame per gate, so a slow decode skips stale frames instead of lagging.

## Frontend

- **Two tabs.**
  - *Monitor*: a gate grid with canvas video, congestion level, count, action, anomaly badges and alerts; click a video to enlarge it.
  - *Setup*: (1) choose the number of gates, 2 to 24; (2) upload a video per gate; (3) start.
- **Rendered from server state.** The whole UI is built from the `/ws` snapshot, so the gate count and layout always match the server's venue.
- **Session control.** A session chip shows the state, next to a Start/Stop button. The Start request has a 20 s timeout with an error message, so it can never hang silently.

## Robustness features worth mentioning

- **Missing models.** Missing CSRNet weights stop startup, because an untrained density head gives plausible-looking nonsense. A missing classifier degrades to the geometric rule.
- **Uploads.** Uploads are streamed to disk in chunks with a size cap, filenames are sanitised, and extensions are allow-listed.
- **Credentials.** RTSP credentials are redacted from logs.
- **Native crashes.** `faulthandler` writes every thread's stack to `logs/crash.log` on a crash in OpenCV, PyTorch or CUDA.
