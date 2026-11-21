"""FastAPI surface: MJPEG panes, telemetry, WebSocket push, and uploads.

The app is created by :func:`create_app` against a live
:class:`~crowdguard.pipeline.orchestrator.Orchestrator`, rather than importing a
module-level singleton. That keeps the server testable and makes the dependency
on the running pipeline explicit.

Uploads are the operator's way of putting footage on a gate: a file is accepted,
written to disk under that gate, registered as a camera, and the producer starts
immediately, with no restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crowdguard.config.schemas import CameraConfig
from crowdguard.pipeline.orchestrator import Orchestrator
from crowdguard.utils.calibration import CalibrationError, GroundCalibration
from crowdguard.utils.exceptions import CameraRegistryError, SessionError
from crowdguard.venue.builder import MAX_GATES, MIN_GATES
from crowdguard.web.overlay import placeholder_jpeg

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
# The React dashboard (web-ui/). `npm run build` writes it here; when present it
# is served at "/", and the original page moves to "/classic".
WEB_UI_DIST = Path(__file__).resolve().parents[2] / "web-ui" / "dist"

# MJPEG pane refresh rate and WebSocket telemetry rate.
# How often an idle stream (setup preview, or between frames) re-sends its
# current image. Browsers drop an MJPEG connection that goes silent.
_STREAM_KEEPALIVE_S = 0.5
# Polling interval for new frames. Well under one frame at 30 fps.
_STREAM_POLL_S = 0.005
# A video-socket client is re-sent an unchanged frame this often, so panes the
# page rebuilds are filled in again without waiting for the next new frame.
_VIDEO_RESEND_S = 1.0
_TELEMETRY_INTERVAL_S = 0.5

# Upload filenames are operator-supplied, so only a conservative character set
# survives into a path we write to.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class CalibrationRequest(BaseModel):
    """Four ground points (normalised image coordinates, in order around a
    rectangle on the floor) and that rectangle's real size."""

    points: list[list[float]] = Field(..., min_length=4, max_length=4)
    width_m: float = Field(..., gt=0)
    length_m: float = Field(..., gt=0)


class VenueRequest(BaseModel):
    """Payload for rebuilding the venue from the setup screen."""

    gate_count: int = Field(..., ge=MIN_GATES, le=MAX_GATES)
    default_capacity: int | None = Field(default=None, gt=0)
    radius_m: float | None = Field(default=None, gt=0)
    capacities: dict[str, int] | None = None
    venue_name: str | None = Field(default=None, max_length=120)
    segment_distances: list[float] | None = Field(
        default=None,
        description="Walking distance in metres from each gate to the next: G1->G2, ..., Gn->G1",
    )


def _sanitize_filename(name: str) -> str:
    """Reduce an uploaded filename to a safe basename."""
    base = Path(name or "upload").name
    cleaned = _SAFE_NAME.sub("_", base).strip("._-")
    return cleaned or "upload"


def create_app(orchestrator: Orchestrator) -> FastAPI:
    """Build the FastAPI application bound to a running orchestrator."""
    app = FastAPI(
        title="CrowdGuard",
        description="Crowd management and anomaly detection for multi-gate venues",
        version="1.0.0",
    )

    state = orchestrator.state
    upload_config = orchestrator.config.uploads
    upload_root = Path(upload_config.directory)

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    web_ui_index = WEB_UI_DIST / "index.html"
    if (WEB_UI_DIST / "assets").is_dir():
        # Vite fingerprints every asset file name, so these can be cached hard;
        # only index.html needs revalidating.
        app.mount("/assets", StaticFiles(directory=str(WEB_UI_DIST / "assets")), name="assets")

    @app.middleware("http")
    async def no_stale_frontend(request, call_next):
        """Make browsers revalidate the page and its assets on every load.

        Without a Cache-Control header, Chrome and Edge apply heuristic caching
        and keep reusing an old app.js across ordinary reloads. A new page paired
        with old JavaScript fails silently: buttons the old script never wired up
        simply do nothing. ``no-cache`` still allows the cached copy, but only
        after a cheap ETag check confirms it is current.
        """
        response = await call_next(request)
        path = request.url.path
        if path in ("/", "/classic") or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        elif path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    # ==================================================================
    # Pages
    # ==================================================================

    def _asset_version(name: str) -> str:
        """Short version tag from an asset's modification time."""
        try:
            return str(int((FRONTEND_DIR / name).stat().st_mtime))
        except OSError:
            return "0"

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        if web_ui_index.exists():
            return HTMLResponse(web_ui_index.read_text(encoding="utf-8"))
        return await classic_index()

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        icon = WEB_UI_DIST / "favicon.svg"
        if not icon.exists():
            raise HTTPException(status_code=404)
        return FileResponse(icon, media_type="image/svg+xml")

    @app.get("/classic", response_class=HTMLResponse)
    async def classic_index() -> HTMLResponse:
        """The original vanilla-JS dashboard, kept as a fallback."""
        index_file = FRONTEND_DIR / "index.html"
        if not index_file.exists():
            return HTMLResponse(
                "<h1>CrowdGuard</h1><p>Frontend assets are missing from "
                f"{FRONTEND_DIR}.</p>",
                status_code=200,
            )
        # Version the asset URLs, so a changed script or stylesheet is a new URL
        # and can never be served from a stale cache entry.
        html = index_file.read_text(encoding="utf-8")
        for asset in ("app.js", "style.css"):
            html = html.replace(f"/static/{asset}\"", f"/static/{asset}?v={_asset_version(asset)}\"")
        return HTMLResponse(html)

    # ==================================================================
    # Video panes
    # ==================================================================

    async def _mjpeg_stream(gate_id: str):
        """Yield each new frame of a gate's video pane exactly once.

        Frames are sent as the playback renderer produces them, at the footage's
        own frame rate, rather than resampled onto a fixed timer. A fixed timer
        would duplicate some frames and drop others, which reads as stutter. When
        nothing new arrives - before Start, or if playback stalls - the current
        image is re-sent occasionally so the browser keeps the connection open.
        """
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        idle = placeholder_jpeg(message=f"{gate_id}: awaiting feed")
        last_seq = None
        last_sent = 0.0

        while True:
            seq, data = state.get_frame(gate_id)
            now = time.monotonic()
            if seq != last_seq or now - last_sent >= _STREAM_KEEPALIVE_S:
                yield boundary + (data or idle) + b"\r\n"
                last_seq = seq
                last_sent = now
            await asyncio.sleep(_STREAM_POLL_S)

    @app.get("/stream/{gate_id}")
    async def video_stream(gate_id: str) -> StreamingResponse:
        """Annotated video for a gate: boxes, trails and crowd-flow arrow."""
        return StreamingResponse(
            _mjpeg_stream(gate_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ==================================================================
    # Telemetry
    # ==================================================================

    @app.get("/api/status")
    async def status() -> dict:
        """Full system snapshot."""
        return state.snapshot()

    @app.get("/api/alerts")
    async def alerts() -> dict:
        """Active alerts and the recent incident log."""
        return {
            "active": [a.to_dict() for a in orchestrator.alerts.active()],
            "counts": orchestrator.alerts.counts_by_severity(),
            "incidents": [a.to_dict() for a in orchestrator.alerts.history(limit=50)],
            "worst": (
                orchestrator.alerts.worst_severity().value
                if orchestrator.alerts.worst_severity() else None
            ),
        }

    @app.get("/api/venue")
    async def venue() -> dict:
        """Venue topology, including which gates are unmonitored."""
        return orchestrator.venue.summary()

    @app.get("/api/venue/config")
    async def venue_config() -> dict:
        """Editable venue parameters, for the setup screen."""
        venue = orchestrator.venue
        return {
            "gate_count": venue.gate_count,
            "radius_m": venue.radius_m,
            "venue_name": venue.venue_name,
            "min_gates": MIN_GATES,
            "max_gates": MAX_GATES,
            "segments": venue.summary()["segments"],
            "perimeter_m": venue.summary()["perimeter_m"],
            "gates": [
                {
                    "gate_id": gate_id,
                    "name": venue.get_gate(gate_id).name,
                    "capacity": venue.get_gate_capacity(gate_id),
                    "camera_id": venue.camera_for_gate(gate_id),
                    "is_monitored": venue.is_monitored(gate_id),
                    "calibration": (
                        orchestrator.calibration_for_gate(gate_id).to_dict()
                        if orchestrator.calibration_for_gate(gate_id) else None
                    ),
                    "effective_capacity": orchestrator.effective_capacity(gate_id),
                }
                for gate_id in venue.gate_ids
            ],
        }

    @app.post("/api/venue/configure")
    async def configure_venue(request: VenueRequest) -> dict:
        """Rebuild the venue with a different gate count.

        Growing the venue keeps every existing gate and its camera and adds
        empty ones. Shrinking it stops any camera bound to a gate that no longer
        exists. Trend, hysteresis and alert state are reset, since they describe
        the previous layout.
        """
        try:
            # Blocking: stops producer threads and rebuilds the graph.
            result = await asyncio.to_thread(
                orchestrator.reconfigure_venue,
                request.gate_count,
                request.default_capacity,
                request.radius_m,
                request.capacities,
                request.venue_name,
                request.segment_distances,
            )
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "reconfigured", **result}

    # ==================================================================
    # Session
    # ==================================================================

    @app.get("/api/session")
    async def session_status() -> dict:
        """Whether monitoring is running, and how much footage is staged."""
        return state.snapshot()["session"]

    @app.post("/api/session/start")
    async def session_start() -> dict:
        """Start every staged gate playing at the same instant."""
        try:
            return await asyncio.to_thread(orchestrator.start_session)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/session/stop")
    async def session_stop() -> dict:
        """Stop monitoring and return to setup. Staged footage is kept."""
        return await asyncio.to_thread(orchestrator.stop_session)

    @app.get("/api/decisions")
    async def decisions() -> dict:
        """Latest decision per gate, with full candidate reasoning."""
        return {
            gate_id: decision.to_dict()
            for gate_id, decision in orchestrator.latest_decisions().items()
        }

    @app.get("/api/stats")
    async def stats() -> dict:
        """Pipeline internals: buffers, worker latency, synchroniser health."""
        return orchestrator.stats()

    # ==================================================================
    # Camera and gate management
    # ==================================================================

    @app.get("/api/cameras")
    async def list_cameras() -> dict:
        """Registered cameras and the gates they serve."""
        result = {}
        for camera_id, camera_stats in orchestrator.cameras.stats().items():
            config = orchestrator.cameras.get_config(camera_id)
            result[camera_id] = {
                **camera_stats,
                "gate_id": orchestrator.venue.gate_for_camera(camera_id),
                "name": config.name if config else None,
                "source": Path(config.source).name if config and config.source_type == "video" else (
                    config.source_type if config else None
                ),
            }
        return {"cameras": result, "count": len(result)}

    @app.post("/api/gates/{gate_id}/upload")
    async def upload_footage(
        gate_id: str,
        file: UploadFile = File(...),
        camera_name: str | None = Form(default=None),
    ) -> JSONResponse:
        """Upload footage for a gate and stage it.

        The feed does not play until monitoring is started, so every gate begins
        together. Rejects uploads while monitoring runs, unknown gates,
        disallowed extensions and oversized files, then streams the upload to
        disk under ``uploads/<gate_id>/``.
        """
        if orchestrator.session_running:
            raise HTTPException(
                status_code=409,
                detail="Setup is locked while monitoring is running. Press Stop to change footage.",
            )

        if not orchestrator.venue.has_gate(gate_id):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown gate {gate_id!r}. This venue has: "
                    f"{', '.join(orchestrator.venue.gate_ids)}"
                ),
            )

        safe_name = _sanitize_filename(file.filename or "upload.mp4")
        suffix = Path(safe_name).suffix.lower()
        if suffix not in upload_config.allowed_suffixes:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type {suffix!r}. Allowed: "
                    f"{', '.join(upload_config.allowed_suffixes)}"
                ),
            )

        gate_dir = upload_root / gate_id
        try:
            gate_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not create upload directory: {exc}"
            ) from exc

        unique = uuid.uuid4().hex[:8]
        destination = gate_dir / f"{Path(safe_name).stem}_{unique}{suffix}"

        # Stream to disk in chunks and enforce the cap as we go, so an oversized
        # upload cannot exhaust memory or the disk before being rejected.
        written = 0
        chunk_size = 1024 * 1024
        try:
            with destination.open("wb") as out:
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > upload_config.max_bytes:
                        out.close()
                        destination.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"File exceeds the {upload_config.max_bytes / 1024 ** 3:.1f} GB limit"
                            ),
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
        finally:
            await file.close()

        if written == 0:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Replace whatever was previously on this gate.
        #
        # Camera lifecycle is blocking: removing one joins its producer thread
        # and adding one opens the file. Run both off the event loop, or every
        # MJPEG stream and the telemetry socket stall for the duration.
        existing_camera = orchestrator.venue.camera_for_gate(gate_id)
        try:
            if existing_camera is not None:
                await asyncio.to_thread(orchestrator.remove_camera, existing_camera)
        except SessionError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        camera_id = f"cam_{gate_id.lower()}_{unique}"
        camera = CameraConfig(
            camera_id=camera_id,
            name=camera_name or f"{gate_id} upload",
            source_type="video",
            source=str(destination),
            enabled=True,
            gate_id=gate_id,
        )

        try:
            await asyncio.to_thread(orchestrator.add_camera, camera, gate_id)
        except SessionError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CameraRegistryError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        producer = orchestrator.cameras.get_producer(camera_id)
        logger.info("Gate %s now served by uploaded file %s", gate_id, destination.name)

        return JSONResponse({
            "status": "ready",
            "gate_id": gate_id,
            "camera_id": camera_id,
            "filename": destination.name,
            "size_bytes": written,
            "source_fps": round(producer.source_fps, 2) if producer else None,
            "frame_count": producer.source.frame_count if producer else None,
            "anomaly_stride": producer.anomaly_stride if producer else None,
            "replaced_camera": existing_camera,
        })

    # ==================================================================
    # Ground calibration
    # ==================================================================

    @app.get("/api/gates/{gate_id}/still")
    async def gate_still(gate_id: str) -> Response:
        """First frame of a gate's footage, without overlays, for calibration."""
        if not orchestrator.venue.has_gate(gate_id):
            raise HTTPException(status_code=404, detail=f"Unknown gate {gate_id!r}")
        jpeg = await asyncio.to_thread(orchestrator.still_frame, gate_id)
        if jpeg is None:
            raise HTTPException(status_code=404, detail=f"{gate_id} has no footage")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.put("/api/gates/{gate_id}/calibration")
    async def set_calibration(gate_id: str, request: CalibrationRequest) -> dict:
        """Calibrate the camera on a gate from four ground points and a size."""
        if not orchestrator.venue.has_gate(gate_id):
            raise HTTPException(status_code=404, detail=f"Unknown gate {gate_id!r}")
        try:
            calibration = GroundCalibration(
                points=tuple((float(x), float(y)) for x, y in request.points),
                width_m=request.width_m,
                length_m=request.length_m,
            )
            saved = await asyncio.to_thread(orchestrator.set_calibration, gate_id, calibration)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (CalibrationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"gate_id": gate_id, "calibration": saved,
                "capacity": orchestrator.effective_capacity(gate_id)}

    @app.delete("/api/gates/{gate_id}/calibration")
    async def clear_calibration(gate_id: str) -> dict:
        if not orchestrator.venue.has_gate(gate_id):
            raise HTTPException(status_code=404, detail=f"Unknown gate {gate_id!r}")
        try:
            await asyncio.to_thread(orchestrator.set_calibration, gate_id, None)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"gate_id": gate_id, "calibration": None,
                "capacity": orchestrator.effective_capacity(gate_id)}

    @app.post("/api/cameras/{camera_id}/reassign")
    async def reassign_camera(camera_id: str, gate_id: str = Form(...)) -> dict:
        """Move a staged camera to a different gate."""
        try:
            await asyncio.to_thread(orchestrator.reassign_camera, camera_id, gate_id)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CameraRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "reassigned", "camera_id": camera_id, "gate_id": gate_id}

    @app.delete("/api/cameras/{camera_id}")
    async def remove_camera(camera_id: str) -> dict:
        """Remove a staged camera."""
        try:
            gate_id = await asyncio.to_thread(orchestrator.remove_camera, camera_id)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if gate_id is None:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id!r} is not registered")
        return {"status": "removed", "camera_id": camera_id, "gate_id": gate_id}

    # ==================================================================
    # WebSocket
    # ==================================================================

    @app.websocket("/ws/video")
    async def video_socket(socket: WebSocket) -> None:
        """Every gate's video over one WebSocket, as tagged binary frames.

        The dashboard used to give each gate its own never-ending MJPEG request.
        Browsers allow six HTTP connections per server, so a handful of video
        panes - plus any panes the page had replaced but not yet released - used
        up every slot, and ordinary requests such as Start queued in the browser
        and never reached the server at all. WebSockets sit outside that limit,
        and one socket carries any number of gates.

        Each message is ``[len(gate_id)][gate_id][jpeg]``. A gate's newest frame
        is sent whenever it changes, so a client that falls behind skips frames
        instead of building a backlog. An unchanged frame is re-sent every
        ``_VIDEO_RESEND_S`` so panes rebuilt by the page, or a fresh connection,
        are filled in even while a gate shows its still before Start.
        """
        await socket.accept()
        sent: dict[str, tuple[int, float]] = {}
        try:
            while True:
                now = time.monotonic()
                for gate_id, seq, jpeg in state.frames():
                    if not jpeg:
                        continue
                    last = sent.get(gate_id)
                    if last is None or last[0] != seq or now - last[1] >= _VIDEO_RESEND_S:
                        tag = gate_id.encode("ascii", "replace")[:255]
                        await socket.send_bytes(bytes([len(tag)]) + tag + jpeg)
                        sent[gate_id] = (seq, now)
                await asyncio.sleep(_STREAM_POLL_S)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.debug("Video socket closed: %s", exc)

    @app.websocket("/ws")
    async def telemetry_socket(socket: WebSocket) -> None:
        """Push the full snapshot to the dashboard on a fixed interval."""
        await socket.accept()
        try:
            while True:
                payload = json.dumps(state.snapshot(), default=str)
                await socket.send_text(payload)
                await asyncio.sleep(_TELEMETRY_INTERVAL_S)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.debug("Telemetry socket closed: %s", exc)

    # ==================================================================
    # Health
    # ==================================================================

    @app.get("/api/health")
    async def health() -> dict:
        """Liveness and a brief readiness summary."""
        snapshot = state.snapshot()
        monitored = sum(1 for g in snapshot["gates"].values() if g["is_monitored"])
        stale = sum(1 for g in snapshot["gates"].values() if g["is_stale"])
        return {
            "status": "ok" if orchestrator.is_running else "stopped",
            "uptime_s": snapshot["uptime_s"],
            "device": snapshot["device"],
            "gates_total": len(snapshot["gates"]),
            "gates_monitored": monitored,
            "gates_stale": stale,
            "cameras": orchestrator.cameras.count,
            "timestamp": time.time(),
        }

    return app
