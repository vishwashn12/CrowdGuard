"""System orchestrator: wires ingestion, models, decisions, alerting and playback.

Topology::

    CameraRegistry (N producers, one read per camera, shared session clock)
        |                    |                              |
        | every frame        | 1 frame in anomaly_stride    | 1 in density_stride
        v                    v                              v
    PlaybackRenderer     anomaly buffer                 density buffer
    (real-time display)      |                              |
        ^                    v                              v
        |               PoseTrackWorker                CSRNetWorker
        |                 |          |                      |
        +-- keyframes --- AnomalyResult  MotionResult   CSRNetResult
                              |          |                  |
                              |          +--> TemporalSynchronizer
                              |                      |
                              |               FusionAdapter -> CrowdState -> DecisionEngine
                              |                                                    |
                              +------------------> RuleEngine <--------------------+
                                                       |
                                                AlertStore / DashboardState

**Session.** Nothing plays until :meth:`start_session`. Uploading footage stages
a camera and shows a still of its first frame. Start releases every staged camera
on one shared clock, so all gates begin together, and setup is locked until
:meth:`stop_session` returns the system to setup.

**Playback is decoupled from analysis.** Every frame read goes to the renderer,
which shows it at its scheduled time on the session clock, a fixed delay behind
live, with boxes interpolated between analysis keyframes. Playback speed is set
by the footage; analysis speed only decides how often boxes refresh.

**Decisions** are evaluated on their own fixed cadence rather than on every frame
arrival, so every gate is judged against the same snapshot, including gates with
no camera at all.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from crowdguard.adapters.fusion_adapter import FusionAdapter
from crowdguard.adapters.synchronizer import TemporalSynchronizer
from crowdguard.alerts.commands import CommandBoard, present_alert
from crowdguard.alerts.contracts import Alert
from crowdguard.alerts.rules import GateView, generate_alerts
from crowdguard.alerts.store import AlertStore
from crowdguard.config.schemas import AppConfig, CameraConfig
from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import CrowdState, DecisionResult
from crowdguard.decision.engine import DecisionEngine
from crowdguard.decision.fusion import calibrated_capacity
from crowdguard.decision.trend import TrendTracker
from crowdguard.display.renderer import PlaybackRenderer
from crowdguard.ingestion.camera_registry import CameraRegistry
from crowdguard.ingestion.frame_buffer import FrameBuffer
from crowdguard.models.contracts import AnomalyResult, CSRNetResult, MotionResult
from crowdguard.models.manager import ModelManager
from crowdguard.pipeline.calibration_store import CalibrationStore
from crowdguard.pipeline.csrnet_worker import CSRNetWorker
from crowdguard.pipeline.posetrack_worker import PoseTrackWorker
from crowdguard.utils.calibration import GroundCalibration
from crowdguard.utils.exceptions import SessionError
from crowdguard.venue.builder import DEFAULT_CAPACITY, build_ring_topology
from crowdguard.venue.loader import load_venue_topology
from crowdguard.venue.topology import VenueTopology
from crowdguard.web.overlay import encode_jpeg, render_preview, scale_for_display
from crowdguard.web.state import DashboardState, dashboard_state

logger = logging.getLogger(__name__)

# Decision sweeps per second.
_EVALUATION_HZ = 2.0

# Frame 0 of every gate is scheduled this far after Start is pressed, so each
# source has time to open and decode before its first frame falls due.
_SESSION_LEAD_S = 1.0

_SETUP_LOCKED = "Setup is locked while monitoring is running. Press Stop first."


class Orchestrator:
    """Owns every component and the threads that connect them."""

    def __init__(
        self,
        config: AppConfig,
        venue: VenueTopology | None = None,
        decision_config: DecisionConfig | None = None,
        state: DashboardState | None = None,
        alert_log_path: str | Path | None = None,
        stale_frame_gap: int = 10,
    ) -> None:
        self.config = config
        self.venue = venue or load_venue_topology(config.venue_config)
        self.decision_config = decision_config or DecisionConfig()
        self.state = state or dashboard_state

        # Motion results arrive one per anomaly_stride frames, plus inference
        # latency. A fixed staleness limit narrower than that spacing discards
        # perfectly current motion as "stale" - at stride 7 a 10-frame limit
        # rejected 14% of valid observations. Allow two strides of headroom.
        self.stale_frame_gap = max(stale_frame_gap, 2 * config.ingestion.anomaly_stride)

        # --- buffers -------------------------------------------------
        # Density: keep the newest, drop the oldest. A stale count is useless.
        # Anomaly: refuse new frames when full rather than dropping from the
        # middle of the queue, because trajectory continuity is what the
        # features depend on.
        camera_headroom = max(3, len(config.cameras))
        self.density_buffer = FrameBuffer(
            capacity=config.ingestion.buffer_size * camera_headroom,
            drop_policy="oldest",
            name="density",
        )
        self.anomaly_buffer = FrameBuffer(
            capacity=config.ingestion.buffer_size * camera_headroom,
            drop_policy="newest",
            name="anomaly",
        )

        # --- models --------------------------------------------------
        self.models = ModelManager(models_config=config.models)

        # --- playback ------------------------------------------------
        self.renderer = PlaybackRenderer(
            config=config.display,
            publish=self._publish_frame,
            header=self._header_for,
        )

        # --- ingestion -----------------------------------------------
        self.cameras = CameraRegistry(
            config=config,
            venue=self.venue,
            density_buffer=self.density_buffer,
            anomaly_buffer=self.anomaly_buffer,
            display_sink=self.renderer.ingest_frame,
        )

        # --- decision half -------------------------------------------
        self.synchronizer = TemporalSynchronizer(stale_frame_gap=stale_frame_gap)
        self.adapter = FusionAdapter(self.decision_config)
        self.trends = TrendTracker(self.decision_config)
        self.engine = DecisionEngine(self.venue, self.decision_config)
        self.alerts = AlertStore(log_path=alert_log_path)
        self.commands = CommandBoard()
        self.calibrations = CalibrationStore(Path(config.uploads.directory) / "calibrations.json")

        # --- workers -------------------------------------------------
        self.csrnet_worker: CSRNetWorker | None = None
        self.posetrack_worker: PoseTrackWorker | None = None

        if self.models.csrnet_enabled:
            self.csrnet_worker = CSRNetWorker(
                input_queue=self.density_buffer,
                engine=self.models.csrnet,
                result_callback=self._on_density,
            )
        if self.models.posetrack_enabled:
            # A frame's analysis has to finish before that frame falls due for
            # display, or its boxes cannot be lined up with it. Frames that have
            # already waited most of the display delay are skipped, which keeps
            # analysis current under load instead of letting it drift behind.
            max_latency = max(0.25, 0.8 * config.display.delay_s)
            self.posetrack_worker = PoseTrackWorker(
                input_queue=self.anomaly_buffer,
                engine=self.models.posetrack,
                motion_callback=self._on_motion,
                anomaly_callback=self._on_anomaly,
                max_latency_s=max_latency,
            )

        # --- latest results, keyed by camera -------------------------
        self._latest_density: dict[str, CSRNetResult] = {}
        self._latest_anomaly: dict[str, AnomalyResult] = {}
        self._results_lock = threading.Lock()

        self._latest_states: dict[str, CrowdState] = {}
        self._latest_decisions: dict[str, DecisionResult] = {}

        self._evaluation_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

        # --- session -------------------------------------------------
        self._session_running = False
        self._session_started_at: float | None = None
        self._session_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the process-level pipeline is up (workers, renderer, loop)."""
        return self._running

    @property
    def session_running(self) -> bool:
        """Whether monitoring has been started and footage is playing."""
        return self._session_running

    def latest_states(self) -> dict[str, CrowdState]:
        with self._results_lock:
            return dict(self._latest_states)

    def latest_decisions(self) -> dict[str, DecisionResult]:
        with self._results_lock:
            return dict(self._latest_decisions)

    def _require_setup(self) -> None:
        if self._session_running:
            raise SessionError(_SETUP_LOCKED)

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def start(self, load_cameras: bool = True, autostart: bool = False) -> None:
        """Bring up workers, renderer and the decision loop, in setup mode.

        Cameras from the config are staged, not started. Monitoring begins on
        :meth:`start_session`, or immediately when ``autostart`` is set.
        """
        if self._running:
            return

        self._stop_event.clear()
        self._running = True

        self.state.pipeline_running = True
        self.state.start_time = time.time()
        self.state.device = self.models.device
        self.state.model_info = self.models.device_info
        self.state.set_session("setup")

        self._sync_gates_to_state()

        if self.csrnet_worker is not None:
            self.csrnet_worker.start()
        if self.posetrack_worker is not None:
            self.posetrack_worker.start()
        self.renderer.start()

        if load_cameras:
            staged = self.cameras.load_from_config(start=False)
            for camera_id in staged:
                self._stage_for_playback(camera_id)
            if staged:
                logger.info("Staged %d camera(s): %s", len(staged), ", ".join(staged))
            self._sync_gates_to_state()

        self._evaluation_thread = threading.Thread(
            target=self._evaluation_loop, name="DecisionLoop", daemon=True
        )
        self._evaluation_thread.start()

        logger.info(
            "Orchestrator up in setup mode | venue=%s gates=%d device=%s csrnet=%s pose=%s",
            self.venue.venue_name, self.venue.gate_count, self.models.device,
            self.models.csrnet_enabled, self.models.posetrack_enabled,
        )

        if autostart and self.cameras.count:
            self.start_session()

    def stop(self) -> None:
        """Shut everything down in reverse dependency order."""
        if not self._running:
            return

        logger.info("Stopping orchestrator...")
        if self._session_running:
            self.stop_session()

        self._stop_event.set()
        self._running = False
        self.state.pipeline_running = False

        self.cameras.stop_all()
        self.renderer.stop()

        if self._evaluation_thread is not None:
            self._evaluation_thread.join(timeout=3.0)
        if self.csrnet_worker is not None:
            self.csrnet_worker.stop()
        if self.posetrack_worker is not None:
            self.posetrack_worker.stop()

        self.density_buffer.close()
        self.anomaly_buffer.close()
        logger.info("Orchestrator stopped.")

    def __enter__(self) -> "Orchestrator":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Session: Start / Stop
    # ------------------------------------------------------------------

    def start_session(self) -> dict:
        """Start monitoring: every staged gate begins playing at the same instant.

        Clears everything derived from any previous session - tracks, trends,
        diversion hysteresis, alerts, counters - then releases every producer
        on one shared clock, so frame 0 of every gate falls due together.

        Raises:
            SessionError: If already running, or if no footage is staged.
        """
        with self._session_lock:
            if self._session_running:
                raise SessionError("Monitoring is already running.")
            camera_ids = self.cameras.camera_ids
            if not camera_ids:
                raise SessionError("Upload footage to at least one gate before starting.")

            self._reset_session_state(camera_ids)

            self.renderer.begin_session()
            t0 = time.monotonic() + _SESSION_LEAD_S
            self.cameras.start_all(t0=t0)

            self._session_running = True
            self._session_started_at = time.time() + _SESSION_LEAD_S
            self.state.set_session("running", self._session_started_at)

            gates = [self.venue.gate_for_camera(cid) for cid in camera_ids]
            logger.info(
                "Monitoring started: %d gate(s) playing in lockstep (%s)",
                len(camera_ids), ", ".join(g for g in gates if g),
            )
            return {
                "state": "running",
                "gates": [g for g in gates if g],
                "cameras": camera_ids,
                "started_at": self._session_started_at,
            }

    def stop_session(self) -> dict:
        """Stop monitoring and return to setup. Staged footage stays in place."""
        with self._session_lock:
            if not self._session_running:
                return {"state": "setup"}

            self._session_running = False
            self.renderer.end_session()
            self.cameras.stop_playback()

            # Drop frames still queued for the models; they belong to a session
            # that has ended. Results still in flight are ignored on arrival.
            self.density_buffer.clear()
            self.anomaly_buffer.clear()

            self._session_started_at = None
            self.state.set_session("setup")
            self.state.reset_gate_metrics()
            self.alerts.clear()
            self.commands.reset()
            self.state.update_alerts(alerts=[], counts={}, incidents=[])
            self.state.update_instructions([], [], [])

            for camera_id in self.cameras.camera_ids:
                self._publish_preview(camera_id)

            logger.info("Monitoring stopped; back in setup.")
            return {"state": "setup"}

    def _reset_session_state(self, camera_ids: list[str]) -> None:
        """Clear everything a previous session left behind."""
        self.density_buffer.clear()
        self.anomaly_buffer.clear()

        for camera_id in camera_ids:
            self.models.reset_tracking(camera_id)
            self.synchronizer.forget_camera(camera_id)

        with self._results_lock:
            self._latest_density.clear()
            self._latest_anomaly.clear()
            self._latest_states = {}
            self._latest_decisions = {}

        self.trends.reset()
        self.engine.reset_hysteresis()
        self.alerts.clear()
        self.commands.reset()
        self.state.update_instructions([], [], [])

        self.synchronizer.reset_counters()
        self.density_buffer.reset_counters()
        self.anomaly_buffer.reset_counters()
        if self.csrnet_worker is not None:
            self.csrnet_worker.reset_counters()
        if self.posetrack_worker is not None:
            self.posetrack_worker.reset_counters()

        self.state.reset_counters()
        self.state.reset_gate_metrics()

    # ------------------------------------------------------------------
    # Camera management (setup only)
    # ------------------------------------------------------------------

    def add_camera(self, camera: CameraConfig, gate_id: str | None = None) -> str:
        """Stage a camera on a gate. It plays when the session is started.

        Raises:
            SessionError: If a session is running.
        """
        with self._session_lock:
            self._require_setup()
            self.cameras.add_camera(camera, gate_id=gate_id, start=False)
            self._stage_for_playback(camera.camera_id)
            resolved_gate = self.venue.gate_for_camera(camera.camera_id)
            self._sync_gates_to_state()
            logger.info("Camera %s staged on %s", camera.camera_id, resolved_gate)
            return resolved_gate or ""

    def remove_camera(self, camera_id: str) -> str | None:
        """Remove a staged camera and everything tied to it.

        Raises:
            SessionError: If a session is running.
        """
        with self._session_lock:
            self._require_setup()
            return self._remove_camera_unlocked(camera_id)

    def _remove_camera_unlocked(self, camera_id: str) -> str | None:
        gate_id = self.cameras.remove_camera(camera_id)
        self.calibrations.delete(camera_id)

        self.renderer.unregister(camera_id)
        self.models.release_camera(camera_id)
        self.synchronizer.forget_camera(camera_id)

        with self._results_lock:
            self._latest_density.pop(camera_id, None)
            self._latest_anomaly.pop(camera_id, None)
            if gate_id:
                self._latest_states.pop(gate_id, None)
                self._latest_decisions.pop(gate_id, None)

        if gate_id:
            self.trends.reset(gate_id)
            self.engine.reset_hysteresis(gate_id)
            self.state.bind_camera(gate_id, None)

        self._sync_gates_to_state()
        return gate_id

    def reassign_camera(self, camera_id: str, gate_id: str) -> None:
        """Move a staged camera to a different gate.

        Raises:
            SessionError: If a session is running.
        """
        with self._session_lock:
            self._require_setup()
            previous_gate = self.venue.gate_for_camera(camera_id)
            self.cameras.reassign_camera(camera_id, gate_id)
            if previous_gate and previous_gate != gate_id:
                self.state.bind_camera(previous_gate, None)
            self._sync_gates_to_state()
            self._publish_preview(camera_id)

    def _stage_for_playback(self, camera_id: str) -> None:
        """Register a camera with the renderer and show its preview still."""
        producer = self.cameras.get_producer(camera_id)
        fps = producer.source_fps if producer is not None else 25.0
        self.renderer.register(camera_id, fps)
        self._apply_calibration(camera_id)
        self._publish_preview(camera_id)

    def _publish_preview(self, camera_id: str) -> None:
        """Show a dimmed still of a staged feed's first frame on its gate."""
        gate_id = self.venue.gate_for_camera(camera_id)
        camera = self.cameras.get_config(camera_id)
        if gate_id is None or camera is None:
            return

        frame = self._first_frame(camera)
        if frame is None:
            frame = np.full((540, 960, 3), 24, dtype=np.uint8)
        image, _ = scale_for_display(frame, self.config.display.width)
        self.state.update_frame(gate_id, encode_jpeg(render_preview(image, gate_id), 80))

    # ------------------------------------------------------------------
    # Ground calibration (setup only)
    # ------------------------------------------------------------------

    def _apply_calibration(self, camera_id: str) -> None:
        if self.models.posetrack_enabled:
            self.models.posetrack.set_ground_calibration(camera_id, self.calibrations.get(camera_id))

    def calibration_for_gate(self, gate_id: str) -> GroundCalibration | None:
        return self.calibrations.get(self.venue.camera_for_gate(gate_id))

    def effective_capacity(self, gate_id: str) -> int:
        """A calibrated gate's capacity follows from its zone; others use the venue value."""
        calibration = self.calibration_for_gate(gate_id)
        if calibration is not None:
            return calibrated_capacity(calibration.area_m2, self.decision_config)
        return self.venue.get_gate_capacity(gate_id)

    def set_calibration(self, gate_id: str, calibration: GroundCalibration | None) -> dict | None:
        """Calibrate (or, with None, uncalibrate) the camera on a gate.

        Raises:
            SessionError: If a session is running.
            ValueError: If the gate has no camera.
        """
        with self._session_lock:
            self._require_setup()
            camera_id = self.venue.camera_for_gate(gate_id)
            if camera_id is None:
                raise ValueError(f"{gate_id} has no footage to calibrate.")
            if calibration is None:
                self.calibrations.delete(camera_id)
            else:
                self.calibrations.set(camera_id, calibration)
            self._apply_calibration(camera_id)
            self._sync_gates_to_state()
            logger.info("Calibration for %s (%s): %s", gate_id, camera_id,
                        calibration.to_dict() if calibration else "cleared")
            return calibration.to_dict() if calibration else None

    def still_frame(self, gate_id: str, max_width: int = 1280) -> bytes | None:
        """The first frame of a gate's footage, without overlays, for calibration."""
        camera_id = self.venue.camera_for_gate(gate_id)
        camera = self.cameras.get_config(camera_id) if camera_id else None
        if camera is None:
            return None
        frame = self._first_frame(camera)
        if frame is None:
            return None
        image, _ = scale_for_display(frame, max_width)
        return encode_jpeg(image, 88)

    @staticmethod
    def _first_frame(camera: CameraConfig) -> np.ndarray | None:
        if camera.source_type != "video":
            return None
        capture = cv2.VideoCapture(camera.source)
        try:
            ok, frame = capture.read()
            return frame if ok else None
        finally:
            capture.release()

    # ------------------------------------------------------------------
    # Venue reconfiguration (setup only)
    # ------------------------------------------------------------------

    def reconfigure_venue(
        self,
        gate_count: int,
        default_capacity: int | None = None,
        radius_m: float | None = None,
        capacities: dict[str, int] | None = None,
        venue_name: str | None = None,
    ) -> dict:
        """Rebuild the venue with a different number of gates.

        Gate ids are positional (``G1``..``Gn``), so growing the venue leaves
        existing gates and their staged footage untouched and adds empty ones.
        Shrinking removes the gates past the new count, together with any
        footage staged on them.

        Raises:
            SessionError: If a session is running.
            ValueError: If ``gate_count`` is outside the supported range.
        """
        with self._session_lock:
            self._require_setup()

            previous_ids = set(self.venue.gate_ids)
            surviving_bindings: dict[str, str] = {}
            dropped_cameras: list[str] = []
            for gate_id in self.venue.gate_ids:
                camera_id = self.venue.camera_for_gate(gate_id)
                if camera_id is None:
                    continue
                if int(gate_id.lstrip("G") or 0) <= gate_count:
                    surviving_bindings[gate_id] = camera_id
                else:
                    dropped_cameras.append(camera_id)

            # Validate the new layout before tearing anything down.
            existing_config = self.venue.config
            new_venue = build_ring_topology(
                gate_count=gate_count,
                default_capacity=default_capacity if default_capacity is not None else DEFAULT_CAPACITY,
                radius_m=radius_m if radius_m is not None else existing_config.geometry.radius_m,
                capacities=capacities,
                camera_bindings=surviving_bindings,
                venue_id=existing_config.id,
                venue_name=venue_name or existing_config.name,
            )

            for camera_id in dropped_cameras:
                self._remove_camera_unlocked(camera_id)

            self.venue = new_venue
            self.cameras.set_venue(new_venue)
            self.engine = DecisionEngine(new_venue, self.decision_config)
            self.trends.reset()
            with self._results_lock:
                self._latest_states = {}
                self._latest_decisions = {}
            self.alerts.clear()
            self.commands.reset()

            removed_gates = self.state.retain_gates(set(new_venue.gate_ids))
            self.state.reset_counters()
            self._sync_gates_to_state()
            for camera_id in surviving_bindings.values():
                self._publish_preview(camera_id)

            logger.info(
                "Venue reconfigured to %d gates. Footage kept: %s. Footage removed: %s",
                gate_count,
                ", ".join(surviving_bindings.values()) or "none",
                ", ".join(dropped_cameras) or "none",
            )

            return {
                "gate_count": gate_count,
                "gate_ids": new_venue.gate_ids,
                "gates_added": sorted(set(new_venue.gate_ids) - previous_ids),
                "gates_removed": sorted(removed_gates),
                "cameras_kept": sorted(surviving_bindings.values()),
                "cameras_stopped": sorted(dropped_cameras),
            }

    def _sync_gates_to_state(self) -> None:
        """Mirror the venue's gate and camera mapping into the dashboard."""
        for gate_id in self.venue.gate_ids:
            gate = self.venue.get_gate(gate_id)
            calibration = self.calibrations.get(gate.camera_id)
            self.state.register_gate(
                gate_id=gate_id,
                capacity=self.effective_capacity(gate_id),
                name=gate.name,
                camera_id=gate.camera_id,
                calibration=calibration.to_dict() if calibration else None,
            )
        self.state.venue_summary = self.venue.summary()

    # ------------------------------------------------------------------
    # Playback callbacks
    # ------------------------------------------------------------------

    def _publish_frame(self, camera_id: str, jpeg: bytes) -> None:
        """Renderer output: hand a finished frame to the gate's video pane."""
        if not self._session_running:
            return
        gate_id = self.venue.gate_for_camera(camera_id)
        if gate_id is not None:
            self.state.update_frame(gate_id, jpeg)

    def _header_for(self, camera_id: str) -> tuple[str | None, float | None, str | None]:
        """Gate, latest count and congestion level for a pane's header strip."""
        gate_id = self.venue.gate_for_camera(camera_id)
        with self._results_lock:
            density = self._latest_density.get(camera_id)
        gate = self.state.gates.get(gate_id) if gate_id else None
        return (
            gate_id,
            density.crowd_count if density is not None else None,
            gate.congestion_level if gate is not None else None,
        )

    # ------------------------------------------------------------------
    # Model callbacks (worker threads)
    # ------------------------------------------------------------------

    def _on_density(self, result: CSRNetResult) -> None:
        """Handle a CSRNet result. Only the count is kept; the map is not drawn."""
        if not self._session_running:
            return
        with self._results_lock:
            self._latest_density[result.camera_id] = result
        gate_id = self.venue.gate_for_camera(result.camera_id)
        if gate_id is not None:
            self.state.update_density(gate_id, result)

    def _on_motion(self, result: MotionResult) -> None:
        """Handle a crowd-motion result."""
        if not self._session_running:
            return
        self.synchronizer.add_motion_result(result)
        gate_id = self.venue.gate_for_camera(result.camera_id)
        if gate_id is not None:
            self.state.update_motion(gate_id, result)

    def _on_anomaly(self, result: AnomalyResult) -> None:
        """Handle a per-person anomaly result.

        The result becomes a playback keyframe; the renderer interpolates boxes
        between keyframes for every displayed frame.
        """
        if not self._session_running:
            return
        with self._results_lock:
            self._latest_anomaly[result.camera_id] = result
        self.renderer.ingest_keyframe(result)
        gate_id = self.venue.gate_for_camera(result.camera_id)
        if gate_id is not None:
            self.state.update_anomaly(gate_id, result)

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    def _evaluation_loop(self) -> None:
        """Periodically fuse observations, decide, and raise alerts."""
        period = 1.0 / _EVALUATION_HZ
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                if self._session_running:
                    self._evaluate_once()
                self.state.set_pipeline_stats(self.stats())
            except Exception as exc:
                logger.exception("Decision sweep failed: %s", exc)
            self._stop_event.wait(max(0.0, period - (time.monotonic() - started)))

    def _evaluate_once(self) -> None:
        """One full sweep: fuse every gate, decide, alert, publish."""
        states: dict[str, CrowdState] = {}

        for gate_id in self.venue.gate_ids:
            gate = self.venue.get_gate(gate_id)
            camera_id = gate.camera_id

            if camera_id is None:
                # No camera: still routable on declared capacity alone. Published
                # too, so the API reports what routing actually assumes.
                unmonitored = self.adapter.unmonitored_state(
                    gate_id=gate_id,
                    gate_capacity=gate.capacity,
                    trend=self.trends.get_trend(gate_id),
                )
                states[gate_id] = unmonitored
                self.state.update_crowd_state(gate_id, unmonitored)
                continue

            with self._results_lock:
                density = self._latest_density.get(camera_id)

            sync_result = None
            motion = None
            if density is not None:
                sync_result = self.synchronizer.synchronize(density, stale_frame_gap=self.stale_frame_gap)
                motion = sync_result.motion_result
            elif self.posetrack_worker is not None:
                motion = self.posetrack_worker.latest_motion(camera_id)

            if density is None and motion is None:
                continue

            crowd_state = self.adapter.fuse(
                csrnet_result=density,
                motion_result=motion,
                gate_id=gate_id,
                gate_capacity=gate.capacity,
                trend=self.trends.get_trend(gate_id),
                stale_frame_gap=self.stale_frame_gap,
                sync_result=sync_result,
                calibration=self.calibrations.get(camera_id),
            )
            states[gate_id] = crowd_state
            self.trends.update(gate_id, crowd_state.congestion_score)
            self.state.update_crowd_state(gate_id, crowd_state)

        if not states:
            return

        decisions = self.engine.evaluate(states)
        for decision in decisions:
            self.state.update_decision(decision.source_gate_id, decision)

        with self._results_lock:
            self._latest_states = states
            self._latest_decisions = {d.source_gate_id: d for d in decisions}

        self._raise_alerts(states, {d.source_gate_id: d for d in decisions})

    def _raise_alerts(self, states: dict[str, CrowdState], decisions: dict[str, DecisionResult]) -> None:
        """Evaluate the rule engine and merge the result into the store."""
        pose_stats = self.posetrack_worker.stats() if self.posetrack_worker else {}
        continuity_ratio = float(pose_stats.get("continuity_break_ratio", 0.0))

        views: dict[str, GateView] = {}
        for gate_id in self.venue.gate_ids:
            gate = self.venue.get_gate(gate_id)
            camera_id = gate.camera_id
            anomaly: AnomalyResult | None = None
            if camera_id is not None:
                with self._results_lock:
                    anomaly = self._latest_anomaly.get(camera_id)

            views[gate_id] = GateView(
                gate_id=gate_id,
                capacity=self.effective_capacity(gate_id),
                state=states.get(gate_id),
                decision=decisions.get(gate_id),
                anomaly=anomaly,
                camera_id=camera_id,
                is_monitored=camera_id is not None,
                continuity_break_ratio=continuity_ratio if camera_id else 0.0,
                max_expected_velocity=self.decision_config.congestion_weights.max_expected_velocity,
                free_walking_speed_ms=self.decision_config.free_walking_speed_ms,
            )

        alerts: list[Alert] = generate_alerts(views)
        self.alerts.submit(alerts)
        active = self.alerts.active()
        history = self.alerts.history(limit=50)
        self.state.update_alerts(
            alerts=active,
            counts=self.alerts.counts_by_severity(),
            incidents=history[:25],
        )

        commands, incidents = self.commands.build(
            decisions=decisions,
            alerts=active,
            states=states,
            capacities={g: view.capacity for g, view in views.items()},
            now=time.time(),
        )
        log = [entry for entry in (present_alert(a) for a in history) if entry is not None]
        self.state.update_instructions(
            [c.to_dict() for c in commands], [i.to_dict() for i in incidents], log,
        )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "running": self._running,
            "session": "running" if self._session_running else "setup",
            "device": self.models.device,
            "cameras": self.cameras.stats(),
            "buffers": {
                "density": self.density_buffer.stats(),
                "anomaly": self.anomaly_buffer.stats(),
            },
            "csrnet": self.csrnet_worker.stats() if self.csrnet_worker else {"enabled": False},
            "posetrack": self.posetrack_worker.stats() if self.posetrack_worker else {"enabled": False},
            "playback": self.renderer.stats(),
            "synchronizer": self.synchronizer.stats(),
            "decisions": {
                "evaluations": self.engine.evaluation_count,
                "active_diversions": self.engine.active_diversions,
            },
            "alerts": self.alerts.stats(),
        }
