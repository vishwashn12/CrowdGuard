"""Thread-safe shared state backing the dashboard.

Pipeline callbacks write here from worker threads; the FastAPI handlers read
from here on the event loop. Every access is guarded by one lock, and
:meth:`DashboardState.snapshot` returns plain JSON-serialisable data so a
response can never hold a reference into live mutable state.

State is keyed by **gate**, not by camera. A camera is an implementation detail
that can be swapped or reassigned at runtime; the gate is what the operator
reasons about.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from crowdguard.alerts.contracts import Alert
from crowdguard.decision.contracts import CrowdState, DecisionResult
from crowdguard.models.contracts import AnomalyResult, CSRNetResult, MotionResult

# Points retained for the dashboard sparklines.
_HISTORY_POINTS = 60


@dataclass
class GateState:
    """Latest known state of one gate."""

    gate_id: str
    name: str | None = None
    capacity: int = 500
    camera_id: str | None = None
    is_monitored: bool = False

    # Encoded frame for the MJPEG stream.
    frame_jpeg: bytes = b""
    # Increments on every new frame, so each stream client sends each frame
    # exactly once instead of resending a cached one on a timer.
    frame_seq: int = 0

    # Density branch
    crowd_count: float = 0.0
    density_score: float = 0.0
    csrnet_latency_ms: float = 0.0

    # Motion branch
    velocity: float = 0.0
    direction: float = 0.0
    flow_consistency: float = 0.0
    track_count: int = 0
    motion_confidence: float = 0.0
    motion_status: str = "UNAVAILABLE"
    pose_latency_ms: float = 0.0

    # Anomaly branch
    sitting_count: int = 0
    opposite_count: int = 0
    anomaly_ratio: float = 0.0
    tracked_count: int = 0
    crowd_direction_deg: float = 0.0
    classifier_active: bool = False
    continuity_ok: bool = True

    # Congestion
    congestion_level: str = "LOW"
    congestion_score: float = 0.0
    trend: str = "STABLE"
    confidence: float = 0.0

    # Decision
    action: str = "INITIALIZING"
    severity: str = "LOW"
    target_gate: str | None = None
    reason: str = ""
    route: list[str] = field(default_factory=list)

    # Provenance
    frame_id: int = 0
    video_timestamp: float = 0.0

    # Sparkline histories
    count_history: list[float] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)
    anomaly_history: list[int] = field(default_factory=list)

    last_update: float = field(default_factory=time.time)
    last_frame_update: float = 0.0

    @property
    def seconds_since_update(self) -> float:
        return max(0.0, time.time() - self.last_update)

    def is_stale_while(self, running: bool) -> bool:
        """No fresh observation for several seconds during a running session.

        Before Start a staged feed produces nothing by design, so it is never
        reported stale.
        """
        return running and self.is_monitored and self.seconds_since_update > 5.0


class DashboardState:
    """Single shared snapshot of the whole system."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.gates: dict[str, GateState] = {}

        self.pipeline_running: bool = False
        self.start_time: float = time.time()
        self.device: str = "cpu"
        self.venue_summary: dict = {}
        self.model_info: dict = {}
        self.pipeline_stats: dict = {}

        self.total_decisions: int = 0
        self.active_alerts: list[dict] = []
        self.alert_counts: dict[str, int] = {}
        self.recent_incidents: list[dict] = []

        # "setup": configuring, feeds staged but not playing.
        # "running": Start pressed, every gate playing on one clock.
        self.session_state: str = "setup"
        self.session_started_at: float | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_gate(
        self,
        gate_id: str,
        capacity: int,
        name: str | None = None,
        camera_id: str | None = None,
    ) -> None:
        """Create or refresh a gate entry, preserving any history it has."""
        with self._lock:
            existing = self.gates.get(gate_id)
            if existing is None:
                self.gates[gate_id] = GateState(
                    gate_id=gate_id,
                    name=name,
                    capacity=capacity,
                    camera_id=camera_id,
                    is_monitored=camera_id is not None,
                    action="NO_DATA" if camera_id is None else "INITIALIZING",
                )
                return

            existing.capacity = capacity
            existing.name = name or existing.name
            existing.camera_id = camera_id
            existing.is_monitored = camera_id is not None

    def bind_camera(self, gate_id: str, camera_id: str | None) -> None:
        """Attach or detach a camera, resetting per-feed fields on detach."""
        with self._lock:
            gate = self.gates.get(gate_id)
            if gate is None:
                return
            gate.camera_id = camera_id
            gate.is_monitored = camera_id is not None
            if camera_id is None:
                gate.frame_jpeg = b""
                gate.frame_seq += 1
                gate.action = "NO_DATA"
                gate.motion_status = "UNAVAILABLE"
                gate.crowd_count = 0.0
                gate.track_count = 0
                gate.sitting_count = 0
                gate.opposite_count = 0

    def gate_for_camera(self, camera_id: str) -> str | None:
        with self._lock:
            for gate_id, gate in self.gates.items():
                if gate.camera_id == camera_id:
                    return gate_id
            return None

    def retain_gates(self, gate_ids: set[str]) -> list[str]:
        """Drop every gate not in ``gate_ids``, returning the ones removed.

        Used when the venue is reconfigured to fewer gates: the stale cards
        must disappear rather than linger with frozen numbers.
        """
        with self._lock:
            removed = [gid for gid in self.gates if gid not in gate_ids]
            for gate_id in removed:
                del self.gates[gate_id]
            return removed

    def reset_counters(self) -> None:
        """Clear venue-wide counters after a reconfiguration."""
        with self._lock:
            self.total_decisions = 0
            self.active_alerts = []
            self.alert_counts = {}
            self.recent_incidents = []

    def set_session(self, state: str, started_at: float | None = None) -> None:
        with self._lock:
            self.session_state = state
            self.session_started_at = started_at

    def reset_gate_metrics(self) -> None:
        """Zero every gate's live readings, keeping layout, cameras and frames.

        Called at Start and Stop so a new session never opens on the previous
        session's numbers.
        """
        with self._lock:
            now = time.time()
            for gate in self.gates.values():
                gate.crowd_count = 0.0
                gate.density_score = 0.0
                gate.velocity = 0.0
                gate.direction = 0.0
                gate.flow_consistency = 0.0
                gate.track_count = 0
                gate.motion_confidence = 0.0
                gate.motion_status = "UNAVAILABLE"
                gate.sitting_count = 0
                gate.opposite_count = 0
                gate.anomaly_ratio = 0.0
                gate.tracked_count = 0
                gate.continuity_ok = True
                gate.congestion_level = "LOW"
                gate.congestion_score = 0.0
                gate.trend = "STABLE"
                gate.confidence = 0.0
                gate.action = "INITIALIZING" if gate.is_monitored else "NO_DATA"
                gate.severity = "LOW"
                gate.target_gate = None
                gate.reason = ""
                gate.route = []
                gate.count_history.clear()
                gate.score_history.clear()
                gate.anomaly_history.clear()
                gate.last_update = now

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    @staticmethod
    def _append_capped(series: list, value) -> None:
        series.append(value)
        if len(series) > _HISTORY_POINTS:
            del series[0]

    def update_density(self, gate_id: str, result: CSRNetResult) -> None:
        with self._lock:
            gate = self.gates.get(gate_id)
            if gate is None:
                return
            gate.crowd_count = result.crowd_count
            gate.frame_id = result.frame_id
            gate.video_timestamp = result.video_timestamp
            gate.csrnet_latency_ms = result.inference_time_ms
            gate.last_update = time.time()
            self._append_capped(gate.count_history, round(result.crowd_count, 1))

    def update_motion(self, gate_id: str, result: MotionResult) -> None:
        with self._lock:
            gate = self.gates.get(gate_id)
            if gate is None:
                return
            gate.velocity = result.median_velocity
            gate.direction = result.dominant_direction_deg
            gate.flow_consistency = result.flow_consistency
            gate.track_count = result.track_count
            gate.motion_confidence = result.motion_confidence
            gate.pose_latency_ms = result.inference_time_ms
            gate.last_update = time.time()

    def update_anomaly(self, gate_id: str, result: AnomalyResult) -> None:
        with self._lock:
            gate = self.gates.get(gate_id)
            if gate is None:
                return
            gate.sitting_count = result.sitting_count
            gate.opposite_count = result.opposite_count
            gate.anomaly_ratio = result.anomaly_ratio
            gate.tracked_count = result.tracked_count
            gate.crowd_direction_deg = result.crowd_direction_deg
            gate.classifier_active = result.classifier_active
            gate.continuity_ok = result.continuity_ok
            gate.last_update = time.time()
            self._append_capped(gate.anomaly_history, result.anomaly_count)

    def update_crowd_state(self, gate_id: str, state: CrowdState) -> None:
        with self._lock:
            gate = self.gates.get(gate_id)
            if gate is None:
                return
            gate.crowd_count = state.crowd_count
            gate.density_score = state.density_score
            gate.velocity = state.velocity
            gate.congestion_level = state.congestion_level.value
            gate.congestion_score = state.congestion_score
            gate.trend = state.trend.value
            gate.motion_status = state.motion_status.value
            gate.confidence = state.confidence
            gate.last_update = time.time()
            self._append_capped(gate.score_history, round(state.congestion_score, 3))

    def update_decision(self, gate_id: str, decision: DecisionResult) -> None:
        with self._lock:
            gate = self.gates.get(gate_id)
            if gate is None:
                return
            gate.action = decision.action.value
            gate.severity = decision.severity.value
            gate.target_gate = decision.target_gate_id
            gate.reason = decision.human_readable_reason
            gate.route = list(decision.route)
            self.total_decisions += 1

    def update_frame(self, gate_id: str, frame_jpeg: bytes) -> None:
        """Store the latest annotated JPEG for a gate's video pane."""
        with self._lock:
            gate = self.gates.get(gate_id)
            if gate is None or not frame_jpeg:
                return
            gate.frame_jpeg = frame_jpeg
            gate.frame_seq += 1
            gate.last_frame_update = time.time()

    def update_alerts(
        self,
        alerts: list[Alert],
        counts: dict[str, int],
        incidents: list[Alert] | None = None,
    ) -> None:
        with self._lock:
            self.active_alerts = [alert.to_dict() for alert in alerts]
            self.alert_counts = dict(counts)
            if incidents is not None:
                self.recent_incidents = [alert.to_dict() for alert in incidents]

    def set_pipeline_stats(self, stats: dict) -> None:
        with self._lock:
            self.pipeline_stats = stats

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_frame_jpeg(self, gate_id: str) -> bytes:
        """Latest annotated frame for a gate, or empty when none exists yet."""
        return self.get_frame(gate_id)[1]

    def get_frame(self, gate_id: str) -> tuple[int, bytes]:
        """``(sequence, jpeg)`` for a gate's latest frame."""
        with self._lock:
            gate = self.gates.get(gate_id)
            return (gate.frame_seq, gate.frame_jpeg) if gate is not None else (-1, b"")

    def frames(self) -> list[tuple[str, int, bytes]]:
        """``(gate_id, sequence, jpeg)`` for every gate, in one locked read."""
        with self._lock:
            return [(gid, g.frame_seq, g.frame_jpeg) for gid, g in self.gates.items()]

    def snapshot(self) -> dict:
        """Full JSON-serialisable view of the system."""
        with self._lock:
            gates = {}
            for gate_id, gate in self.gates.items():
                gates[gate_id] = {
                    "gate_id": gate.gate_id,
                    "name": gate.name,
                    "capacity": gate.capacity,
                    "camera_id": gate.camera_id,
                    "is_monitored": gate.is_monitored,
                    "is_stale": gate.is_stale_while(self.session_state == "running"),

                    "crowd_count": round(gate.crowd_count, 1),
                    "density_score": round(gate.density_score, 3),
                    "utilization": round(
                        gate.crowd_count / gate.capacity, 3
                    ) if gate.capacity else 0.0,

                    "velocity": round(gate.velocity, 1),
                    "direction": round(gate.direction, 1),
                    "flow_consistency": round(gate.flow_consistency, 3),
                    "track_count": gate.track_count,
                    "motion_confidence": round(gate.motion_confidence, 3),
                    "motion_status": gate.motion_status,

                    "sitting_count": gate.sitting_count,
                    "opposite_count": gate.opposite_count,
                    "anomaly_count": gate.sitting_count + gate.opposite_count,
                    "anomaly_ratio": round(gate.anomaly_ratio, 3),
                    "tracked_count": gate.tracked_count,
                    "crowd_direction_deg": round(gate.crowd_direction_deg, 1),
                    "classifier_active": gate.classifier_active,
                    "continuity_ok": gate.continuity_ok,

                    "congestion_level": gate.congestion_level,
                    "congestion_score": round(gate.congestion_score, 3),
                    "trend": gate.trend,
                    "confidence": round(gate.confidence, 3),

                    "action": gate.action,
                    "severity": gate.severity,
                    "target_gate": gate.target_gate,
                    "reason": gate.reason,
                    "route": gate.route,

                    "frame_id": gate.frame_id,
                    "video_timestamp": round(gate.video_timestamp, 2),
                    "csrnet_latency_ms": round(gate.csrnet_latency_ms, 0),
                    "pose_latency_ms": round(gate.pose_latency_ms, 0),

                    "count_history": gate.count_history[-30:],
                    "score_history": gate.score_history[-30:],
                    "anomaly_history": gate.anomaly_history[-30:],

                    "seconds_since_update": round(gate.seconds_since_update, 1),
                }

            return {
                "gates": gates,
                "gate_order": sorted(gates.keys()),
                "pipeline_running": self.pipeline_running,
                "session": {
                    "state": self.session_state,
                    "started_at": self.session_started_at,
                    "running_s": round(time.time() - self.session_started_at, 1)
                    if self.session_started_at else 0.0,
                    "gates_with_footage": sum(1 for g in self.gates.values() if g.is_monitored),
                },
                "uptime_s": round(time.time() - self.start_time, 1),
                "device": self.device,
                "total_decisions": self.total_decisions,
                "venue": self.venue_summary,
                "models": self.model_info,
                "pipeline": self.pipeline_stats,
                "alerts": self.active_alerts,
                "alert_counts": self.alert_counts,
                "incidents": self.recent_incidents,
                "timestamp": time.time(),
            }


# Process-wide singleton, written by the orchestrator and read by the web layer.
dashboard_state = DashboardState()
