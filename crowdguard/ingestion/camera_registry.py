"""Runtime registry of cameras and their producers.

Cameras can be added and removed while the system runs, which is what lets an
operator upload footage for a gate and see it go live without a restart.

All producers write into the same two shared buffers - one per model path -
rather than per-camera buffers with a router thread in front. A single FIFO per
path keeps dynamic registration trivial (start a producer and it is simply
part of the stream) and preserves per-camera frame order, because each camera
has exactly one producer thread appending in sequence.

The trade-off is that the paths are shared: a very high frame-rate camera can
crowd out a slower one. Buffer capacity is therefore scaled by the number of
registered cameras.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from crowdguard.config.schemas import AppConfig, CameraConfig
from crowdguard.ingestion.frame_buffer import FrameBuffer
from crowdguard.ingestion.producer import DisplaySink, FrameProducer
from crowdguard.ingestion.video_source import build_source
from crowdguard.utils.exceptions import CameraRegistryError
from crowdguard.venue.topology import VenueTopology

logger = logging.getLogger(__name__)


class CameraRegistry:
    """Owns the lifecycle of every camera feeding the pipeline."""

    def __init__(
        self,
        config: AppConfig,
        venue: VenueTopology,
        density_buffer: FrameBuffer,
        anomaly_buffer: FrameBuffer,
        display_sink: DisplaySink | None = None,
    ) -> None:
        self._config = config
        self._venue = venue
        self._density_buffer = density_buffer
        self._anomaly_buffer = anomaly_buffer
        self._display_sink = display_sink

        self._producers: dict[str, FrameProducer] = {}
        self._camera_configs: dict[str, CameraConfig] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def set_venue(self, venue: VenueTopology) -> None:
        """Point the registry at a rebuilt venue.

        Called when the gate layout changes. Camera bindings for surviving gates
        are already carried into the new topology by the caller, so this only
        swaps the reference the registry validates against.
        """
        with self._lock:
            self._venue = venue

    @property
    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._producers.keys())

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._producers)

    def get_producer(self, camera_id: str) -> FrameProducer | None:
        with self._lock:
            return self._producers.get(camera_id)

    def get_config(self, camera_id: str) -> CameraConfig | None:
        with self._lock:
            return self._camera_configs.get(camera_id)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_camera(
        self,
        camera: CameraConfig,
        gate_id: str | None = None,
        start: bool = False,
    ) -> FrameProducer:
        """Register a camera and bind it to a gate, optionally starting it.

        Cameras are staged, not started, by default: playback begins for every
        gate together when the session is started (:meth:`start_all`).

        The gate is taken from ``gate_id``, else ``camera.gate_id``, else the
        venue's existing mapping for this camera id.

        Raises:
            CameraRegistryError: If the camera id is taken, no gate can be
                resolved, the gate is unknown, or the source cannot be opened.
        """
        with self._lock:
            if camera.camera_id in self._producers:
                raise CameraRegistryError(
                    f"Camera {camera.camera_id!r} is already registered. "
                    f"Remove it first to change its source."
                )

            resolved_gate = gate_id or camera.gate_id or self._venue.gate_for_camera(camera.camera_id)
            if resolved_gate is None:
                raise CameraRegistryError(
                    f"No gate for camera {camera.camera_id!r}. Pass gate_id, set it in the "
                    f"camera config, or map the camera in the venue configuration."
                )
            if not self._venue.has_gate(resolved_gate):
                raise CameraRegistryError(
                    f"Gate {resolved_gate!r} is not in venue {self._venue.venue_id!r}. "
                    f"Known gates: {', '.join(self._venue.gate_ids)}"
                )

            if camera.source_type == "video":
                source_path = Path(camera.source)
                if not source_path.exists():
                    raise CameraRegistryError(f"Video source does not exist: {source_path}")

            try:
                source = build_source(camera.source_type, camera.source)
            except Exception as exc:
                raise CameraRegistryError(
                    f"Could not build source for camera {camera.camera_id!r}: {exc}"
                ) from exc

            ingestion = self._config.ingestion
            producer = FrameProducer(
                camera_id=camera.camera_id,
                source=source,
                density_buffer=self._density_buffer,
                anomaly_buffer=self._anomaly_buffer,
                density_fps=ingestion.density_fps,
                anomaly_stride=ingestion.anomaly_stride,
                real_time_mode=ingestion.real_time_mode,
                loop_source=ingestion.loop_sources,
                display_sink=self._display_sink,
            )

            self._producers[camera.camera_id] = producer
            self._camera_configs[camera.camera_id] = camera
            self._venue.bind_camera(camera.camera_id, resolved_gate)

            logger.info(
                "Registered camera %s -> gate %s (%s: %s)",
                camera.camera_id, resolved_gate, camera.source_type,
                Path(camera.source).name if camera.source_type == "video" else camera.source_type,
            )

            if start:
                producer.start()

            return producer

    def remove_camera(self, camera_id: str) -> str | None:
        """Stop and deregister a camera. Returns the gate it was bound to."""
        with self._lock:
            producer = self._producers.pop(camera_id, None)
            self._camera_configs.pop(camera_id, None)
            if producer is None:
                return None
            producer.shutdown()
            gate_id = self._venue.unbind_camera(camera_id)
            logger.info("Removed camera %s (was gate %s)", camera_id, gate_id)
            return gate_id

    def reassign_camera(self, camera_id: str, gate_id: str) -> None:
        """Move an already-registered camera to a different gate.

        Raises:
            CameraRegistryError: If the camera or gate is unknown.
        """
        with self._lock:
            if camera_id not in self._producers:
                raise CameraRegistryError(f"Camera {camera_id!r} is not registered")
            if not self._venue.has_gate(gate_id):
                raise CameraRegistryError(f"Gate {gate_id!r} is not in this venue")
            self._venue.bind_camera(camera_id, gate_id)
            logger.info("Reassigned camera %s -> gate %s", camera_id, gate_id)

    def load_from_config(self, start: bool = False) -> list[str]:
        """Register every enabled camera in the application config.

        A camera that cannot be registered is logged and skipped, so one bad
        path does not prevent the rest of the venue from coming up.
        """
        registered: list[str] = []
        for camera in self._config.enabled_cameras:
            try:
                self.add_camera(camera, start=start)
                registered.append(camera.camera_id)
            except CameraRegistryError as exc:
                logger.error("Skipping camera %s: %s", camera.camera_id, exc)
        return registered

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self, t0: float | None = None) -> None:
        """Start every registered camera on one shared clock.

        Each producer releases its frame 0 at ``t0``, so every gate begins
        playing at the same instant and stays in lockstep.
        """
        with self._lock:
            for producer in self._producers.values():
                producer.start(t0=t0)

    def stop_playback(self) -> None:
        """Stop every camera reading but keep them registered, ready to restart."""
        with self._lock:
            producers = list(self._producers.values())
        for producer in producers:
            producer.stop()

    def pause_all(self) -> None:
        with self._lock:
            for producer in self._producers.values():
                producer.pause()

    def resume_all(self) -> None:
        with self._lock:
            for producer in self._producers.values():
                producer.resume()

    def stop_all(self) -> None:
        with self._lock:
            for producer in self._producers.values():
                producer.shutdown()
            self._producers.clear()
            self._camera_configs.clear()

    def all_complete(self) -> bool:
        """True when every registered camera has reached end of stream."""
        with self._lock:
            if not self._producers:
                return False
            return all(p.complete for p in self._producers.values())

    def stats(self) -> dict[str, dict]:
        with self._lock:
            return {cam_id: producer.stats() for cam_id, producer in self._producers.items()}
