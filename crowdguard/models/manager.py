"""Central lifecycle manager for the vision models."""

from __future__ import annotations

import logging
from typing import Any

import torch

from crowdguard.config.schemas import ModelsConfig
from crowdguard.models.csrnet.infer import CSRNetInference
from crowdguard.models.posetrack.classifier import AnomalyClassifier
from crowdguard.models.posetrack.infer import PoseTrackInference
from crowdguard.utils.exceptions import CheckpointError

logger = logging.getLogger(__name__)


class ModelManager:
    """Owns device selection and the lifetime of every model.

    Both branches are optional. Running with only CSRNet gives congestion
    management with no anomaly detection; running with only pose tracking gives
    anomaly detection plus motion-only congestion scoring. The decision layer
    degrades its confidence accordingly rather than guessing.
    """

    def __init__(self, models_config: ModelsConfig | None = None) -> None:
        self.config = models_config or ModelsConfig()

        requested = self.config.device
        if requested in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested
            if requested == "cuda" and not torch.cuda.is_available():
                logger.warning("device=cuda requested but CUDA is unavailable; using CPU")
                self.device = "cpu"

        self._log_device_report()

        self._csrnet: CSRNetInference | None = None
        self._posetrack: PoseTrackInference | None = None
        self._classifier: AnomalyClassifier | None = None

        if self.config.csrnet.enabled:
            self._csrnet = CSRNetInference(
                checkpoint_path=self.config.csrnet.checkpoint_path,
                device=self.device,
                inference_resolution=self.config.csrnet.inference_resolution,
                strict_checkpoint=self.config.csrnet.strict_checkpoint,
                scales=self.config.csrnet.scales,
            )

        if self.config.posetrack.enabled:
            if self.config.anomaly.enabled:
                try:
                    self._classifier = AnomalyClassifier.load(self.config.anomaly.classifier_path)
                except CheckpointError as exc:
                    # Pose tracking still delivers crowd motion, and the
                    # geometric counter-flow rule still works, so degrade
                    # rather than refuse to start.
                    logger.error(
                        "Anomaly classifier unavailable, falling back to the "
                        "rule-based counter-flow check only: %s", exc,
                    )
                    self._classifier = None

            self._posetrack = PoseTrackInference(
                pose_config=self.config.posetrack,
                anomaly_config=self.config.anomaly,
                classifier=self._classifier,
                device=self.device,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_device_report(self) -> None:
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        vram = ""
        if cuda_available:
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            vram = f" ({total:.1f} GB VRAM)"

        logger.info(
            "Compute device: %s | CUDA available: %s | GPU: %s%s",
            self.device, cuda_available, gpu_name, vram,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def csrnet_enabled(self) -> bool:
        return self._csrnet is not None

    @property
    def posetrack_enabled(self) -> bool:
        return self._posetrack is not None

    @property
    def csrnet(self) -> CSRNetInference:
        if self._csrnet is None:
            raise RuntimeError("CSRNet is not enabled on this ModelManager")
        return self._csrnet

    @property
    def posetrack(self) -> PoseTrackInference:
        if self._posetrack is None:
            raise RuntimeError("Pose tracking is not enabled on this ModelManager")
        return self._posetrack

    @property
    def classifier(self) -> AnomalyClassifier | None:
        return self._classifier

    @property
    def device_info(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "csrnet_enabled": self.csrnet_enabled,
            "posetrack_enabled": self.posetrack_enabled,
            "anomaly_classifier": self._classifier.model_name if self._classifier else None,
            "pose_model": self.config.posetrack.model_path if self.posetrack_enabled else None,
        }

    def release_camera(self, camera_id: str) -> None:
        """Free per-camera model state when a camera is removed."""
        if self._posetrack is not None:
            self._posetrack.release_camera(camera_id)

    def reset_tracking(self, camera_id: str) -> None:
        """Clear a camera's tracking history while keeping its model loaded."""
        if self._posetrack is not None:
            self._posetrack.reset_tracking(camera_id)
