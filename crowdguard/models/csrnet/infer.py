"""CSRNet density inference, reproducing the original paper pipeline.

Inference runs at several scales and averages the corrected counts. A
single-scale pass systematically under-counts dense regions, because the
network's receptive field suits one head size; averaging across scales recovers
much of that.

Each scale's raw sum is divided by ``s * s`` to undo the area change introduced
by resizing, so the estimates are comparable before averaging.

The scale set is configurable because it is the dominant cost in this branch:
measured on an RTX 3050, three scales at 720x540 cost ~203 ms per frame against
~74 ms for one. Trading it away is legitimate, but it shifts the absolute count
(about -10% in that measurement), and the count feeds ``density_score`` and
therefore every congestion threshold. **Gate capacities are calibrated against
whatever scale set is in use**, so changing it means re-checking them, or the
system will under-react.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.models.contracts import CSRNetResult
from crowdguard.models.csrnet.architecture import CSRNet
from crowdguard.models.csrnet.weights import build_csrnet_model, resolve_checkpoint_path

logger = logging.getLogger(__name__)

# Scale factors from the reference implementation, used when none are given.
_DEFAULT_SCALES = (0.7, 1.0, 1.3)

# Three pooling stages mean every input side must be a multiple of 8.
_ALIGNMENT = 8


class CSRNetInference:
    """Persistent CSRNet inference worker."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        inference_resolution: tuple[int, int] | None = None,
        model_name: str = "CSRNet-VGG16",
        strict_checkpoint: bool = True,
        scales: tuple[float, ...] | None = None,
    ) -> None:
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.inference_resolution = inference_resolution
        self.model_name = model_name
        self.scales = tuple(scales) if scales else _DEFAULT_SCALES

        if len(self.scales) == 1:
            logger.warning(
                "CSRNet is running single-scale, which under-counts relative to the "
                "calibrated %d-scale pipeline. Verify gate capacities were set against "
                "this configuration, or congestion will be under-reported.",
                len(_DEFAULT_SCALES),
            )

        resolved = resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_identifier = resolved.name if resolved else "CSRNet-ShanghaiTech-A"

        self.model: CSRNet = build_csrnet_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            strict_checkpoint=strict_checkpoint,
        )

        # Normalisation must match training exactly or counts drift.
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.frames_processed = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _align(value: int) -> int:
        """Round down to a multiple of 8, with a floor of 8."""
        return max(_ALIGNMENT, (value // _ALIGNMENT) * _ALIGNMENT)

    def _multiscale_count(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Run the three-scale pass.

        Returns:
            ``(crowd_count, density_map_at_scale_1)``.
        """
        original_width, original_height = image.size
        counts: list[float] = []
        density_at_full_scale: np.ndarray | None = None

        for scale in self.scales:
            width = self._align(int(original_width * scale))
            height = self._align(int(original_height * scale))

            scaled = image.resize((width, height))
            tensor = self._transform(scaled).unsqueeze(0).to(self.device)

            with torch.no_grad():
                density = self.model(tensor)

            density_np = density.squeeze().cpu().numpy().astype(np.float32)
            if density_np.ndim == 0:
                density_np = np.array([[float(density_np)]], dtype=np.float32)

            # Undo the area scaling so the estimates are comparable.
            counts.append(float(np.sum(density_np)) / (scale * scale))

            if abs(scale - 1.0) < 1e-6:
                density_at_full_scale = density_np

        crowd_count = max(0.0, float(np.mean(counts)))

        if density_at_full_scale is None:
            density_at_full_scale = np.zeros(
                (original_height // _ALIGNMENT, original_width // _ALIGNMENT),
                dtype=np.float32,
            )

        return crowd_count, density_at_full_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, packet: FramePacket) -> CSRNetResult:
        """Estimate crowd density and count for one frame."""
        started = time.perf_counter()

        frame = packet.frame
        original_height, original_width = frame.shape[:2]

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self.inference_resolution is not None:
            target_width = self._align(self.inference_resolution[0])
            target_height = self._align(self.inference_resolution[1])
            image = image.resize((target_width, target_height))

        crowd_count, density_map = self._multiscale_count(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.frames_processed += 1

        map_height, map_width = density_map.shape[:2]

        return CSRNetResult(
            camera_id=packet.camera_id,
            frame_id=packet.frame_id,
            source_frame_index=packet.source_frame_index,
            video_timestamp=packet.video_timestamp,
            wall_clock_timestamp=packet.wall_clock_timestamp,
            crowd_count=crowd_count,
            density_map=density_map,
            density_map_width=map_width,
            density_map_height=map_height,
            input_width=original_width,
            input_height=original_height,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
            device=self.device,
            checkpoint_identifier=self.checkpoint_identifier,
            frame=frame,
            result_timestamp=time.time(),
        )

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "checkpoint": self.checkpoint_identifier,
            "device": self.device,
            "inference_resolution": self.inference_resolution,
            "scales": list(self.scales),
            "frames_processed": self.frames_processed,
        }
