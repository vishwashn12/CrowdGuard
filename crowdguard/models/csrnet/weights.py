"""CSRNet checkpoint resolution and fail-fast weight loading."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from crowdguard.models.csrnet.architecture import CSRNet
from crowdguard.utils.exceptions import CheckpointError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Filenames looked for under checkpoints/ when no explicit path is given.
_DEFAULT_NAMES = ("PartAmodel_best.pth", "csrnet_shanghaitech_part_a.pth")


def resolve_checkpoint_path(path: str | Path | None) -> Path | None:
    """Resolve a checkpoint path, searching the default location if needed."""
    if path is None or str(path) in {"default", "auto", ""}:
        for name in _DEFAULT_NAMES:
            for base in (PROJECT_ROOT, Path.cwd()):
                candidate = base / "checkpoints" / name
                if candidate.exists():
                    return candidate
        return None

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    for base in (PROJECT_ROOT, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved

    return candidate


def build_csrnet_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict_checkpoint: bool = True,
) -> CSRNet:
    """Build CSRNet and load verified crowd-counting weights.

    The density head is checked explicitly after loading. An architecture with a
    randomly initialised backend still produces a plausible-looking density map
    and a number that means nothing, so a partial load must be an error rather
    than a warning.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint.
        device: Torch device.
        strict_checkpoint: Fail when weights are missing or invalid. Only set
            False in unit tests that do not assert on counts.

    Raises:
        CheckpointError: If the checkpoint is missing, unreadable, or lacks the
            regression head.
    """
    resolved = resolve_checkpoint_path(checkpoint_path)

    if strict_checkpoint and (resolved is None or not resolved.exists()):
        raise CheckpointError(
            "CSRNet checkpoint is missing.\n"
            f"  requested: {checkpoint_path}\n"
            f"  resolved : {resolved}\n"
            "Without trained regression weights the crowd count is meaningless, so "
            "startup is refused. Place PartAmodel_best.pth in checkpoints/, point "
            "CSRNET_CHECKPOINT at it, or disable CSRNet with CSRNET_ENABLED=false."
        )

    has_checkpoint = resolved is not None and resolved.exists()
    model = CSRNet(load_weights=has_checkpoint)

    weights_loaded = False
    if has_checkpoint:
        logger.info("Loading CSRNet checkpoint: %s", resolved)
        try:
            try:
                state_dict = torch.load(str(resolved), map_location="cpu", weights_only=False)
            except Exception:
                state_dict = torch.load(str(resolved), map_location="cpu")

            if isinstance(state_dict, dict):
                for wrapper_key in ("model_state_dict", "state_dict"):
                    if wrapper_key in state_dict:
                        state_dict = state_dict[wrapper_key]
                        break

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            head_missing = [
                key for key in missing
                if key.startswith("backend") or key.startswith("output_layer")
            ]
            if head_missing:
                raise CheckpointError(
                    f"Checkpoint {resolved.name} is missing density-head weights "
                    f"{head_missing[:3]}. Counts from this model would be invented."
                )

            if unexpected:
                logger.debug("Ignoring %d unexpected checkpoint keys", len(unexpected))

            weights_loaded = True

        except CheckpointError:
            raise
        except Exception as exc:
            if strict_checkpoint:
                raise CheckpointError(
                    f"Failed to load CSRNet checkpoint {resolved}: {exc}"
                ) from exc
            logger.warning("Checkpoint load failed (%s); continuing untrained", exc)

    parameters = sum(p.numel() for p in model.parameters())
    logger.info(
        "CSRNet initialised: VGG-16 frontend + 6 dilated conv backend, %s weights, "
        "%s params, device %s",
        "trained" if weights_loaded else "UNINITIALISED",
        f"{parameters:,}", device,
    )

    model.to(device)
    model.eval()
    return model
