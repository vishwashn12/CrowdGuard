"""YAML configuration loading with path resolution and .env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from crowdguard.config.schemas import AppConfig
from crowdguard.utils.exceptions import ConfigurationError

# Repository root: <root>/crowdguard/config/loader.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_relative(value: str, base_dir: Path) -> str:
    """Resolve a relative path against the project root, then the CWD."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if (base_dir / path).exists():
        return str((base_dir / path).resolve())
    if (Path.cwd() / path).exists():
        return str((Path.cwd() / path).resolve())
    # Keep a best-effort absolute path so error messages stay actionable.
    return str((base_dir / path).resolve())


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay a small set of environment variables onto the loaded mapping.

    Only operational knobs are exposed this way, so a deployment can be retuned
    without editing YAML. Unset variables are ignored.
    """
    models = data.setdefault("models", {})
    ingestion = data.setdefault("ingestion", {})
    csrnet = models.setdefault("csrnet", {})
    posetrack = models.setdefault("posetrack", {})
    anomaly = models.setdefault("anomaly", {})

    env_map: list[tuple[str, dict, str, Any]] = [
        ("DEVICE", models, "device", str),
        ("DENSITY_FPS", ingestion, "density_fps", float),
        ("ANOMALY_STRIDE", ingestion, "anomaly_stride", int),
        ("REAL_TIME_MODE", ingestion, "real_time_mode", bool),
        ("LOOP_SOURCES", ingestion, "loop_sources", bool),
        ("CSRNET_ENABLED", csrnet, "enabled", bool),
        ("CSRNET_CHECKPOINT", csrnet, "checkpoint_path", str),
        ("POSE_ENABLED", posetrack, "enabled", bool),
        ("POSE_MODEL", posetrack, "model_path", str),
        ("POSE_CONF", posetrack, "confidence_threshold", float),
        ("POSE_IMGSZ", posetrack, "imgsz", int),
        ("ANOMALY_ENABLED", anomaly, "enabled", bool),
        ("ANOMALY_CLASSIFIER", anomaly, "classifier_path", str),
        ("VENUE_CONFIG", data, "venue_config", str),
    ]

    for env_key, target, field, caster in env_map:
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        if caster is bool:
            target[field] = raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            target[field] = caster(raw)

    return data


def load_app_config(config_path: str | Path, apply_env: bool = True) -> AppConfig:
    """Load, env-overlay, path-resolve and validate an application config.

    Args:
        config_path: Path to the YAML file.
        apply_env: Overlay recognised environment variables (see ``.env.example``).

    Raises:
        ConfigurationError: If the file is missing or fails validation.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigurationError(f"Configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    if apply_env:
        loaded = _apply_env_overrides(loaded)

    base = PROJECT_ROOT

    for index, camera in enumerate(loaded.get("cameras") or []):
        if isinstance(camera, dict) and isinstance(camera.get("source"), str):
            if camera.get("source_type", "video") == "video":
                loaded["cameras"][index]["source"] = _resolve_relative(camera["source"], base)

    if isinstance(loaded.get("venue_config"), str):
        loaded["venue_config"] = _resolve_relative(loaded["venue_config"], base)

    models_block = loaded.get("models") or {}
    csrnet_block = models_block.get("csrnet") or {}
    if isinstance(csrnet_block.get("checkpoint_path"), str):
        csrnet_block["checkpoint_path"] = _resolve_relative(csrnet_block["checkpoint_path"], base)
    anomaly_block = models_block.get("anomaly") or {}
    if isinstance(anomaly_block.get("classifier_path"), str):
        anomaly_block["classifier_path"] = _resolve_relative(anomaly_block["classifier_path"], base)

    uploads_block = loaded.get("uploads") or {}
    if isinstance(uploads_block.get("directory"), str):
        directory = Path(uploads_block["directory"])
        if not directory.is_absolute():
            uploads_block["directory"] = str((base / directory).resolve())

    try:
        return AppConfig.model_validate(loaded)
    except Exception as exc:
        raise ConfigurationError(f"Invalid application configuration: {exc}") from exc


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader - sets variables that are not already defined."""
    env_file = Path(path) if path is not None else (PROJECT_ROOT / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
