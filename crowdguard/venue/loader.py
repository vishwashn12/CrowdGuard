"""Load venue configuration from YAML.

No dependency on torch, cv2, or any vision model code.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from crowdguard.utils.exceptions import ConfigurationError
from crowdguard.venue.models import VenueConfig
from crowdguard.venue.topology import VenueTopology


def load_venue_config(config_path: str | Path) -> VenueConfig:
    """Load and validate a venue YAML file.

    Accepts either a top-level ``venue:`` key or a flat mapping.

    Raises:
        ConfigurationError: If the file is missing or fails validation.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigurationError(f"Venue configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigurationError("Venue configuration root must be a mapping")

    venue_data = raw.get("venue", raw)

    try:
        return VenueConfig.model_validate(venue_data)
    except Exception as exc:
        raise ConfigurationError(f"Invalid venue configuration: {exc}") from exc


def load_venue_topology(config_path: str | Path) -> VenueTopology:
    """Load a venue configuration and build its topology graph."""
    return VenueTopology(load_venue_config(config_path))
