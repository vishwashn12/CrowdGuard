"""Static venue configuration models.

These describe the venue topology: gates, connections, geometry, and the
camera-to-gate mapping. They carry no dynamic crowd state and no dependency on
any vision model (torch, cv2, CSRNet, YOLO).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GateConfig(BaseModel):
    """Configuration for a single gate in the venue.

    ``camera_id`` is optional. A gate with no camera is *unmonitored*: it has no
    observations of its own, but remains part of the topology and can still be
    chosen as a diversion target on the strength of its declared capacity.
    """

    gate_id: str = Field(..., min_length=1, description="Unique gate identifier, e.g. 'G1'")
    capacity: int = Field(..., gt=0, description="Maximum crowd capacity for this gate")
    position_index: int = Field(..., ge=0, description="Position index on the circular venue (0-based)")
    camera_id: str | None = Field(default=None, description="Camera observing this gate (None if unmonitored)")
    name: str | None = Field(default=None, description="Human-readable gate name")


class ConnectionConfig(BaseModel):
    """Explicit connection between two gates with a travel distance."""

    from_gate: str = Field(..., alias="from", min_length=1)
    to_gate: str = Field(..., alias="to", min_length=1)
    distance_m: float = Field(..., gt=0, description="Travel distance in metres between gates")

    model_config = {"populate_by_name": True}


class GeometryConfig(BaseModel):
    """Venue geometry specification."""

    type: Literal["circular"] = "circular"
    radius_m: float = Field(default=100.0, gt=0, description="Radius of the circular venue in metres")


class VenueConfig(BaseModel):
    """Complete static venue configuration loaded from YAML.

    Supports any number of gates. Connections may be given explicitly or derived
    from circular ordering, in which case adjacent gates are joined along the
    perimeter ring so that diversion routes never cut through the event grounds.
    """

    id: str = Field(..., min_length=1, description="Unique venue identifier")
    name: str = Field(..., min_length=1, description="Human-readable venue name")
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    gates: list[GateConfig] = Field(..., min_length=1)
    connections: list[ConnectionConfig] = Field(default_factory=list)
    segment_distances_m: list[float] | None = Field(
        default=None,
        description="Walking distance in metres along the perimeter from each gate to the "
                    "next, in position order (the last entry closes the ring back to the "
                    "first gate). Omitted: gates are spaced evenly on the ring.",
    )
    auto_circular_connections: bool = Field(
        default=True,
        description="Derive circular adjacency from gate ordering when no explicit "
                    "connections are supplied",
    )

    @model_validator(mode="after")
    def validate_venue(self) -> "VenueConfig":
        gate_ids = [g.gate_id for g in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError(f"Duplicate gate IDs found: {gate_ids}")

        positions = [g.position_index for g in self.gates]
        if len(positions) != len(set(positions)):
            raise ValueError(f"Duplicate position indices found: {positions}")

        camera_ids = [g.camera_id for g in self.gates if g.camera_id is not None]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError(f"Duplicate camera IDs found: {camera_ids}")

        if self.segment_distances_m is not None:
            if len(self.segment_distances_m) != len(self.gates):
                raise ValueError(
                    f"segment_distances_m needs one distance per gate ({len(self.gates)}), "
                    f"got {len(self.segment_distances_m)}"
                )
            bad = [d for d in self.segment_distances_m if not (0 < d <= 5000)]
            if bad:
                raise ValueError(f"Walking distances must be between 0 and 5000 m, got {bad}")

        for conn in self.connections:
            if conn.from_gate not in gate_ids:
                raise ValueError(f"Connection references unknown gate: {conn.from_gate}")
            if conn.to_gate not in gate_ids:
                raise ValueError(f"Connection references unknown gate: {conn.to_gate}")
            if conn.from_gate == conn.to_gate:
                raise ValueError(f"Self-connection not allowed: {conn.from_gate}")

        return self

    def get_camera_to_gate_map(self) -> dict[str, str]:
        """Mapping of camera_id -> gate_id for all monitored gates."""
        return {g.camera_id: g.gate_id for g in self.gates if g.camera_id is not None}

    def get_gate_to_camera_map(self) -> dict[str, str | None]:
        """Mapping of gate_id -> camera_id (None if unmonitored)."""
        return {g.gate_id: g.camera_id for g in self.gates}

    def derive_circular_connections(self) -> list[ConnectionConfig]:
        """Derive ring adjacency from gate ordering.

        Gates are joined in a ring by ``position_index``. Each edge is the walking
        distance from a gate to the next: ``segment_distances_m`` when given,
        otherwise the arc between evenly spaced gates.
        """
        sorted_gates = sorted(self.gates, key=lambda g: g.position_index)
        n = len(sorted_gates)
        if n < 2:
            return []

        segments = self.ring_segment_distances()
        derived: list[ConnectionConfig] = []
        for i in range(n):
            g_from = sorted_gates[i]
            g_to = sorted_gates[(i + 1) % n]
            derived.append(ConnectionConfig(
                **{"from": g_from.gate_id, "to": g_to.gate_id, "distance_m": round(segments[i], 2)}
            ))
        return derived

    def ring_segment_distances(self) -> list[float]:
        """Walking distance from each gate (in position order) to the next."""
        n = len(self.gates)
        if self.segment_distances_m is not None:
            return [float(d) for d in self.segment_distances_m]
        return [self.geometry.radius_m * 2 * math.pi / n] * n if n else []

    def get_effective_connections(self) -> list[ConnectionConfig]:
        """Explicit connections when present, otherwise the derived ring."""
        if self.connections:
            return self.connections
        if self.auto_circular_connections:
            return self.derive_circular_connections()
        return []
