"""Programmatic venue construction.

Lets the venue be defined at runtime from the operator's own numbers instead of
only from a YAML file, which is what makes "choose how many gates you have, then
upload footage to them" possible without editing config and restarting.

Gates are laid out evenly on a perimeter ring and named ``G1``..``Gn``. The ring
ordering is what keeps diversion routes on the outer concourse rather than
cutting through the event grounds.
"""

from __future__ import annotations

from crowdguard.venue.models import GateConfig, GeometryConfig, VenueConfig
from crowdguard.venue.topology import VenueTopology

# Guard rails. One gate has nowhere to divert to; beyond twelve the ring view
# and the operator's attention both stop scaling.
# DEFAULT_CAPACITY applies to gates whose camera is not calibrated. A
# calibrated gate's capacity follows from its measured ground zone instead.
MIN_GATES = 2
MAX_GATES = 12

DEFAULT_CAPACITY = 300
DEFAULT_RADIUS_M = 120.0

# Compass-style names for the common small layouts, so a 4-gate venue reads
# "North / East / South / West" rather than "Gate 1..4".
_COMPASS: dict[int, tuple[str, ...]] = {
    2: ("North", "South"),
    3: ("North", "South-East", "South-West"),
    4: ("North", "East", "South", "West"),
    6: ("North", "North-East", "South-East", "South", "South-West", "North-West"),
    8: (
        "North", "North-East", "East", "South-East",
        "South", "South-West", "West", "North-West",
    ),
}


def default_gate_name(index: int, gate_count: int) -> str:
    """Human-readable name for gate ``index`` (0-based) in an n-gate ring."""
    compass = _COMPASS.get(gate_count)
    if compass is not None and index < len(compass):
        return f"{compass[index]} Gate"
    return f"Gate {index + 1}"


def build_ring_venue(
    gate_count: int,
    default_capacity: int = DEFAULT_CAPACITY,
    radius_m: float = DEFAULT_RADIUS_M,
    capacities: dict[str, int] | None = None,
    names: dict[str, str] | None = None,
    camera_bindings: dict[str, str] | None = None,
    venue_id: str = "venue",
    venue_name: str = "Venue",
) -> VenueConfig:
    """Build a ring venue configuration.

    Args:
        gate_count: Number of gates, between :data:`MIN_GATES` and :data:`MAX_GATES`.
        default_capacity: Capacity for any gate not named in ``capacities``.
        radius_m: Ring radius, which sets the arc distance between neighbours.
        capacities: Per-gate capacity overrides, keyed by gate id.
        names: Per-gate name overrides, keyed by gate id.
        camera_bindings: gate_id -> camera_id to carry over existing feeds.
        venue_id: Identifier for the venue.
        venue_name: Display name.

    Raises:
        ValueError: If ``gate_count`` is outside the supported range.
    """
    if not (MIN_GATES <= gate_count <= MAX_GATES):
        raise ValueError(
            f"gate_count must be between {MIN_GATES} and {MAX_GATES}, got {gate_count}. "
            f"A single gate has nowhere to divert to."
        )

    capacities = capacities or {}
    names = names or {}
    camera_bindings = camera_bindings or {}

    valid_ids = {f"G{i + 1}" for i in range(gate_count)}

    # Reject bindings for gates this venue will not contain. Silently dropping
    # them would make a camera disappear with no signal, which is exactly the
    # kind of thing that is painful to debug later.
    unknown_bindings = sorted(set(camera_bindings) - valid_ids)
    if unknown_bindings:
        raise ValueError(
            f"camera_bindings reference gates that a {gate_count}-gate venue does not "
            f"have: {', '.join(unknown_bindings)}. Release those cameras before "
            f"shrinking the venue."
        )

    unknown_capacities = sorted(set(capacities) - valid_ids)
    if unknown_capacities:
        raise ValueError(
            f"capacities reference unknown gates: {', '.join(unknown_capacities)}"
        )

    gates: list[GateConfig] = []
    for index in range(gate_count):
        gate_id = f"G{index + 1}"
        gates.append(GateConfig(
            gate_id=gate_id,
            name=names.get(gate_id) or default_gate_name(index, gate_count),
            capacity=int(capacities.get(gate_id, default_capacity)),
            position_index=index,
            camera_id=camera_bindings.get(gate_id),
        ))

    return VenueConfig(
        id=venue_id,
        name=venue_name,
        geometry=GeometryConfig(type="circular", radius_m=radius_m),
        gates=gates,
        auto_circular_connections=True,
    )


def build_ring_topology(gate_count: int, **kwargs) -> VenueTopology:
    """Convenience wrapper returning a built :class:`VenueTopology`."""
    return VenueTopology(build_ring_venue(gate_count, **kwargs))
