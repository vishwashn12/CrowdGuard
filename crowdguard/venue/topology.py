"""Venue topology graph for gate connectivity, distance, and adjacency queries.

An in-memory graph the Decision Engine uses for routing. No dependency on
torch, cv2, or any vision model code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from crowdguard.venue.models import VenueConfig


@dataclass
class GateNode:
    """In-memory representation of a gate in the topology graph."""

    gate_id: str
    capacity: int
    position_index: int
    camera_id: str | None
    name: str | None

    @property
    def is_monitored(self) -> bool:
        """Whether a camera is bound to this gate."""
        return self.camera_id is not None


@dataclass
class Edge:
    """Directed edge between two gates."""

    from_gate: str
    to_gate: str
    distance_m: float


class VenueTopology:
    """Graph-based venue topology for connectivity and routing queries.

    Built from a :class:`VenueConfig`. Provides gate lookup, camera resolution,
    adjacency enumeration, all-pairs shortest distances, and candidate
    enumeration for routing.

    The camera mapping is mutable at runtime (:meth:`bind_camera`,
    :meth:`unbind_camera`) so operators can attach uploaded footage to a gate
    without a restart. The graph structure itself is immutable.
    """

    def __init__(self, config: VenueConfig) -> None:
        self._config = config
        self._gates: dict[str, GateNode] = {}
        self._adjacency: dict[str, list[Edge]] = defaultdict(list)
        self._camera_to_gate: dict[str, str] = {}
        self._gate_to_camera: dict[str, str | None] = {}
        self._edge_distances: dict[tuple[str, str], float] = {}
        self._shortest_distances: dict[tuple[str, str], float] = {}

        self._build(config)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, config: VenueConfig) -> None:
        for gate_cfg in config.gates:
            self._gates[gate_cfg.gate_id] = GateNode(
                gate_id=gate_cfg.gate_id,
                capacity=gate_cfg.capacity,
                position_index=gate_cfg.position_index,
                camera_id=gate_cfg.camera_id,
                name=gate_cfg.name,
            )
            self._gate_to_camera[gate_cfg.gate_id] = gate_cfg.camera_id
            if gate_cfg.camera_id is not None:
                self._camera_to_gate[gate_cfg.camera_id] = gate_cfg.gate_id

        for conn in config.get_effective_connections():
            if not any(e.to_gate == conn.to_gate for e in self._adjacency[conn.from_gate]):
                self._adjacency[conn.from_gate].append(Edge(conn.from_gate, conn.to_gate, conn.distance_m))
            if not any(e.to_gate == conn.from_gate for e in self._adjacency[conn.to_gate]):
                self._adjacency[conn.to_gate].append(Edge(conn.to_gate, conn.from_gate, conn.distance_m))
            # Two gates can be joined twice (a 2-gate ring: once each way round).
            # The shorter link is the one anyone would walk.
            for key in ((conn.from_gate, conn.to_gate), (conn.to_gate, conn.from_gate)):
                self._edge_distances[key] = min(conn.distance_m, self._edge_distances.get(key, float("inf")))
                for edge in self._adjacency[key[0]]:
                    if edge.to_gate == key[1] and edge.distance_m > self._edge_distances[key]:
                        edge.distance_m = self._edge_distances[key]

        self._compute_all_pairs_shortest()

    def _compute_all_pairs_shortest(self) -> None:
        """Floyd-Warshall over the (small) gate graph."""
        gate_ids = list(self._gates.keys())
        INF = float("inf")

        dist: dict[tuple[str, str], float] = {}
        for g1 in gate_ids:
            for g2 in gate_ids:
                if g1 == g2:
                    dist[(g1, g2)] = 0.0
                elif (g1, g2) in self._edge_distances:
                    dist[(g1, g2)] = self._edge_distances[(g1, g2)]
                else:
                    dist[(g1, g2)] = INF

        for k in gate_ids:
            for i in gate_ids:
                for j in gate_ids:
                    through = dist[(i, k)] + dist[(k, j)]
                    if through < dist[(i, j)]:
                        dist[(i, j)] = through

        self._shortest_distances = dist

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> VenueConfig:
        return self._config

    @property
    def gate_ids(self) -> list[str]:
        """All gate IDs in ``position_index`` order."""
        return [g.gate_id for g in sorted(self._gates.values(), key=lambda g: g.position_index)]

    @property
    def monitored_gate_ids(self) -> list[str]:
        """Gate IDs that currently have a camera bound."""
        return [gid for gid in self.gate_ids if self._gate_to_camera.get(gid) is not None]

    @property
    def unmonitored_gate_ids(self) -> list[str]:
        """Gate IDs with no camera. Still valid diversion targets."""
        return [gid for gid in self.gate_ids if self._gate_to_camera.get(gid) is None]

    @property
    def gate_count(self) -> int:
        return len(self._gates)

    @property
    def venue_id(self) -> str:
        return self._config.id

    @property
    def venue_name(self) -> str:
        return self._config.name

    @property
    def geometry_type(self) -> str:
        return self._config.geometry.type

    @property
    def radius_m(self) -> float:
        return self._config.geometry.radius_m

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def has_gate(self, gate_id: str) -> bool:
        return gate_id in self._gates

    def get_gate(self, gate_id: str) -> GateNode:
        """Look up a gate by ID.

        Raises:
            KeyError: If the gate is not in the venue.
        """
        if gate_id not in self._gates:
            raise KeyError(f"Unknown gate_id: {gate_id}")
        return self._gates[gate_id]

    def get_gate_capacity(self, gate_id: str) -> int:
        return self.get_gate(gate_id).capacity

    def gate_for_camera(self, camera_id: str) -> str | None:
        """Resolve camera_id -> gate_id, or None when unmapped."""
        return self._camera_to_gate.get(camera_id)

    def camera_for_gate(self, gate_id: str) -> str | None:
        """Resolve gate_id -> camera_id, or None when unmonitored."""
        return self._gate_to_camera.get(gate_id)

    def is_monitored(self, gate_id: str) -> bool:
        return self._gate_to_camera.get(gate_id) is not None

    # ------------------------------------------------------------------
    # Runtime camera binding
    # ------------------------------------------------------------------

    def bind_camera(self, camera_id: str, gate_id: str) -> None:
        """Attach a camera to a gate at runtime.

        Any previous camera on that gate, and any previous gate for that camera,
        are released first so the mapping stays one-to-one.

        Raises:
            KeyError: If the gate does not exist in the venue.
        """
        if gate_id not in self._gates:
            raise KeyError(f"Cannot bind camera to unknown gate: {gate_id}")

        previous_gate = self._camera_to_gate.get(camera_id)
        if previous_gate is not None and previous_gate != gate_id:
            self._gate_to_camera[previous_gate] = None
            self._gates[previous_gate].camera_id = None

        existing_camera = self._gate_to_camera.get(gate_id)
        if existing_camera is not None and existing_camera != camera_id:
            self._camera_to_gate.pop(existing_camera, None)

        self._camera_to_gate[camera_id] = gate_id
        self._gate_to_camera[gate_id] = camera_id
        self._gates[gate_id].camera_id = camera_id

    def unbind_camera(self, camera_id: str) -> str | None:
        """Detach a camera, returning the gate it was bound to (if any)."""
        gate_id = self._camera_to_gate.pop(camera_id, None)
        if gate_id is not None:
            self._gate_to_camera[gate_id] = None
            self._gates[gate_id].camera_id = None
        return gate_id

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def neighbors(self, gate_id: str) -> list[str]:
        """Directly adjacent gate IDs."""
        seen: set[str] = set()
        result: list[str] = []
        for edge in self._adjacency.get(gate_id, []):
            if edge.to_gate not in seen:
                seen.add(edge.to_gate)
                result.append(edge.to_gate)
        return result

    def direct_distance(self, from_gate: str, to_gate: str) -> float | None:
        """Direct edge distance, or None when not directly connected."""
        return self._edge_distances.get((from_gate, to_gate))

    def shortest_distance(self, from_gate: str, to_gate: str) -> float:
        """Shortest path distance, or ``float('inf')`` when unreachable."""
        return self._shortest_distances.get((from_gate, to_gate), float("inf"))

    def reachable_gates(self, from_gate: str, max_distance_m: float | None = None) -> list[str]:
        """All gates reachable from ``from_gate``, nearest first."""
        candidates: list[tuple[str, float]] = []
        for gid in self._gates:
            if gid == from_gate:
                continue
            d = self.shortest_distance(from_gate, gid)
            if d < float("inf") and (max_distance_m is None or d <= max_distance_m):
                candidates.append((gid, d))
        candidates.sort(key=lambda x: x[1])
        return [gid for gid, _ in candidates]

    def max_distance(self) -> float:
        """Largest finite shortest-path distance in the venue (for normalisation)."""
        finite = [d for d in self._shortest_distances.values() if 0 < d < float("inf")]
        return max(finite) if finite else 1.0

    def perimeter_path(self, from_gate: str, to_gate: str) -> list[str]:
        """Gate sequence walking the perimeter ring from one gate to another.

        Picks the shorter of the two ring directions **in metres**, not in
        number of gates: with uneven spacing, fewer gates can be further. Used to
        describe and draw a diversion route that stays on the outer concourse.
        """
        ring = self.gate_ids
        if from_gate not in ring or to_gate not in ring:
            return []

        i, j = ring.index(from_gate), ring.index(to_gate)
        n = len(ring)
        segments = self.ring_segments_m()

        forward = [ring[(i + step) % n] for step in range(((j - i) % n) + 1)]
        backward = [ring[(i - step) % n] for step in range(((i - j) % n) + 1)]
        forward_m = sum(segments[(i + step) % n] for step in range((j - i) % n))
        backward_m = sum(segments[(i - step - 1) % n] for step in range((i - j) % n))
        return forward if forward_m <= backward_m else backward

    def ring_segments_m(self) -> list[float]:
        """Walking distance from each gate (perimeter order) to the next one."""
        return self._config.ring_segment_distances()

    def summary(self) -> dict:
        """Summary dictionary for logging and the dashboard."""
        return {
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "gate_count": self.gate_count,
            "gate_ids": self.gate_ids,
            "monitored_gates": self.monitored_gate_ids,
            "unmonitored_gates": self.unmonitored_gate_ids,
            "geometry_type": self.geometry_type,
            "radius_m": self.radius_m,
            "connection_count": sum(len(v) for v in self._adjacency.values()) // 2,
            "segments": [
                {"from": a, "to": self.gate_ids[(k + 1) % len(self.gate_ids)], "distance_m": round(d, 1)}
                for k, (a, d) in enumerate(zip(self.gate_ids, self.ring_segments_m()))
            ] if len(self.gate_ids) >= 2 else [],
            "perimeter_m": round(sum(self.ring_segments_m()), 1),
            "gates": [
                {
                    "gate_id": g.gate_id,
                    "name": g.name,
                    "capacity": g.capacity,
                    "position_index": g.position_index,
                    "camera_id": g.camera_id,
                    "is_monitored": g.is_monitored,
                }
                for g in sorted(self._gates.values(), key=lambda g: g.position_index)
            ],
        }
