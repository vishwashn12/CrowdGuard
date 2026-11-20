"""Venue topology: ring derivation, distances, camera binding, perimeter routes."""

from __future__ import annotations

import math

import pytest

from conftest import make_venue

from crowdguard.utils.exceptions import ConfigurationError
from crowdguard.venue.loader import load_venue_config
from crowdguard.venue.models import GateConfig, GeometryConfig, VenueConfig
from crowdguard.venue.topology import VenueTopology


class TestRingDerivation:
    def test_gates_ordered_by_position_index(self):
        venue = make_venue(5)
        assert venue.gate_ids == ["G1", "G2", "G3", "G4", "G5"]

    def test_derived_arc_length_matches_geometry(self):
        venue = make_venue(4, radius_m=100.0)
        expected = 100.0 * (2 * math.pi / 4)
        assert venue.direct_distance("G1", "G2") == pytest.approx(expected, abs=0.01)

    def test_ring_closes_on_itself(self):
        venue = make_venue(4)
        assert "G1" in venue.neighbors("G4")
        assert "G4" in venue.neighbors("G1")

    def test_each_gate_has_exactly_two_neighbours(self):
        venue = make_venue(6)
        for gate_id in venue.gate_ids:
            assert len(venue.neighbors(gate_id)) == 2

    def test_explicit_connections_override_derivation(self):
        config = VenueConfig(
            id="v", name="V",
            gates=[
                GateConfig(gate_id="A", capacity=10, position_index=0),
                GateConfig(gate_id="B", capacity=10, position_index=1),
                GateConfig(gate_id="C", capacity=10, position_index=2),
            ],
            connections=[{"from": "A", "to": "B", "distance_m": 50.0}],
        )
        venue = VenueTopology(config)
        assert venue.direct_distance("A", "B") == 50.0
        # C was never connected, so it is unreachable rather than silently ringed.
        assert venue.shortest_distance("A", "C") == float("inf")


class TestDistances:
    def test_shortest_path_goes_around_the_ring(self):
        venue = make_venue(4, radius_m=100.0)
        arc = 100.0 * (2 * math.pi / 4)
        # Opposite gates: two hops either way.
        assert venue.shortest_distance("G1", "G3") == pytest.approx(2 * arc, abs=0.1)

    def test_self_distance_is_zero(self):
        venue = make_venue(4)
        assert venue.shortest_distance("G2", "G2") == 0.0

    def test_reachable_gates_sorted_nearest_first(self):
        venue = make_venue(6)
        reachable = venue.reachable_gates("G1")
        distances = [venue.shortest_distance("G1", g) for g in reachable]
        assert distances == sorted(distances)
        assert "G1" not in reachable

    def test_max_distance_is_finite_and_positive(self):
        venue = make_venue(5)
        assert 0 < venue.max_distance() < float("inf")

    def test_reachable_respects_max_distance(self):
        venue = make_venue(6, radius_m=100.0)
        arc = 100.0 * (2 * math.pi / 6)
        near = venue.reachable_gates("G1", max_distance_m=arc * 1.5)
        assert set(near) == {"G2", "G6"}


class TestMonitoring:
    def test_unmonitored_gates_are_reported(self):
        venue = make_venue(4, monitored=2)
        assert venue.monitored_gate_ids == ["G1", "G2"]
        assert venue.unmonitored_gate_ids == ["G3", "G4"]

    def test_unmonitored_gate_is_still_reachable(self):
        """An unmonitored gate must stay routable: its capacity is still known."""
        venue = make_venue(4, monitored=1)
        assert "G3" in venue.reachable_gates("G1")
        assert venue.get_gate_capacity("G3") == 500

    def test_bind_camera_at_runtime(self):
        venue = make_venue(3, monitored=0)
        assert not venue.is_monitored("G2")

        venue.bind_camera("cam_x", "G2")

        assert venue.is_monitored("G2")
        assert venue.gate_for_camera("cam_x") == "G2"
        assert venue.camera_for_gate("G2") == "cam_x"

    def test_binding_a_camera_to_a_new_gate_releases_the_old_one(self):
        venue = make_venue(3, monitored=0)
        venue.bind_camera("cam_x", "G1")
        venue.bind_camera("cam_x", "G2")

        assert venue.camera_for_gate("G1") is None
        assert venue.camera_for_gate("G2") == "cam_x"
        assert venue.gate_for_camera("cam_x") == "G2"

    def test_binding_a_second_camera_to_one_gate_displaces_the_first(self):
        venue = make_venue(3, monitored=0)
        venue.bind_camera("cam_a", "G1")
        venue.bind_camera("cam_b", "G1")

        assert venue.camera_for_gate("G1") == "cam_b"
        assert venue.gate_for_camera("cam_a") is None

    def test_unbind_returns_the_gate(self):
        venue = make_venue(3, monitored=1)
        assert venue.unbind_camera("cam_1") == "G1"
        assert not venue.is_monitored("G1")

    def test_bind_to_unknown_gate_raises(self):
        venue = make_venue(3)
        with pytest.raises(KeyError):
            venue.bind_camera("cam_x", "G99")


class TestPerimeterPath:
    def test_adjacent_gates_give_a_two_step_path(self):
        venue = make_venue(6)
        assert venue.perimeter_path("G1", "G2") == ["G1", "G2"]

    def test_path_takes_the_shorter_way_round(self):
        venue = make_venue(6)
        # G6 is one hop backwards, five hops forwards.
        assert venue.perimeter_path("G1", "G6") == ["G1", "G6"]

    def test_path_walks_through_intermediate_gates(self):
        venue = make_venue(6)
        assert venue.perimeter_path("G1", "G3") == ["G1", "G2", "G3"]

    def test_unknown_gate_yields_no_path(self):
        venue = make_venue(4)
        assert venue.perimeter_path("G1", "G99") == []


class TestValidation:
    def test_duplicate_gate_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate gate IDs"):
            VenueConfig(
                id="v", name="V",
                gates=[
                    GateConfig(gate_id="A", capacity=10, position_index=0),
                    GateConfig(gate_id="A", capacity=10, position_index=1),
                ],
            )

    def test_duplicate_position_indices_rejected(self):
        with pytest.raises(ValueError, match="Duplicate position indices"):
            VenueConfig(
                id="v", name="V",
                gates=[
                    GateConfig(gate_id="A", capacity=10, position_index=0),
                    GateConfig(gate_id="B", capacity=10, position_index=0),
                ],
            )

    def test_duplicate_camera_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate camera IDs"):
            VenueConfig(
                id="v", name="V",
                gates=[
                    GateConfig(gate_id="A", capacity=10, position_index=0, camera_id="c1"),
                    GateConfig(gate_id="B", capacity=10, position_index=1, camera_id="c1"),
                ],
            )

    def test_connection_to_unknown_gate_rejected(self):
        with pytest.raises(ValueError, match="unknown gate"):
            VenueConfig(
                id="v", name="V",
                gates=[GateConfig(gate_id="A", capacity=10, position_index=0)],
                connections=[{"from": "A", "to": "Z", "distance_m": 10.0}],
            )

    def test_zero_capacity_rejected(self):
        with pytest.raises(ValueError):
            GateConfig(gate_id="A", capacity=0, position_index=0)

    def test_missing_venue_file_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="not found"):
            load_venue_config("does/not/exist.yaml")


class TestRealVenueConfig:
    def test_shipped_default_venue_loads(self):
        """The config the project ships with must actually be valid."""
        from conftest import ROOT

        venue_file = ROOT / "config" / "venue" / "circular_venue.yaml"
        if not venue_file.exists():
            pytest.skip("default venue config not present")

        venue = VenueTopology(load_venue_config(venue_file))
        assert venue.gate_count >= 2
        # Shipped with no cameras: gates are bound by upload at runtime.
        assert venue.unmonitored_gate_ids == venue.gate_ids
        for gate_id in venue.gate_ids:
            assert len(venue.neighbors(gate_id)) == 2
