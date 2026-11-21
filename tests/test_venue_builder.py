"""Programmatic venue construction, used by the runtime setup screen."""

from __future__ import annotations

import math

import pytest

from conftest import ROOT  # noqa: F401  (ensures sys.path setup)

from crowdguard.venue.builder import (
    MAX_GATES,
    MIN_GATES,
    build_ring_topology,
    build_ring_venue,
    default_gate_name,
)


class TestGateCount:
    @pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 8, 10, 12])
    def test_builds_requested_number_of_gates(self, count):
        venue = build_ring_topology(count)
        assert venue.gate_count == count
        assert venue.gate_ids == [f"G{i + 1}" for i in range(count)]

    def test_rejects_too_few_gates(self):
        """One gate has nowhere to divert to, so it is not a venue."""
        with pytest.raises(ValueError, match="nowhere to divert"):
            build_ring_venue(1)

    def test_rejects_zero_and_negative(self):
        for bad in (0, -3):
            with pytest.raises(ValueError):
                build_ring_venue(bad)

    def test_rejects_absurd_gate_count(self):
        with pytest.raises(ValueError, match=f"between {MIN_GATES} and {MAX_GATES}"):
            build_ring_venue(MAX_GATES + 1)

    def test_bounds_are_inclusive(self):
        assert build_ring_topology(MIN_GATES).gate_count == MIN_GATES
        assert build_ring_topology(MAX_GATES).gate_count == MAX_GATES


class TestRingGeometry:
    def test_every_gate_has_two_neighbours(self):
        for count in (3, 5, 7, 11):
            venue = build_ring_topology(count)
            for gate_id in venue.gate_ids:
                assert len(venue.neighbors(gate_id)) == 2, f"{count}-gate ring"

    def test_arc_length_follows_radius_and_count(self):
        venue = build_ring_topology(6, radius_m=120.0)
        expected = 120.0 * (2 * math.pi / 6)
        assert venue.direct_distance("G1", "G2") == pytest.approx(expected, abs=0.01)

    def test_larger_radius_means_longer_walk(self):
        near = build_ring_topology(4, radius_m=50.0)
        far = build_ring_topology(4, radius_m=200.0)
        assert far.direct_distance("G1", "G2") > near.direct_distance("G1", "G2")

    def test_all_gates_mutually_reachable(self):
        venue = build_ring_topology(8)
        for gate_id in venue.gate_ids:
            assert len(venue.reachable_gates(gate_id)) == 7

    def test_two_gate_venue_is_a_single_pair(self):
        venue = build_ring_topology(2)
        assert venue.neighbors("G1") == ["G2"]
        assert venue.reachable_gates("G1") == ["G2"]


class TestCapacityAndNames:
    def test_default_capacity_applied_to_all(self):
        venue = build_ring_topology(4, default_capacity=750)
        assert all(venue.get_gate_capacity(g) == 750 for g in venue.gate_ids)

    def test_per_gate_capacity_overrides(self):
        venue = build_ring_topology(3, default_capacity=500, capacities={"G2": 900})
        assert venue.get_gate_capacity("G1") == 500
        assert venue.get_gate_capacity("G2") == 900

    def test_compass_names_for_common_layouts(self):
        assert default_gate_name(0, 4) == "North Gate"
        assert default_gate_name(2, 4) == "South Gate"

    def test_numeric_names_for_unusual_layouts(self):
        assert default_gate_name(0, 7) == "Gate 1"
        assert default_gate_name(6, 7) == "Gate 7"

    def test_name_overrides_win(self):
        venue = build_ring_topology(2, names={"G1": "Players Entrance"})
        assert venue.get_gate("G1").name == "Players Entrance"


class TestCameraBindings:
    def test_bindings_are_carried_into_the_new_venue(self):
        venue = build_ring_topology(4, camera_bindings={"G1": "cam_a", "G3": "cam_b"})
        assert venue.camera_for_gate("G1") == "cam_a"
        assert venue.camera_for_gate("G3") == "cam_b"
        assert venue.gate_for_camera("cam_b") == "G3"
        assert venue.unmonitored_gate_ids == ["G2", "G4"]

    def test_unbound_gates_remain_routing_targets(self):
        venue = build_ring_topology(3, camera_bindings={"G1": "cam_a"})
        assert "G2" in venue.reachable_gates("G1")
        assert venue.get_gate_capacity("G2") > 0


class TestGrowAndShrink:
    """The operations the setup screen performs."""

    def test_growing_preserves_lower_gate_ids(self):
        small = build_ring_topology(3)
        large = build_ring_topology(6)
        assert small.gate_ids == large.gate_ids[:3]

    def test_growing_keeps_bindings_for_surviving_gates(self):
        bindings = {"G1": "cam_a", "G2": "cam_b"}
        grown = build_ring_topology(6, camera_bindings=bindings)
        assert grown.camera_for_gate("G1") == "cam_a"
        assert grown.camera_for_gate("G2") == "cam_b"

    def test_shrinking_drops_the_higher_gates(self):
        shrunk = build_ring_topology(2)
        assert "G3" not in shrunk.gate_ids
        assert not shrunk.has_gate("G5")

    def test_shrinking_rejects_bindings_to_removed_gates(self):
        """A stale binding must fail loudly, not make a camera vanish."""
        with pytest.raises(ValueError, match="does not have"):
            build_ring_venue(2, camera_bindings={"G9": "cam_x"})

    def test_unknown_capacity_override_rejected(self):
        with pytest.raises(ValueError, match="unknown gates"):
            build_ring_venue(3, capacities={"G7": 100})

    def test_arc_length_changes_with_gate_count(self):
        """Adding gates shortens the walk between neighbours on a fixed ring."""
        four = build_ring_topology(4, radius_m=100.0)
        eight = build_ring_topology(8, radius_m=100.0)
        assert eight.direct_distance("G1", "G2") < four.direct_distance("G1", "G2")
