"""Checks on the closed-loop simulator itself.

The evaluation results are only as good as the model, so the model's basic
physics, bookkeeping and determinism are tested like any other component.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT

from crowdguard.simulation.crowd_model import (
    DANGER_DENSITY,
    MAX_DENSITY,
    Directive,
    GateSpec,
    Scenario,
    SimConfig,
    Simulation,
    area_density,
    throughput_factor,
    weidmann_speed,
)
from crowdguard.simulation.policies import POLICIES, EngineV1, NoControl
from crowdguard.simulation.runner import mean_ci, paired_change, run_grid, summarise
from crowdguard.simulation.scenarios import SCENARIOS, get_scenario


def tiny(demand, n=4, rate=1.0, area=50.0, end=600.0, max_time=1800.0, **kw) -> Scenario:
    return Scenario(
        name="tiny", title="tiny", description="",
        gates=tuple(GateSpec(f"G{i + 1}", rate, area) for i in range(n)),
        demand=demand, demand_end_s=end, max_time_s=max_time, radius_m=40.0, **kw,
    )


class TestPhysics:
    def test_walking_speed_falls_with_density(self):
        speeds = [weidmann_speed(d) for d in (0.0, 0.5, 1, 2, 3, 4, 5, 5.4, 7)]
        assert speeds == sorted(speeds, reverse=True)
        assert speeds[0] == pytest.approx(1.34)
        assert speeds[-1] == 0.0

    def test_throughput_degrades_only_when_overcrowded(self):
        assert throughput_factor(2.0) == 1.0
        assert throughput_factor(5.0) < 1.0
        assert throughput_factor(20.0) == pytest.approx(0.5)

    def test_density_saturates_at_the_physical_ceiling(self):
        assert area_density(10_000, 100) == MAX_DENSITY


class TestBookkeeping:
    @pytest.mark.parametrize("policy", list(POLICIES))
    @pytest.mark.parametrize("scenario", ["single_surge", "near_saturation"])
    def test_people_are_conserved(self, scenario, policy):
        m = Simulation(get_scenario(scenario), SimConfig(seed=3)).run(POLICIES[policy]()).metrics
        assert m["arrived"] == pytest.approx(m["served"] + m["remaining"], abs=1e-6)

    def test_light_load_never_builds_a_queue(self):
        sc = tiny(lambda t: np.full(4, 0.3))
        m = Simulation(sc, SimConfig(seed=1)).run(NoControl()).metrics
        assert m["peak_density"] < 0.2
        assert m["danger_person_s"] == 0.0

    def test_overload_without_control_becomes_dangerous(self):
        sc = tiny(lambda t: np.array([2.0, 0.3, 0.3, 0.3]))
        m = Simulation(sc, SimConfig(seed=1)).run(NoControl()).metrics
        assert m["peak_density"] >= DANGER_DENSITY
        assert m["danger_person_s"] > 0


class _AlwaysDivert:
    name = "always_divert"

    def reset(self, venue, clock):
        pass

    def decide(self, t, states, truth):
        return {"G1": Directive("divert", "G3", 1.0)}


class TestControl:
    def test_diversion_moves_the_compliant_share(self):
        sc = tiny(lambda t: np.array([1.0, 0, 0, 0]))
        cfg = SimConfig(seed=2, compliance=0.5, actuation_delay_s=0.0)
        m = Simulation(sc, cfg).run(_AlwaysDivert()).metrics
        assert m["diverted_share"] == pytest.approx(0.5, abs=0.06)
        assert m["extra_walk_m"] == pytest.approx(2 * np.pi * 40 / 4 * 2, rel=0.01)  # two hops

    def test_instructions_take_effect_after_the_actuation_delay(self):
        sc = tiny(lambda t: np.array([1.0, 0, 0, 0]), end=120.0)
        cfg = SimConfig(seed=2, compliance=1.0, actuation_delay_s=120.0)
        m = Simulation(sc, cfg).run(_AlwaysDivert()).metrics
        assert m["diverted_share"] == 0.0

    def test_engine_prevents_danger_in_a_single_surge(self):
        sc = get_scenario("single_surge")
        base = Simulation(sc, SimConfig(seed=0)).run(NoControl()).metrics
        eng = Simulation(sc, SimConfig(seed=0)).run(EngineV1()).metrics
        assert base["danger_person_s"] > 0
        assert eng["danger_person_s"] < base["danger_person_s"] * 0.1


class TestReproducibility:
    def test_same_seed_same_result(self):
        sc = get_scenario("twin_surge")
        a = Simulation(sc, SimConfig(seed=7)).run(EngineV1()).metrics
        b = Simulation(sc, SimConfig(seed=7)).run(EngineV1()).metrics
        assert a == b

    def test_policies_face_identical_arrivals(self):
        sc = get_scenario("stadium_entry")
        arrivals = {p: Simulation(sc, SimConfig(seed=5)).run(POLICIES[p]()).metrics["arrived"] for p in POLICIES}
        assert len(set(arrivals.values())) == 1

    @pytest.mark.parametrize("name", list(SCENARIOS))
    def test_every_scenario_builds_a_valid_venue(self, name):
        sc = get_scenario(name)
        venue = sc.venue()
        assert venue.gate_ids == sc.gate_ids
        assert sc.demand(0.0).shape == (len(sc.gates),)


class TestRunner:
    def test_grid_and_summary(self):
        rows = run_grid(["single_surge"], ["no_control", "engine_v1"], seeds=3, processes=1)
        assert len(rows) == 6
        summary = summarise(rows)
        assert {r["policy"] for r in summary} == {"no_control", "engine_v1"}
        mu, hw, rel = paired_change(rows, "single_surge", "danger_person_s", "engine_v1", "no_control")
        assert mu < 0 and rel < 0

    def test_mean_ci(self):
        mu, hw = mean_ci([1.0, 2.0, 3.0, 4.0])
        assert mu == 2.5 and 0 < hw < 3


def test_simulation_package_imports_no_model_code():
    """Like the decision layer, the simulator must never touch vision code."""
    forbidden = ("crowdguard.models", "crowdguard.ingestion", "torch", "cv2", "ultralytics")
    offenders = []
    for source in (Path(ROOT) / "crowdguard" / "simulation").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{source.name}: {n}" for n in names
                          if any(n == b or n.startswith(b + ".") for b in forbidden)]
    assert offenders == []
