"""Shared pytest fixtures.

Puts the project root on ``sys.path`` so the tests run from a checkout without
installing the package.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdguard.config.schemas import AnomalyConfig
from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import CongestionLevel, CrowdState, MotionStatus, Trend
from crowdguard.decision.fusion import build_crowd_state
from crowdguard.venue.models import GateConfig, GeometryConfig, VenueConfig
from crowdguard.venue.topology import VenueTopology


@pytest.fixture
def decision_config() -> DecisionConfig:
    return DecisionConfig()


@pytest.fixture
def anomaly_config() -> AnomalyConfig:
    return AnomalyConfig()


def make_venue(gate_count: int = 4, monitored: int | None = None, radius_m: float = 100.0) -> VenueTopology:
    """Build a ring venue where the first ``monitored`` gates have cameras."""
    if monitored is None:
        monitored = gate_count

    gates = [
        GateConfig(
            gate_id=f"G{i + 1}",
            name=f"Gate {i + 1}",
            capacity=500,
            position_index=i,
            camera_id=f"cam_{i + 1}" if i < monitored else None,
        )
        for i in range(gate_count)
    ]
    config = VenueConfig(
        id="test_venue",
        name="Test Venue",
        geometry=GeometryConfig(type="circular", radius_m=radius_m),
        gates=gates,
        auto_circular_connections=True,
    )
    return VenueTopology(config)


@pytest.fixture
def venue() -> VenueTopology:
    return make_venue(4)


def make_state(
    gate_id: str,
    crowd_count: float = 100.0,
    capacity: int = 500,
    velocity: float = 10.0,
    trend: Trend = Trend.STABLE,
    confidence: float = 1.0,
    result_age_s: float = 0.0,
    has_motion: bool = True,
    config: DecisionConfig | None = None,
) -> CrowdState:
    """Build a CrowdState through the real fusion path, not by hand.

    Going through ``build_crowd_state`` keeps the derived fields - score, level,
    degraded confidence - consistent with production rather than asserting on
    combinations the system could never actually produce.
    """
    return build_crowd_state(
        gate_id=gate_id,
        gate_capacity=capacity,
        crowd_count=crowd_count,
        velocity=velocity,
        confidence=confidence,
        result_age_s=result_age_s,
        has_motion=has_motion,
        motion_status=MotionStatus.OK if has_motion else MotionStatus.UNAVAILABLE,
        trend=trend,
        config=config or DecisionConfig(),
        timestamp=time.time(),
    )


@pytest.fixture
def state_factory():
    return make_state


@pytest.fixture
def venue_factory():
    return make_venue


def saturated_state(gate_id: str, config: DecisionConfig | None = None) -> CrowdState:
    """A gate that is unambiguously CRITICAL: full and barely moving."""
    state = make_state(
        gate_id, crowd_count=490.0, capacity=500, velocity=0.5, config=config
    )
    assert state.congestion_level is CongestionLevel.CRITICAL, (
        f"fixture expected CRITICAL, produced {state.congestion_level}"
    )
    return state


def congested_flowing_state(gate_id: str, config: DecisionConfig | None = None) -> CrowdState:
    """A gate that is CRITICAL on density but still physically moving.

    Useful for exercising the diversion rules on their own: the crush-risk rule
    fires first in the exclusive congestion chain and would otherwise mask them,
    which is correct behaviour but makes those rules untestable in isolation.
    """
    state = make_state(
        gate_id, crowd_count=480.0, capacity=500, velocity=8.0, config=config
    )
    assert state.congestion_level is CongestionLevel.CRITICAL, (
        f"fixture expected CRITICAL, produced {state.congestion_level}"
    )
    return state


def quiet_state(gate_id: str, config: DecisionConfig | None = None) -> CrowdState:
    """A gate with plenty of headroom and a freely flowing crowd."""
    state = make_state(
        gate_id, crowd_count=40.0, capacity=500, velocity=14.0, config=config
    )
    assert state.congestion_level is CongestionLevel.LOW, (
        f"fixture expected LOW, produced {state.congestion_level}"
    )
    return state
