"""Feature fusion: build a model-agnostic CrowdState from raw observations.

Computes ``congestion_score`` and ``congestion_level`` from raw density,
velocity and capacity. Imports nothing from torch, cv2, CSRNet or YOLO.

The scoring formula is documented on
:class:`crowdguard.decision.config.CongestionWeights`.
"""

from __future__ import annotations

import time

from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import (
    CongestionLevel,
    CrowdState,
    MotionStatus,
    Trend,
)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def classify_congestion(score: float, config: DecisionConfig) -> CongestionLevel:
    """Map a congestion score to a level using the configured thresholds."""
    thresholds = config.congestion_thresholds
    if score >= thresholds.critical_threshold:
        return CongestionLevel.CRITICAL
    if score >= thresholds.high_threshold:
        return CongestionLevel.HIGH
    if score >= thresholds.moderate_threshold:
        return CongestionLevel.MODERATE
    return CongestionLevel.LOW


def compute_congestion_score(
    density_score: float,
    velocity: float,
    capacity_utilization: float,
    config: DecisionConfig,
    has_motion: bool = True,
    max_expected_velocity: float | None = None,
) -> float:
    """Compose a congestion score in [0, 1] from an observation.

    When motion is unavailable the velocity term is dropped and the remaining
    weights are renormalised. Substituting zero velocity would invent congestion
    that was never observed; substituting maximum velocity would invent freedom.

    Args:
        density_score: Normalised density in [0, 1].
        velocity: Crowd velocity magnitude in px/s.
        capacity_utilization: count/capacity clamped to [0, 1].
        config: Weights and thresholds.
        has_motion: Whether a valid, fresh motion observation exists.
        max_expected_velocity: Free-flow speed in the same unit as ``velocity``.
            Defaults to the configured px/s value; calibrated gates pass m/s.
    """
    w = config.congestion_weights

    if not has_motion:
        weight_sum = w.w_density + w.w_capacity
        if weight_sum <= 0:
            return _clamp01(density_score)
        raw = (
            w.w_density * _clamp01(density_score)
            + w.w_capacity * _clamp01(capacity_utilization)
        ) / weight_sum
        return _clamp01(raw)

    # Faster crowd means less congestion: the crowd is still flowing.
    free_flow = max_expected_velocity or w.max_expected_velocity
    inv_velocity = 1.0 - _clamp01(velocity / free_flow)

    raw = (
        w.w_density * _clamp01(density_score)
        + w.w_velocity * inv_velocity
        + w.w_capacity * _clamp01(capacity_utilization)
    )

    weight_sum = w.w_density + w.w_velocity + w.w_capacity
    if weight_sum > 0:
        raw /= weight_sum

    return _clamp01(raw)


def build_crowd_state(
    gate_id: str,
    gate_capacity: int,
    crowd_count: float = 0.0,
    velocity: float = 0.0,
    direction: float = 0.0,
    flow_consistency: float = 0.0,
    confidence: float = 1.0,
    source_frame_id: int | None = None,
    result_age_s: float = 0.0,
    has_density: bool = True,
    has_motion: bool = True,
    motion_status: MotionStatus = MotionStatus.OK,
    motion_provenance: str | None = None,
    motion_age_frames: int = 0,
    trend: Trend = Trend.STABLE,
    config: DecisionConfig | None = None,
    timestamp: float | None = None,
    is_synthetic: bool = False,
    area_m2: float | None = None,
) -> CrowdState:
    """Assemble a complete :class:`CrowdState` from raw observation values.

    When ``area_m2`` is given the gate is calibrated: ``crowd_count`` is the
    count inside the ground zone, ``velocity`` is in m/s, and capacity is the
    zone area times ``calibrated_capacity_density`` (``gate_capacity`` is then
    ignored).

    Derives ``density_score``, ``congestion_score`` and ``congestion_level``,
    and degrades ``confidence`` when either half of the observation is missing.

    Motion fields are zeroed whenever motion is invalid, so a stale reading can
    never be mistaken downstream for a genuine measurement of a still crowd.
    """
    if config is None:
        config = DecisionConfig()
    if timestamp is None:
        timestamp = time.time()

    calibrated = area_m2 is not None and area_m2 > 0
    if calibrated:
        gate_capacity = max(1, int(round(area_m2 * config.calibrated_capacity_density)))
    density_score = _clamp01(crowd_count / gate_capacity) if gate_capacity > 0 else 0.0
    capacity_utilization = density_score

    motion_valid = has_motion and motion_status == MotionStatus.OK
    effective_velocity = max(0.0, velocity) if motion_valid else 0.0
    effective_direction = (direction % 360.0) if motion_valid else 0.0
    effective_flow = _clamp01(flow_consistency) if motion_valid else 0.0

    congestion_score = compute_congestion_score(
        density_score=density_score,
        velocity=effective_velocity,
        capacity_utilization=capacity_utilization,
        config=config,
        has_motion=motion_valid,
        max_expected_velocity=config.free_walking_speed_ms if calibrated else None,
    )

    effective_confidence = _clamp01(confidence)
    if not has_density:
        effective_confidence *= 0.5
    if not motion_valid:
        effective_confidence *= 0.7

    return CrowdState(
        gate_id=gate_id,
        timestamp=timestamp,
        crowd_count=max(0.0, crowd_count),
        density_score=density_score,
        velocity=effective_velocity,
        direction=effective_direction,
        flow_consistency=effective_flow,
        congestion_score=congestion_score,
        congestion_level=classify_congestion(congestion_score, config),
        trend=trend,
        confidence=effective_confidence,
        source_frame_id=source_frame_id,
        result_age_s=max(0.0, result_age_s),
        has_density=has_density,
        has_motion=motion_valid,
        motion_status=motion_status,
        motion_provenance=motion_provenance,
        motion_age_frames=motion_age_frames,
        is_synthetic=is_synthetic,
        velocity_unit="m/s" if calibrated else "px/s",
        area_m2=area_m2 if calibrated else None,
    )


def calibrated_capacity(area_m2: float, config: DecisionConfig | None = None) -> int:
    """Capacity of a calibrated zone: its area at the 'full' density."""
    config = config or DecisionConfig()
    return max(1, int(round(area_m2 * config.calibrated_capacity_density)))


def build_unmonitored_state(
    gate_id: str,
    gate_capacity: int,
    config: DecisionConfig | None = None,
    trend: Trend = Trend.STABLE,
    timestamp: float | None = None,
) -> CrowdState:
    """Build a placeholder state for a gate that has no camera.

    The gate has declared capacity but no observation, so a neutral,
    deliberately non-empty utilisation is assumed (see
    ``DecisionConfig.assumed_unmonitored_utilization``). Both ``has_density``
    and ``has_motion`` are false, which drops confidence and keeps the engine
    honest about what it does and does not know.
    """
    if config is None:
        config = DecisionConfig()

    assumed_util = config.assumed_unmonitored_utilization
    return build_crowd_state(
        gate_id=gate_id,
        gate_capacity=gate_capacity,
        crowd_count=assumed_util * gate_capacity,
        velocity=0.0,
        confidence=0.5,
        has_density=False,
        has_motion=False,
        motion_status=MotionStatus.UNAVAILABLE,
        motion_provenance="no camera bound to this gate",
        trend=trend,
        config=config,
        timestamp=timestamp,
    )
