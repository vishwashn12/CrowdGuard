"""Configuration for the Decision Engine.

Every threshold and weight lives here. No dependency on torch, cv2, or any
vision model code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CongestionThresholds(BaseModel):
    """Cut points mapping ``congestion_score`` to a :class:`CongestionLevel`.

    LOW      score < moderate_threshold
    MODERATE moderate_threshold <= score < high_threshold
    HIGH     high_threshold <= score < critical_threshold
    CRITICAL score >= critical_threshold
    """

    moderate_threshold: float = Field(default=0.4, ge=0, le=1)
    high_threshold: float = Field(default=0.65, ge=0, le=1)
    critical_threshold: float = Field(default=0.8, ge=0, le=1)


class CongestionWeights(BaseModel):
    """Weights for composing ``congestion_score`` from an observation.

    ``congestion_score = w_density * density_score
                       + w_velocity * (1 - clamp(velocity / max_expected_velocity))
                       + w_capacity * capacity_utilization``

    This formula is a transparent, tunable heuristic. It has not been validated
    against ground-truth crush incidents, and should be calibrated per venue
    before being trusted operationally.
    """

    w_density: float = Field(default=0.5, ge=0, le=1)
    w_velocity: float = Field(default=0.2, ge=0, le=1)
    w_capacity: float = Field(default=0.3, ge=0, le=1)
    max_expected_velocity: float = Field(
        default=75.0, gt=0,
        description="Velocity (px/s) at or above which the inverse-velocity term "
                    "reaches 0, i.e. a freely flowing crowd. "
                    "MUST BE CALIBRATED PER CAMERA: this is a pixel rate, so it "
                    "scales with resolution, zoom and how far the camera sits from "
                    "the crowd. The default is measured from freely flowing crowd "
                    "in 1440x1080 footage, which ran at 70-78 px/s. Set it too low "
                    "and the term saturates at zero for every observation, which "
                    "silently reduces congestion scoring to density alone; the "
                    "VELOCITY_UNCALIBRATED advisory warns when that is happening.",
    )


class RoutingWeights(BaseModel):
    """Weights for scoring candidate diversion targets.

    ``routing_score = w_congestion * (1 - candidate_congestion_score)
                    + w_capacity   * (1 - capacity_utilization)
                    + w_distance   * (1 - normalised_distance)
                    + w_trend      * trend_bonus``

    Higher is better.
    """

    w_congestion: float = Field(default=0.35, ge=0, le=1)
    w_capacity: float = Field(default=0.30, ge=0, le=1)
    w_distance: float = Field(default=0.20, ge=0, le=1)
    w_trend: float = Field(default=0.15, ge=0, le=1)


class HysteresisConfig(BaseModel):
    """Stability controls that stop diversion targets oscillating."""

    cooldown_seconds: float = Field(
        default=10.0, ge=0,
        description="Minimum dwell time before a source gate may change target",
    )
    switching_margin: float = Field(
        default=0.15, ge=0, le=1,
        description="A challenger must beat the incumbent target by this routing-score "
                    "margin to take over during the cooldown",
    )


class DecisionConfig(BaseModel):
    """Complete Decision Engine configuration."""

    congestion_thresholds: CongestionThresholds = Field(default_factory=CongestionThresholds)
    congestion_weights: CongestionWeights = Field(default_factory=CongestionWeights)
    routing_weights: RoutingWeights = Field(default_factory=RoutingWeights)
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)

    divert_above_level: str = Field(
        default="HIGH",
        description="Minimum congestion level that triggers diversion evaluation",
    )

    max_observation_age_s: float = Field(
        default=5.0, gt=0,
        description="Age beyond which an observation is too stale to act on",
    )
    min_confidence: float = Field(
        default=0.3, ge=0, le=1,
        description="Minimum observation confidence required to act",
    )

    max_candidate_congestion: float = Field(
        default=0.75, ge=0, le=1,
        description="Congestion score above which a gate is rejected as a target",
    )
    min_capacity_remaining_ratio: float = Field(
        default=0.15, ge=0, le=1,
        description="Minimum (1 - capacity_utilization) for a candidate to be viable",
    )

    unmonitored_target_penalty: float = Field(
        default=0.15, ge=0, le=1,
        description="Routing-score penalty applied to a gate with no camera. Such a "
                    "gate is still a legal target - its capacity is known - but it is "
                    "preferred less than an observed gate of equal standing.",
    )
    allow_unmonitored_targets: bool = Field(
        default=True,
        description="Permit diversion to gates that have no camera bound. Disable to "
                    "require live observation of every target.",
    )
    assumed_unmonitored_utilization: float = Field(
        default=0.3, ge=0, le=1,
        description="Capacity utilisation assumed for an unobserved gate. Deliberately "
                    "non-zero: assuming an unseen gate is empty would make it the most "
                    "attractive target in the venue.",
    )

    calibrated_capacity_density: float = Field(
        default=5.0, gt=0,
        description="People per square metre at which a calibrated zone counts as full. "
                    "A calibrated gate's capacity is its zone area times this. 5/m^2 is "
                    "where crowd-safety guidance places serious crush risk.",
    )
    free_walking_speed_ms: float = Field(
        default=1.34, gt=0,
        description="Walking speed of a freely flowing crowd in m/s (Weidmann). For "
                    "calibrated gates this replaces max_expected_velocity, which is in px/s.",
    )

    trend_window_size: int = Field(
        default=5, ge=2,
        description="Observations retained per gate for trend calculation",
    )
    trend_change_threshold: float = Field(
        default=0.05, ge=0,
        description="Minimum score delta between window halves to call a trend",
    )
