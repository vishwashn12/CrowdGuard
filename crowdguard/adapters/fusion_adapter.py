"""The model-to-decision boundary.

**This is the only module in the project that imports both model-specific result
types and model-agnostic decision types.** If CSRNet or the pose model is
swapped out, this file is the only one that has to change: the decision engine,
venue topology, routing and alerting are all written against
:class:`CrowdState` and never see a model type.

Dependency direction::

    CSRNetResult / MotionResult   (models)
                 |
            FusionAdapter        (here)
                 |
             CrowdState          (model-agnostic)
                 |
          DecisionEngine         (no model imports)

Anomaly results deliberately do **not** pass through here. Per-person
behaviour travels on its own channel to ``crowdguard.alerts``, keeping
congestion routing a pure function of density, flow and capacity.
"""

from __future__ import annotations

from crowdguard.adapters.synchronizer import MotionFreshness, SyncResult, TemporalSynchronizer
from crowdguard.decision.config import DecisionConfig
from crowdguard.decision.contracts import CrowdState, MotionStatus, Trend
from crowdguard.decision.fusion import build_crowd_state, build_unmonitored_state

# The only model-layer imports permitted anywhere in the decision half.
from crowdguard.models.contracts import CSRNetResult, MotionResult
from crowdguard.utils.calibration import GroundCalibration

_FRESHNESS_TO_STATUS = {
    MotionFreshness.NEW: MotionStatus.OK,
    MotionFreshness.LAST_KNOWN: MotionStatus.OK,
    MotionFreshness.STALE: MotionStatus.STALE,
    MotionFreshness.UNAVAILABLE: MotionStatus.UNAVAILABLE,
}


class FusionAdapter:
    """Fuses a density result and a motion result into one :class:`CrowdState`.

    Handles missing density, missing or stale motion, camera/gate resolution,
    and refuses to mix observations from different cameras.
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self._config = config or DecisionConfig()

    @property
    def config(self) -> DecisionConfig:
        return self._config

    def fuse(
        self,
        csrnet_result: CSRNetResult | None,
        motion_result: MotionResult | None,
        gate_id: str,
        gate_capacity: int,
        trend: Trend = Trend.STABLE,
        stale_frame_gap: int = 10,
        sync_result: SyncResult | None = None,
        now: float | None = None,
        calibration: GroundCalibration | None = None,
    ) -> CrowdState:
        """Build a :class:`CrowdState` for one gate.

        Args:
            csrnet_result: Density result, or None when unavailable.
            motion_result: Motion result, or None when unavailable.
            gate_id: Gate identifier from the venue config.
            gate_capacity: Declared capacity of this gate.
            trend: Pre-computed trend from the tracker.
            stale_frame_gap: Frame gap past which motion is excluded.
            sync_result: A pre-computed pairing, if the caller already has one.
            now: Wall-clock override, for deterministic tests.
            calibration: Ground calibration of this gate's camera. When given,
                people are counted only inside the calibrated zone, speed is
                the zone's walking speed in m/s, and capacity follows from the
                zone's area. The zone count is the larger of the density map's
                masked sum and the number of tracked people standing in the
                zone: CSRNet (trained on dense, distant crowds) under-counts
                large people near the camera by about 3x on this footage,
                while the detector misses small distant ones. Taking the
                larger errs on the side of safety.

        Raises:
            ValueError: If the two results come from different cameras.
        """
        import time

        current_time = time.time() if now is None else now

        if csrnet_result is not None and motion_result is not None:
            if csrnet_result.camera_id != motion_result.camera_id:
                raise ValueError(
                    f"Refusing to fuse observations from different cameras: "
                    f"density={csrnet_result.camera_id}, motion={motion_result.camera_id}"
                )

        # --- density half ---
        has_density = csrnet_result is not None
        crowd_count = 0.0
        source_frame_id: int | None = None
        result_age_s = 0.0
        timestamp = current_time

        if csrnet_result is not None:
            crowd_count = csrnet_result.crowd_count
            source_frame_id = csrnet_result.frame_id
            result_age_s = max(0.0, current_time - csrnet_result.result_timestamp)
            timestamp = csrnet_result.result_timestamp

        # --- motion half ---
        if sync_result is None:
            sync_result = TemporalSynchronizer.validate_single_pair(
                csrnet_result=csrnet_result,
                motion_result=motion_result,
                stale_frame_gap=stale_frame_gap,
            )

        motion = sync_result.motion_result
        motion_valid = sync_result.is_valid_motion
        motion_status = _FRESHNESS_TO_STATUS.get(sync_result.status, MotionStatus.UNAVAILABLE)

        area_m2: float | None = None
        provenance = sync_result.provenance
        if calibration is not None:
            area_m2 = calibration.area_m2
            if csrnet_result is not None:
                crowd_count = calibration.count_in_zone(csrnet_result.density_map)
            # The tracked head-count is usable even when the motion result is a
            # little late for speed purposes: people do not vanish in a stride.
            if motion is not None:
                crowd_count = max(crowd_count, float(motion.zone_track_count))
            zone_speed = motion.median_speed_ms if (motion_valid and motion) else None
            if zone_speed is None and motion_valid:
                # Motion exists, but too few people are in the zone to measure
                # their walking speed. Score on density alone rather than
                # inventing a speed.
                motion_valid = False
                motion_status = MotionStatus.UNAVAILABLE
                provenance = "too few people in the calibrated zone to measure speed"

        velocity = motion.median_velocity if (motion_valid and motion) else 0.0
        if calibration is not None and motion_valid and motion is not None:
            velocity = float(motion.median_speed_ms or 0.0)
        direction = motion.dominant_direction_deg if (motion_valid and motion) else 0.0
        flow_consistency = motion.flow_consistency if (motion_valid and motion) else 0.0

        confidence = self._blend_confidence(
            has_density=has_density,
            motion_valid=motion_valid,
            motion=motion,
            freshness=sync_result.status,
        )

        return build_crowd_state(
            gate_id=gate_id,
            gate_capacity=gate_capacity,
            crowd_count=crowd_count,
            velocity=velocity,
            direction=direction,
            flow_consistency=flow_consistency,
            confidence=confidence,
            source_frame_id=source_frame_id,
            result_age_s=result_age_s,
            has_density=has_density,
            has_motion=motion_valid,
            motion_status=motion_status,
            motion_provenance=provenance,
            motion_age_frames=sync_result.age_frames,
            trend=trend,
            config=self._config,
            timestamp=timestamp,
            area_m2=area_m2,
        )

    @staticmethod
    def _blend_confidence(
        has_density: bool,
        motion_valid: bool,
        motion: MotionResult | None,
        freshness: MotionFreshness,
    ) -> float:
        """Combine the confidence of both halves of an observation.

        Density from a trained CSRNet checkpoint is treated as fully confident;
        motion confidence is reported by the model itself. With both present the
        two are averaged, so one weak half cannot carry a decision alone.
        """
        if motion_valid and motion is not None:
            confidence = motion.motion_confidence
            if has_density:
                return min(1.0, (confidence + 1.0) / 2.0)
            return confidence

        if freshness is MotionFreshness.STALE and motion is not None:
            return motion.motion_confidence * 0.7

        return 0.8 if has_density else 0.5

    def unmonitored_state(
        self,
        gate_id: str,
        gate_capacity: int,
        trend: Trend = Trend.STABLE,
    ) -> CrowdState:
        """Placeholder state for a gate with no camera bound.

        Gates without cameras still take part in routing: their capacity is
        known even though their occupancy is not. See
        :func:`crowdguard.decision.fusion.build_unmonitored_state`.
        """
        return build_unmonitored_state(
            gate_id=gate_id,
            gate_capacity=gate_capacity,
            config=self._config,
            trend=trend,
        )
