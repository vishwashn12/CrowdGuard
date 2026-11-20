"""Temporal synchronisation between the density and motion streams.

The two model paths run at different rates on separate threads, so a density
result and a motion result almost never describe the same instant. This module
decides which motion observation may legitimately be paired with a given
density frame.

Both paths share the per-camera ``frame_id`` clock assigned by the producer,
which is what makes the comparison meaningful.

Rules:

1. **Same camera.** Observations are never mixed across cameras.
2. **No future motion.** A motion result whose ``frame_id`` exceeds the density
   frame's describes a moment that had not happened yet; pairing it would let a
   later surge justify an earlier decision.
3. **Historical reuse.** With no exact match, the most recent earlier result is
   reused and marked ``LAST_KNOWN``.
4. **Staleness.** Once the frame gap exceeds the threshold the motion is marked
   ``STALE`` and excluded from scoring.
5. **Unavailable.** With no usable history at all, motion is ``UNAVAILABLE``.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum

from crowdguard.models.contracts import CSRNetResult, MotionResult


class MotionFreshness(str, Enum):
    """Freshness of a motion observation relative to its density frame."""

    NEW = "NEW"
    LAST_KNOWN = "LAST_KNOWN"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SyncResult:
    """Outcome of pairing a density frame with a motion observation."""

    motion_result: MotionResult | None
    status: MotionFreshness
    age_frames: int
    is_valid_motion: bool
    provenance: str | None
    reason: str

    @property
    def display_tag(self) -> str:
        if self.status is MotionFreshness.NEW:
            return "[NEW]"
        if self.status is MotionFreshness.LAST_KNOWN:
            return f"[LAST KNOWN +{self.age_frames}f]"
        if self.status is MotionFreshness.STALE:
            return f"[STALE +{self.age_frames}f - EXCLUDED]"
        return "[UNAVAILABLE]"


class TemporalSynchronizer:
    """Thread-safe, per-camera history of motion results.

    Motion results arrive from the pose worker while density results arrive from
    the CSRNet worker, so every method here is guarded.
    """

    def __init__(
        self,
        max_history_per_camera: int = 50,
        stale_frame_gap: int = 10,
    ) -> None:
        """
        Args:
            max_history_per_camera: Motion results retained per camera.
            stale_frame_gap: Frame gap beyond which motion counts as stale.
        """
        self._max_history = max_history_per_camera
        self._stale_frame_gap = stale_frame_gap
        self._history: dict[str, deque[MotionResult]] = {}
        self._lock = threading.Lock()

        self.total_queries = 0
        self.new_count = 0
        self.last_known_count = 0
        self.stale_count = 0
        self.unavailable_count = 0

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_motion_result(self, result: MotionResult) -> None:
        """Record a freshly computed motion result."""
        with self._lock:
            history = self._history.get(result.camera_id)
            if history is None:
                history = deque(maxlen=self._max_history)
                self._history[result.camera_id] = history
            history.append(result)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def synchronize(
        self,
        csrnet_result: CSRNetResult,
        stale_frame_gap: int | None = None,
    ) -> SyncResult:
        """Find the newest admissible motion result for a density frame."""
        threshold = stale_frame_gap if stale_frame_gap is not None else self._stale_frame_gap
        camera_id = csrnet_result.camera_id
        density_frame = csrnet_result.frame_id

        with self._lock:
            self.total_queries += 1
            history = list(self._history.get(camera_id, ()))

        if not history:
            with self._lock:
                self.unavailable_count += 1
            return SyncResult(
                motion_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                provenance=None,
                reason=f"No motion history for camera {camera_id}",
            )

        # Newest result that does not postdate the density frame.
        candidate: MotionResult | None = None
        for result in reversed(history):
            if result.frame_id <= density_frame:
                candidate = result
                break

        if candidate is None:
            with self._lock:
                self.unavailable_count += 1
            earliest = history[0].frame_id
            return SyncResult(
                motion_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                provenance=None,
                reason=(
                    f"All motion history for {camera_id} postdates density frame "
                    f"{density_frame} (earliest held: {earliest}). Using future motion "
                    f"to justify a past decision is not permitted."
                ),
            )

        age = density_frame - candidate.frame_id
        provenance = f"motion@frame_{candidate.frame_id}"

        if age > threshold:
            with self._lock:
                self.stale_count += 1
            return SyncResult(
                motion_result=candidate,
                status=MotionFreshness.STALE,
                age_frames=age,
                is_valid_motion=False,
                provenance=provenance,
                reason=(
                    f"Motion for {camera_id} is {age} frames behind density frame "
                    f"{density_frame}, over the {threshold}-frame limit. Excluded from scoring."
                ),
            )

        if age == 0:
            with self._lock:
                self.new_count += 1
            return SyncResult(
                motion_result=candidate,
                status=MotionFreshness.NEW,
                age_frames=0,
                is_valid_motion=True,
                provenance=provenance,
                reason=f"Motion and density both describe frame {density_frame}",
            )

        with self._lock:
            self.last_known_count += 1
        return SyncResult(
            motion_result=candidate,
            status=MotionFreshness.LAST_KNOWN,
            age_frames=age,
            is_valid_motion=True,
            provenance=provenance,
            reason=(
                f"Reusing motion from frame {candidate.frame_id}, {age} frames before "
                f"density frame {density_frame} (within the {threshold}-frame limit)"
            ),
        )

    @staticmethod
    def validate_single_pair(
        csrnet_result: CSRNetResult | None,
        motion_result: MotionResult | None,
        stale_frame_gap: int = 10,
    ) -> SyncResult:
        """Validate one explicit pair, without consulting any history.

        Used when a caller already holds both results, and by the tests.
        """
        if motion_result is None:
            return SyncResult(
                motion_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                provenance=None,
                reason="No motion result supplied",
            )

        if csrnet_result is None:
            # Motion only: there is no density frame to be stale against.
            return SyncResult(
                motion_result=motion_result,
                status=MotionFreshness.NEW,
                age_frames=0,
                is_valid_motion=True,
                provenance=f"motion@frame_{motion_result.frame_id}",
                reason="Motion-only observation; no density frame to synchronise against",
            )

        if csrnet_result.camera_id != motion_result.camera_id:
            return SyncResult(
                motion_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                provenance=None,
                reason=(
                    f"Camera mismatch: density from {csrnet_result.camera_id}, "
                    f"motion from {motion_result.camera_id}"
                ),
            )

        age = csrnet_result.frame_id - motion_result.frame_id

        if age < 0:
            return SyncResult(
                motion_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                provenance=None,
                reason=(
                    f"Motion frame {motion_result.frame_id} postdates density frame "
                    f"{csrnet_result.frame_id}"
                ),
            )

        if age > stale_frame_gap:
            return SyncResult(
                motion_result=motion_result,
                status=MotionFreshness.STALE,
                age_frames=age,
                is_valid_motion=False,
                provenance=f"motion@frame_{motion_result.frame_id}",
                reason=f"Motion is {age} frames stale, over the {stale_frame_gap}-frame limit",
            )

        status = MotionFreshness.NEW if age == 0 else MotionFreshness.LAST_KNOWN
        return SyncResult(
            motion_result=motion_result,
            status=status,
            age_frames=age,
            is_valid_motion=True,
            provenance=f"motion@frame_{motion_result.frame_id}",
            reason=f"Motion accepted at {age} frames behind density",
        )

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def forget_camera(self, camera_id: str) -> None:
        with self._lock:
            self._history.pop(camera_id, None)

    def reset_counters(self) -> None:
        """Zero the freshness tallies, keeping the per-camera history."""
        with self._lock:
            self.total_queries = 0
            self.new_count = 0
            self.last_known_count = 0
            self.stale_count = 0
            self.unavailable_count = 0

    def stats(self) -> dict:
        with self._lock:
            total = max(1, self.total_queries)
            return {
                "total_queries": self.total_queries,
                "new": self.new_count,
                "last_known": self.last_known_count,
                "stale": self.stale_count,
                "unavailable": self.unavailable_count,
                "valid_ratio": round((self.new_count + self.last_known_count) / total, 3),
                "cameras_tracked": list(self._history.keys()),
            }
