"""The 10-dimensional feature vector consumed by the anomaly classifier.

This math is reproduced exactly from the training-time extractor
(``realtime/extract_features.py``). The classifier in ``assets/best_model.pkl``
was fitted on these features, in this order, computed this way. Any change here
silently shifts the input distribution and degrades predictions without raising
an error, so treat this module as frozen unless the model is retrained
alongside it.

Feature order, matching ``FEATURE_COLS`` in the training script:

===  ====================  ==============================================
idx  name                  meaning
===  ====================  ==============================================
0    displacement          net pixel distance across the tracked window
1    direction_sin         sin of absolute heading
2    direction_cos         cos of absolute heading
3    cos_sim               alignment with the local crowd flow
4    straightness          displacement / total path length
5    aspect_ratio          bbox width / height
6    pos_y_norm            vertical frame position (pseudo-depth)
7    torso_height_ratio    (hip_y - shoulder_y) / bbox height
8    head_to_hip_ratio     (hip_y - head_y) / bbox height
9    knee_bend_angle       hip-knee-ankle angle in degrees
===  ====================  ==============================================

Indices 7-9 are the skeletal features. Without them the sitting class collapses
(F1 0.143 -> 0.033 in the original ablation), because a stationary standing
person and a seated person are nearly identical as 2D boxes.
"""

from __future__ import annotations

import numpy as np

from crowdguard.utils.geometry import joint_angle

FEATURE_NAMES: tuple[str, ...] = (
    "displacement",
    "direction_sin",
    "direction_cos",
    "cos_sim",
    "straightness",
    "aspect_ratio",
    "pos_y_norm",
    "torso_height_ratio",
    "head_to_hip_ratio",
    "knee_bend_angle",
)

FEATURE_COUNT = len(FEATURE_NAMES)

# COCO-17 keypoint indices used by YOLO pose models.
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Keypoint confidence gates, as used at training time.
_TORSO_CONF = 0.5
_LEG_CONF = 0.3

# Returned when the legs are not visible: a straight leg, i.e. "not sitting".
_DEFAULT_KNEE_ANGLE = 180.0


def _keypoint(keypoints: np.ndarray, index: int) -> tuple[float, float, float]:
    """Return ``(x, y, confidence)`` for a keypoint row.

    Pose arrays come as either ``(17, 3)`` with confidence or ``(17, 2)``
    without; a missing confidence column is treated as fully confident.
    """
    row = keypoints[index]
    if keypoints.shape[1] == 3:
        return float(row[0]), float(row[1]), float(row[2])
    return float(row[0]), float(row[1]), 1.0


def _mean_visible_y(
    keypoints: np.ndarray,
    left_index: int,
    right_index: int,
    min_conf: float = _TORSO_CONF,
) -> float | None:
    """Average y of a symmetric keypoint pair, using whichever side is visible."""
    _, left_y, left_c = _keypoint(keypoints, left_index)
    _, right_y, right_c = _keypoint(keypoints, right_index)

    if left_c > min_conf and right_c > min_conf:
        return (left_y + right_y) / 2.0
    if left_c > min_conf:
        return left_y
    if right_c > min_conf:
        return right_y
    return None


def compute_pose_features(
    keypoints: np.ndarray | None,
    bbox_height: float,
) -> tuple[float, float, float]:
    """Compute the three skeletal features.

    Returns:
        ``(torso_height_ratio, head_to_hip_ratio, knee_bend_angle)``. Falls back
        to ``(0.0, 0.0, 180.0)`` for any part that is not visible, which is the
        same default the training extractor used.
    """
    torso_height_ratio = 0.0
    head_to_hip_ratio = 0.0
    knee_bend_angle = _DEFAULT_KNEE_ANGLE

    if keypoints is None or len(keypoints) == 0 or bbox_height <= 0:
        return torso_height_ratio, head_to_hip_ratio, knee_bend_angle

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] <= KP_RIGHT_ANKLE:
        return torso_height_ratio, head_to_hip_ratio, knee_bend_angle

    _, head_y, head_c = _keypoint(keypoints, KP_NOSE)
    shoulder_y = _mean_visible_y(keypoints, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    hip_y = _mean_visible_y(keypoints, KP_LEFT_HIP, KP_RIGHT_HIP)

    # Seated posture compresses the upper body: the torso takes up a larger
    # share of the box, and the head sits closer to the hips.
    if shoulder_y is not None and hip_y is not None:
        torso_height_ratio = max(0.0, hip_y - shoulder_y) / bbox_height
    if head_c > _TORSO_CONF and hip_y is not None:
        head_to_hip_ratio = max(0.0, hip_y - head_y) / bbox_height

    # Knee flexion is the strongest single sitting cue: ~180 deg standing,
    # dropping toward 90 deg when seated or squatting.
    angles: list[float] = []
    for hip_idx, knee_idx, ankle_idx in (
        (KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE),
        (KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
    ):
        hip = _keypoint(keypoints, hip_idx)
        knee = _keypoint(keypoints, knee_idx)
        ankle = _keypoint(keypoints, ankle_idx)
        if hip[2] > _LEG_CONF and knee[2] > _LEG_CONF and ankle[2] > _LEG_CONF:
            angles.append(joint_angle(hip[:2], knee[:2], ankle[:2]))

    if angles:
        knee_bend_angle = sum(angles) / len(angles)

    return torso_height_ratio, head_to_hip_ratio, knee_bend_angle


def build_feature_vector(
    positions: list[tuple[float, float]],
    step_speeds: list[float],
    direction_rad: float,
    bbox: tuple[float, float, float, float],
    keypoints: np.ndarray | None,
    crowd_direction_rad: float,
    crowd_direction_valid: bool,
    frame_height: int,
) -> np.ndarray:
    """Assemble the 10-D feature vector for one tracked person.

    Args:
        positions: Track centre history, oldest first.
        step_speeds: Per-frame step distances for this track.
        direction_rad: Current heading, radians.
        bbox: Current ``(x1, y1, x2, y2)``.
        keypoints: Pose keypoints as ``(17, 2|3)``, or None.
        crowd_direction_rad: Smoothed dominant crowd heading.
        crowd_direction_valid: Whether a crowd direction could be established.
        frame_height: Frame height, for normalising vertical position.

    Returns:
        ``float32`` array of shape ``(10,)``.
    """
    # -- motion ------------------------------------------------------
    if len(positions) >= 2:
        displacement = float(np.hypot(
            positions[-1][0] - positions[0][0],
            positions[-1][1] - positions[0][1],
        ))
    else:
        displacement = 0.0

    direction_sin = float(np.sin(direction_rad))
    direction_cos = float(np.cos(direction_rad))

    # Zero rather than 1.0 when no flow exists: an unknown alignment must not
    # read as "perfectly with the crowd".
    cos_sim = float(np.cos(direction_rad - crowd_direction_rad)) if crowd_direction_valid else 0.0

    total_path = float(sum(step_speeds))
    straightness = displacement / total_path if total_path > 1e-6 else 0.0

    # -- appearance --------------------------------------------------
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    aspect_ratio = box_width / box_height if box_height > 0 else 0.0
    pos_y_norm = ((y1 + y2) / 2.0) / frame_height if frame_height > 0 else 0.0

    # -- skeleton ----------------------------------------------------
    torso_height_ratio, head_to_hip_ratio, knee_bend_angle = compute_pose_features(
        keypoints, box_height
    )

    return np.array([
        displacement,
        direction_sin,
        direction_cos,
        cos_sim,
        straightness,
        aspect_ratio,
        pos_y_norm,
        torso_height_ratio,
        head_to_hip_ratio,
        knee_bend_angle,
    ], dtype=np.float32)
