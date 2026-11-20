"""Small geometry helpers shared by the tracking and motion layers.

Pure numpy/math only -- no cv2, no torch.
"""

from __future__ import annotations

import math

import numpy as np


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union of two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Centre point of an ``(x1, y1, x2, y2)`` box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def weighted_circular_mean(angles: list[float], weights: list[float]) -> float:
    """Weighted circular mean of angles in radians.

    Uses ``atan2(sum(w*sin), sum(w*cos))`` so the wraparound at +/-pi is handled
    correctly (e.g. the mean of 350 deg and 10 deg is 0 deg, not 180 deg).
    """
    if not angles:
        return 0.0
    ang = np.asarray(angles, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    total = wts.sum()
    if total <= 0:
        wts = np.ones_like(ang) / len(ang)
    else:
        wts = wts / total
    return float(np.arctan2(np.sum(wts * np.sin(ang)), np.sum(wts * np.cos(ang))))


def exponential_circular_smooth(history: list[float]) -> float:
    """Exponentially-weighted circular mean over a history of angles (newest last)."""
    if not history:
        return 0.0
    if len(history) == 1:
        return float(history[0])
    hist = np.asarray(history, dtype=np.float64)
    wts = np.exp(np.linspace(-1.0, 0.0, len(hist)))
    wts /= wts.sum()
    return float(np.arctan2(np.sum(wts * np.sin(hist)), np.sum(wts * np.cos(hist))))


def joint_angle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Interior angle at vertex ``b`` of the path a-b-c, in degrees [0, 180]."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosine = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
