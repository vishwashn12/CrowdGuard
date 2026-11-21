"""Ground-plane calibration of a camera view.

The operator marks four points on the ground in a gate camera's image - the
corners of a rectangle on the floor of the queue area, in order around it - and
says how big that rectangle is in metres. From that:

* a **homography** maps any image point on the ground to metres, so a tracked
  person's foot position, and therefore their walking speed, is in m/s rather
  than px/s;
* the rectangle becomes the gate's **monitored zone**: people are counted only
  inside it, and dividing by its area gives a physical density in people/m^2.

Corner order: P1 -> P2 is the side measured as ``width_m``, P2 -> P3 the side
measured as ``length_m``; P3 -> P4 and P4 -> P1 close the rectangle. Points are
stored normalised to [0, 1] of the frame, so they survive any resizing.

Pure numpy: usable from both the vision side and the decision adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_SIDE_M = 0.5
MAX_SIDE_M = 500.0
MIN_IMAGE_AREA = 0.002   # the zone must cover at least 0.2% of the frame


class CalibrationError(ValueError):
    """The marked points or dimensions cannot form a valid ground rectangle."""


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography H with dst ~ H @ src, from exactly four correspondences."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    h = np.linalg.solve(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return homog[:, :2] / homog[:, 2:3]


@dataclass(frozen=True)
class GroundCalibration:
    points: tuple[tuple[float, float], ...]
    width_m: float
    length_m: float
    _cache: dict = field(default_factory=dict, compare=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float)
        if pts.shape != (4, 2):
            raise CalibrationError("Mark exactly four corner points.")
        if not np.all((pts >= 0.0) & (pts <= 1.0)):
            raise CalibrationError("Corner points must lie inside the frame.")
        for name, side in (("Width", self.width_m), ("Length", self.length_m)):
            if not (MIN_SIDE_M <= side <= MAX_SIDE_M):
                raise CalibrationError(f"{name} must be between {MIN_SIDE_M} and {MAX_SIDE_M:.0f} m.")
        crosses = [_cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) for i in range(4)]
        if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
            raise CalibrationError("The four points must go around the rectangle in order, "
                                   "forming a shape without crossing sides.")
        if self.image_area < MIN_IMAGE_AREA:
            raise CalibrationError("The marked area is too small; mark a larger part of the queue area.")

    # ── geometry ────────────────────────────────────────────────────

    @property
    def area_m2(self) -> float:
        return self.width_m * self.length_m

    @property
    def image_area(self) -> float:
        """Area of the marked quadrilateral as a share of the frame."""
        p = np.asarray(self.points)
        x, y = p[:, 0], p[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    def homography(self, width_px: int, height_px: int) -> np.ndarray:
        """Image pixels -> metres on the ground (P1 at the origin)."""
        key = ("h", width_px, height_px)
        if key not in self._cache:
            src = np.asarray(self.points) * np.array([width_px, height_px])
            dst = np.array([[0, 0], [self.width_m, 0], [self.width_m, self.length_m], [0, self.length_m]],
                           dtype=float)
            self._cache[key] = solve_homography(src, dst)
        return self._cache[key]

    def to_ground(self, points_px: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
        return apply_homography(self.homography(width_px, height_px), points_px)

    def inside_ground(self, ground: np.ndarray, margin_m: float = 0.0) -> np.ndarray:
        g = np.asarray(ground).reshape(-1, 2)
        return ((g[:, 0] >= -margin_m) & (g[:, 0] <= self.width_m + margin_m)
                & (g[:, 1] >= -margin_m) & (g[:, 1] <= self.length_m + margin_m))

    def coverage(self, map_width: int, map_height: int, supersample: int = 4) -> np.ndarray:
        """Share of each cell of a ``map_height x map_width`` grid inside the zone.

        Used to count people in the zone from a density map: the masked sum of
        the map. Cells cut by the boundary get a fractional weight.
        """
        key = ("cov", map_width, map_height, supersample)
        if key in self._cache:
            return self._cache[key]
        s = supersample
        ys = (np.arange(map_height * s) + 0.5) / (map_height * s)
        xs = (np.arange(map_width * s) + 0.5) / (map_width * s)
        gx, gy = np.meshgrid(xs, ys)
        p = np.asarray(self.points)
        signs = []
        for i in range(4):
            a, b = p[i], p[(i + 1) % 4]
            signs.append((b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0]))
        signs = np.stack(signs)
        inside = np.all(signs >= 0, axis=0) | np.all(signs <= 0, axis=0)
        weights = inside.reshape(map_height, s, map_width, s).mean(axis=(1, 3))
        self._cache[key] = weights
        return weights

    def count_in_zone(self, density_map: np.ndarray) -> float:
        """People inside the zone, from a crowd-density map of the whole frame."""
        d = np.asarray(density_map, dtype=float)
        if d.ndim == 3:
            d = d.squeeze()
        return float(max(0.0, (d * self.coverage(d.shape[1], d.shape[0])).sum()))

    # ── serialisation ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
            "width_m": self.width_m,
            "length_m": self.length_m,
            "area_m2": round(self.area_m2, 2),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundCalibration":
        return cls(
            points=tuple((float(x), float(y)) for x, y in data["points"]),
            width_m=float(data["width_m"]),
            length_m=float(data["length_m"]),
        )
