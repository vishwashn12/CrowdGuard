"""Persisted ground calibrations, keyed by camera.

A calibration describes one camera's view, so it is stored per camera, not per
gate: replacing a gate's footage starts uncalibrated, and moving a camera to a
different gate keeps its calibration. Saved as JSON beside the uploads so it
survives a restart.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from crowdguard.utils.calibration import CalibrationError, GroundCalibration

logger = logging.getLogger(__name__)


class CalibrationStore:
    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._items: dict[str, GroundCalibration] = {}
        self._load()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read calibrations from %s: %s", self._path, exc)
            return
        for camera_id, data in raw.items():
            try:
                self._items[camera_id] = GroundCalibration.from_dict(data)
            except (CalibrationError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Ignoring invalid calibration for %s: %s", camera_id, exc)

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v.to_dict() for k, v in self._items.items()}, indent=2),
                           encoding="utf-8")
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("Could not save calibrations to %s: %s", self._path, exc)

    def get(self, camera_id: str | None) -> GroundCalibration | None:
        if camera_id is None:
            return None
        with self._lock:
            return self._items.get(camera_id)

    def set(self, camera_id: str, calibration: GroundCalibration) -> None:
        with self._lock:
            self._items[camera_id] = calibration
            self._save()

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            removed = self._items.pop(camera_id, None) is not None
            if removed:
                self._save()
            return removed
