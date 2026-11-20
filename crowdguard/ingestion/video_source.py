"""Frame sources: video files, RTSP streams, and local webcams."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crowdguard.utils.exceptions import VideoSourceError


class BaseFrameSource(ABC):
    """Interface every frame source implements."""

    source_type: str = "abstract"

    @abstractmethod
    def open(self) -> bool:
        """Open the source and verify it is usable."""

    @abstractmethod
    def read_next(self) -> tuple[np.ndarray | None, int | None]:
        """Return ``(frame, source_frame_index)``, or ``(None, None)`` at end of stream."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    @abstractmethod
    def rewind(self) -> bool:
        """Seek back to the start. Returns False when the source cannot seek."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for unbounded sources (RTSP, webcam) that cannot be rewound."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        ...

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total frames if known, else 0."""

    @property
    @abstractmethod
    def source_fps(self) -> float:
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """``(width, height)``."""


class _CaptureSource(BaseFrameSource):
    """Shared OpenCV ``VideoCapture`` behaviour."""

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._opened = False
        self._eof = False
        self._frame_index = 0
        self._source_fps = 0.0
        self._width = 0
        self._height = 0
        self._frame_count = 0

    # -- capture target, provided by subclasses -------------------------

    def _capture_target(self) -> Any:
        raise NotImplementedError

    def _describe_target(self) -> str:
        raise NotImplementedError

    # -- lifecycle ------------------------------------------------------

    def _read_properties(self, capture: cv2.VideoCapture) -> None:
        self._source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Some streams report nonsense; fall back to a sane assumption so the
        # px/frame to px/s conversion downstream stays finite.
        if not (1.0 <= self._source_fps <= 240.0):
            self._source_fps = 25.0

    def open(self) -> bool:
        if self.is_open:
            return True

        capture = cv2.VideoCapture(self._capture_target())
        if not capture.isOpened():
            capture.release()
            self._capture = None
            self._opened = False
            raise VideoSourceError(f"Could not open video source: {self._describe_target()}")

        self._capture = capture
        self._read_properties(capture)
        self._opened = True
        self._eof = False
        self._frame_index = 0
        return True

    def read_next(self) -> tuple[np.ndarray | None, int | None]:
        if not self.is_open:
            raise VideoSourceError(f"Source is not open: {self._describe_target()}")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._eof = True
            return None, None

        index = self._frame_index
        self._frame_index += 1
        return np.asarray(frame, dtype=np.uint8), index

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._opened = False
        self._eof = True

    def rewind(self) -> bool:
        if self.is_live or self._capture is None:
            return False
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_index = 0
        self._eof = False
        return True

    # -- properties -----------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._opened and self._capture is not None and self._capture.isOpened()

    @property
    def is_eof(self) -> bool:
        return self._eof

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "target": self._describe_target(),
            "fps": self._source_fps,
            "width": self._width,
            "height": self._height,
            "frame_count": self._frame_count,
            "duration_s": round(self._frame_count / self._source_fps, 2) if self._source_fps > 0 else 0.0,
            "opened": self.is_open,
            "eof": self._eof,
            "is_live": self.is_live,
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VideoFileSource(_CaptureSource):
    """A video file read in order from disk."""

    source_type = "video"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._probe()

    def _probe(self) -> None:
        """Read metadata without holding the file open."""
        if not self.path.exists():
            return
        capture = cv2.VideoCapture(str(self.path))
        if capture.isOpened():
            self._read_properties(capture)
        capture.release()

    def _capture_target(self) -> str:
        return str(self.path)

    def _describe_target(self) -> str:
        return str(self.path)

    def open(self) -> bool:
        if not self.path.exists():
            raise VideoSourceError(f"Video file does not exist: {self.path}")
        return super().open()

    @property
    def is_live(self) -> bool:
        return False

    def timestamp_for_index(self, source_frame_index: int) -> float:
        if self._source_fps <= 0:
            return float(source_frame_index)
        return float(source_frame_index) / float(self._source_fps)


class RtspSource(_CaptureSource):
    """A live RTSP / HTTP network stream."""

    source_type = "rtsp"

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = str(url)

    def _capture_target(self) -> str:
        return self.url

    def _describe_target(self) -> str:
        # Never echo embedded credentials into logs.
        if "@" in self.url:
            scheme, _, rest = self.url.partition("://")
            _, _, host = rest.partition("@")
            return f"{scheme}://<redacted>@{host}"
        return self.url

    @property
    def is_live(self) -> bool:
        return True

    def timestamp_for_index(self, source_frame_index: int) -> float:
        if self._source_fps <= 0:
            return float(source_frame_index)
        return float(source_frame_index) / float(self._source_fps)


class WebcamSource(_CaptureSource):
    """A locally attached camera, addressed by device index."""

    source_type = "webcam"

    def __init__(self, device_index: int | str = 0) -> None:
        super().__init__()
        self.device_index = int(device_index)

    def _capture_target(self) -> int:
        return self.device_index

    def _describe_target(self) -> str:
        return f"webcam:{self.device_index}"

    @property
    def is_live(self) -> bool:
        return True

    def timestamp_for_index(self, source_frame_index: int) -> float:
        if self._source_fps <= 0:
            return float(source_frame_index)
        return float(source_frame_index) / float(self._source_fps)


def build_source(source_type: str, source: str) -> BaseFrameSource:
    """Construct the source matching a camera configuration.

    Raises:
        VideoSourceError: On an unknown ``source_type``.
    """
    kind = (source_type or "video").lower()
    if kind == "video":
        return VideoFileSource(source)
    if kind == "rtsp":
        return RtspSource(source)
    if kind == "webcam":
        return WebcamSource(source)
    raise VideoSourceError(f"Unsupported source_type: {source_type!r}")
