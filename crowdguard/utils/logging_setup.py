"""Centralised logging configuration."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int | str = logging.INFO, quiet_libs: bool = True) -> None:
    """Install a single stream handler on the root logger.

    Args:
        level: Root log level.
        quiet_libs: Suppress noisy third-party loggers (ultralytics, PIL, uvicorn access).
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)

    if quiet_libs:
        for name in ("ultralytics", "PIL", "matplotlib", "uvicorn.access", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
