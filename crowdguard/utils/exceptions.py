"""Project-wide exception hierarchy."""

from __future__ import annotations


class CrowdGuardError(Exception):
    """Base class for all CrowdGuard errors."""


class ConfigurationError(CrowdGuardError):
    """Raised when a configuration file is missing or invalid."""


class VideoSourceError(CrowdGuardError):
    """Raised when a video source cannot be opened or read."""


class FrameBufferError(CrowdGuardError):
    """Raised on illegal frame-buffer operations."""


class CheckpointError(CrowdGuardError):
    """Raised when model weights are missing or invalid."""


class CameraRegistryError(CrowdGuardError):
    """Raised on illegal camera registration / lifecycle operations."""


class SessionError(CrowdGuardError):
    """Raised when an action conflicts with the session state.

    Setup changes are refused while a session is running, and a session cannot
    start with no footage staged.
    """
