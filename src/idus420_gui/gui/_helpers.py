"""Shared helpers for GUI panels: ROI validation, settings I/O, and worker lifecycle.

These utilities consolidate logic that was previously copy-pasted across the
Live, Demodulation, and Acquisition panels.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSettings


def roi_error_message(start: int, end: int, detector_width: int) -> str | None:
    """Return a human-readable error if the ROI is invalid, else ``None``."""
    if start > end:
        return "ROI start must be ≤ ROI end."
    if end >= detector_width:
        return f"ROI end {end} exceeds detector width {detector_width}."
    return None


def stop_worker(worker: Any, msec: int = 2000) -> None:
    """Request a clean stop and block briefly until the worker thread exits.

    Safe to call with ``None`` or with an already-finished worker.
    """
    if worker is None:
        return
    worker.stop()
    worker.wait(msec)


class PanelSettings:
    """Typed wrapper around :class:`QSettings` with a per-panel key prefix.

    Centralizes value coercion (notably the boolean parsing that differed
    between panels) so persisted settings round-trip consistently.
    """

    def __init__(self, application: str, prefix: str) -> None:
        self._settings = QSettings("idus420_gui", application)
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key}"

    def set(self, key: str, value: Any) -> None:
        self._settings.setValue(self._key(key), value)

    def get_float(self, key: str, default: float) -> float:
        value = self._settings.value(self._key(key))
        return float(value) if value is not None else default

    def get_int(self, key: str, default: int) -> int:
        value = self._settings.value(self._key(key))
        return int(value) if value is not None else default

    def get_str(self, key: str, default: str) -> str:
        value = self._settings.value(self._key(key))
        return str(value) if value is not None else default

    def get_bool(self, key: str, default: bool) -> bool:
        value = self._settings.value(self._key(key))
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"true", "1", "yes"}
