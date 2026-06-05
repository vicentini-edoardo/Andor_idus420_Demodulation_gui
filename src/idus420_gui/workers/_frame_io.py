"""Shared camera frame I/O helpers used by acquisition and scan workers."""

from __future__ import annotations

import numpy as np

from idus420_gui.camera.base import CameraBackend, TriggerMode

_MAX_REARM_ATTEMPTS = 3


def _read_ready_frames(backend: CameraBackend, max_frames: int | None = None) -> np.ndarray:
    """Return all SDK-reported ready frames, falling back to one-frame reads.

    Call only after wait_next_frame() reports True.
    """
    frame_width = backend.frame_width()
    try:
        batch = backend.get_new_frames_batch()
    except (AttributeError, NotImplementedError, TypeError):
        batch = None
    if batch is None:
        frame = backend.get_oldest_frame()
        return frame.reshape(1, frame_width).copy()

    frames = np.asarray(batch, dtype=np.uint16)
    if frames.size == 0:
        frame = backend.get_oldest_frame()
        return frame.reshape(1, frame_width).copy()
    frames = frames.reshape(-1, frame_width)
    if max_frames is not None:
        frames = frames[:max(0, int(max_frames))]
    return frames.copy()


def _read_pending_frames(backend: CameraBackend, max_frames: int | None = None) -> np.ndarray | None:
    """Drain SDK-reported frames after a wait timeout, without unsafe fallback reads."""
    try:
        batch = backend.get_new_frames_batch()
    except Exception:  # noqa: BLE001 - timeout recovery must fall back to re-arm.
        return None
    if batch is None:
        return None
    frame_width = backend.frame_width()
    frames = np.asarray(batch, dtype=np.uint16)
    if frames.size == 0:
        return None
    frames = frames.reshape(-1, frame_width)
    if max_frames is not None:
        frames = frames[:max(0, int(max_frames))]
    if frames.size == 0:
        return None
    return frames.copy()


def _timeout_error(timeout_ms: int) -> str:
    return (
        f"No completed camera frames after {timeout_ms / 1000:.1f} s"
        " (×3). Trigger may be present; camera/SDK did not deliver frames."
    )


def _rearm_message(
    timeout_ms: int,
    attempt: int,
    max_attempts: int,
    diagnostics: str,
) -> str:
    return (
        f"No completed camera frames after {timeout_ms / 1000:.1f} s"
        f" (×3). Re-arming camera acquisition ({attempt}/{max_attempts}). "
        f"Diagnostics: {diagnostics}"
    )


def _timeout_failure_message(timeout_ms: int, diagnostics: str) -> str:
    return f"{_timeout_error(timeout_ms)} Diagnostics: {diagnostics}"


def _rearm_acquisition(
    backend: CameraBackend,
    exposure_s: float,
    remaining_frames: int,
) -> None:
    backend.setup_kinetic(
        exposure_s,
        max(1, int(remaining_frames)),
        TriggerMode.EXTERNAL,
    )
    backend.start()
