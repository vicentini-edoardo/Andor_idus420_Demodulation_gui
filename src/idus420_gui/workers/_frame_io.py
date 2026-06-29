"""Shared camera frame I/O helpers used by acquisition and scan workers."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from idus420_gui.camera.base import CameraBackend, TriggerMode

_MAX_REARM_ATTEMPTS = 3
_TIMEOUTS_BEFORE_REARM = 3


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


def _read_pending_frames(
    backend: CameraBackend, max_frames: int | None = None
) -> np.ndarray | None:
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


def pump_frames(
    backend: CameraBackend,
    *,
    total_frames: int,
    exposure_s: float,
    timeout_ms: int,
    is_running: Callable[[], bool],
    emit_error: Callable[[str], None],
    on_fatal: Callable[[], None] | None = None,
    error_prefix: str = "",
) -> Iterator[np.ndarray]:
    """Drive the wait / read / timeout / re-arm loop for one kinetic series.

    Yields batches of ready frames (each shaped ``(k, frame_width)``) until
    ``total_frames`` frames have been delivered, ``is_running()`` returns
    ``False``, or the camera repeatedly fails to deliver frames.  The caller is
    responsible for calling ``setup_kinetic()`` + ``start()`` before iterating
    and ``abort()`` afterwards.

    On an unrecoverable timeout the failure message (prefixed with
    ``error_prefix``) is sent through ``emit_error`` and, if provided,
    ``on_fatal`` is invoked so the caller can stop any enclosing loop before the
    generator returns.  This is the single source of truth for the polling
    policy shared by every acquisition/scan worker.
    """
    acquired = 0
    consecutive_timeouts = 0
    rearm_attempts = 0
    while is_running() and acquired < total_frames:
        if backend.wait_next_frame(timeout_ms):
            consecutive_timeouts = 0
            rearm_attempts = 0
            ready = _read_ready_frames(backend, total_frames - acquired)
        else:
            pending = _read_pending_frames(backend, total_frames - acquired)
            if pending is not None:
                consecutive_timeouts = 0
                rearm_attempts = 0
                ready = pending
            else:
                consecutive_timeouts += 1
                if consecutive_timeouts < _TIMEOUTS_BEFORE_REARM:
                    continue
                diagnostics = backend.acquisition_diagnostics()
                remaining = total_frames - acquired
                if rearm_attempts < _MAX_REARM_ATTEMPTS and remaining > 0:
                    rearm_attempts += 1
                    emit_error(
                        error_prefix
                        + _rearm_message(
                            timeout_ms, rearm_attempts, _MAX_REARM_ATTEMPTS, diagnostics
                        )
                    )
                    _rearm_acquisition(backend, exposure_s, remaining)
                    consecutive_timeouts = 0
                    continue
                emit_error(
                    error_prefix + _timeout_failure_message(timeout_ms, diagnostics)
                )
                if on_fatal is not None:
                    on_fatal()
                return
        acquired += len(ready)
        yield ready
