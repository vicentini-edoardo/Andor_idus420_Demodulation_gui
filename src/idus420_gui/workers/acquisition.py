"""Worker threads that keep blocking camera operations off the GUI thread."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from idus420_gui.camera.base import CameraBackend, TriggerMode
from idus420_gui.processing.demodulation import DemodResult, demodulate
from idus420_gui.processing.roi import integrate_roi
from idus420_gui.workers._frame_io import (
    _MAX_REARM_ATTEMPTS,
    _read_pending_frames,
    _read_ready_frames,
    _rearm_acquisition,
    _rearm_message,
    _timeout_failure_message,
)

LOG = logging.getLogger(__name__)

_LIVE_ACQUISITION_FRAMES = 4_096
_GUI_UPDATE_INTERVAL_S = 0.05


@dataclass(frozen=True)
class DemodulationSettings:
    """Runtime settings for a demodulation block."""

    exposure_s: float
    trigger_frequency_hz: float
    pixel_start: int
    pixel_end: int
    roi_method: Literal["sum", "mean"]
    n_block: int
    f_expected: float
    f_search_halfwidth: float
    window: Literal["hann", "blackman", "none"]

    def frame_timeout_ms(self) -> int:
        """Per-frame timeout: 5× the kinetic cycle time, minimum 5 s.

        The generous floor absorbs OS scheduler starvation on a busy host
        without affecting acquisition speed (triggers still drive the pace).
        """
        cycle_s = 1.0 / max(self.trigger_frequency_hz, 1e-6)
        return int(max(5000, cycle_s * 5 * 1000))


class DemodulationWorker(QThread):
    """Continuously acquires kinetic blocks and emits demodulated results."""

    frame_acquired = pyqtSignal(object)
    block_complete = pyqtSignal(object)
    demod_result = pyqtSignal(object)
    error = pyqtSignal(str)
    worker_finished = pyqtSignal()

    def __init__(
        self,
        backend: CameraBackend,
        settings: DemodulationSettings,
        continuous: bool = True,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.settings = settings
        self.continuous = continuous
        self._running = True

    def stop(self) -> None:
        """Request a clean stop; abort unblocks any pending wait_next_frame call."""
        self._running = False
        self.backend.abort()

    def run(self) -> None:
        self.setPriority(self.Priority.HighPriority)
        try:
            timeout_ms = self.settings.frame_timeout_ms()
            acquisition_frames = (
                self.settings.n_block
                if not self.continuous
                else max(self.settings.n_block * 64, _LIVE_ACQUISITION_FRAMES)
            )
            block_frames: list[np.ndarray] = []
            last_preview_s = 0.0
            while self._running:
                self.backend.setup_kinetic(
                    self.settings.exposure_s,
                    acquisition_frames,
                    TriggerMode.EXTERNAL,
                )
                self.backend.start()
                acquired = 0
                consecutive_timeouts = 0
                rearm_attempts = 0
                while self._running and acquired < acquisition_frames:
                    if not self._running:
                        break
                    if not self.backend.wait_next_frame(timeout_ms):
                        pending = _read_pending_frames(
                            self.backend,
                            acquisition_frames - acquired,
                        )
                        if pending is not None:
                            consecutive_timeouts = 0
                            rearm_attempts = 0
                            ready = pending
                        else:
                            consecutive_timeouts += 1
                            if consecutive_timeouts >= 3:
                                diagnostics = self.backend.acquisition_diagnostics()
                                remaining = acquisition_frames - acquired
                                if rearm_attempts < _MAX_REARM_ATTEMPTS and remaining > 0:
                                    rearm_attempts += 1
                                    self.error.emit(
                                        _rearm_message(
                                            timeout_ms,
                                            rearm_attempts,
                                            _MAX_REARM_ATTEMPTS,
                                            diagnostics,
                                        )
                                    )
                                    _rearm_acquisition(
                                        self.backend,
                                        self.settings.exposure_s,
                                        remaining,
                                    )
                                    consecutive_timeouts = 0
                                    continue
                                self.error.emit(
                                    _timeout_failure_message(timeout_ms, diagnostics)
                                )
                                self._running = False
                                break
                            continue
                    else:
                        consecutive_timeouts = 0
                        rearm_attempts = 0
                        ready = _read_ready_frames(
                            self.backend,
                            acquisition_frames - acquired,
                        )
                    now = time.monotonic()
                    for frame in ready:
                        block_frames.append(frame.copy())
                        acquired += 1
                        if now - last_preview_s >= _GUI_UPDATE_INTERVAL_S:
                            self.frame_acquired.emit(frame.copy())
                            last_preview_s = now
                        if len(block_frames) == self.settings.n_block:
                            block = np.stack(block_frames, axis=0)
                            block_frames.clear()
                            roi_ts = integrate_roi(
                                block,
                                self.settings.pixel_start,
                                self.settings.pixel_end,
                                self.settings.roi_method,
                            )
                            self.block_complete.emit(roi_ts.copy())
                            if roi_ts.size >= 4:
                                result = demodulate(
                                    roi_ts,
                                    self.settings.trigger_frequency_hz,
                                    self.settings.f_expected,
                                    self.settings.f_search_halfwidth,
                                    self.settings.window,
                                )
                                self.demod_result.emit(result)
                if not self.continuous:
                    break
        except Exception as exc:  # noqa: BLE001 - worker must report and exit cleanly.
            LOG.exception("DemodulationWorker error")
            self.error.emit(str(exc))
        finally:
            self.backend.abort()
            self.worker_finished.emit()


class AcquisitionWorker(QThread):
    """One-shot acquisition worker that emits partial previews and final arrays."""

    frame_acquired = pyqtSignal(object)
    progress = pyqtSignal(int, int, float)
    demod_result = pyqtSignal(object)   # emitted per completed block during acquisition
    run_finished = pyqtSignal(object, object)
    error = pyqtSignal(str)
    worker_finished = pyqtSignal()

    def __init__(
        self,
        backend: CameraBackend,
        settings: DemodulationSettings,
        total_seconds: float | None = None,
        total_frames: int | None = None,
        sif_path: str | None = None,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        if total_frames is None:
            if total_seconds is None:
                raise ValueError("Either total_seconds or total_frames is required.")
            total_frames = int(math.ceil(total_seconds * settings.trigger_frequency_hz))
        self.backend = backend
        self.settings = settings
        self.total_frames = int(total_frames)
        self.sif_path = sif_path
        self._running = True

    def stop(self) -> None:
        """Stop acquisition and save whatever frames were captured so far."""
        self._running = False
        self.backend.abort()

    def run(self) -> None:
        self.setPriority(self.Priority.HighPriority)
        all_frames: list[np.ndarray] = []
        demod_results: list[DemodResult] = []
        roi_buffer: list[float] = []   # running accumulator for per-block demodulation
        roi_all: list[float] = []      # all ROI values in acquisition order
        try:
            timeout_ms = self.settings.frame_timeout_ms()
            self.backend.setup_kinetic(
                self.settings.exposure_s,
                self.total_frames,
                TriggerMode.EXTERNAL,
            )
            self.backend.start()
            start_wall = time.monotonic()
            consecutive_timeouts = 0
            rearm_attempts = 0
            while self._running and len(all_frames) < self.total_frames:
                if not self.backend.wait_next_frame(timeout_ms):
                    pending = _read_pending_frames(
                        self.backend,
                        self.total_frames - len(all_frames),
                    )
                    if pending is not None:
                        consecutive_timeouts = 0
                        rearm_attempts = 0
                        ready = pending
                    else:
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= 3:
                            diagnostics = self.backend.acquisition_diagnostics()
                            remaining = self.total_frames - len(all_frames)
                            if rearm_attempts < _MAX_REARM_ATTEMPTS and remaining > 0:
                                rearm_attempts += 1
                                self.error.emit(
                                    _rearm_message(
                                        timeout_ms,
                                        rearm_attempts,
                                        _MAX_REARM_ATTEMPTS,
                                        diagnostics,
                                    )
                                )
                                _rearm_acquisition(
                                    self.backend,
                                    self.settings.exposure_s,
                                    remaining,
                                )
                                consecutive_timeouts = 0
                                continue
                            self.error.emit(
                                _timeout_failure_message(timeout_ms, diagnostics)
                            )
                            break
                        continue
                else:
                    consecutive_timeouts = 0
                    rearm_attempts = 0
                    ready = _read_ready_frames(
                        self.backend,
                        self.total_frames - len(all_frames),
                    )
                for frame in ready:
                    all_frames.append(frame.copy())
                    self.frame_acquired.emit(frame.copy())

                    # Real-time ROI integration and per-block demodulation.
                    roi_val = float(
                        integrate_roi(
                            frame.reshape(1, -1),
                            self.settings.pixel_start,
                            self.settings.pixel_end,
                            self.settings.roi_method,
                        )[0]
                    )
                    roi_buffer.append(roi_val)
                    roi_all.append(roi_val)

                    if len(roi_buffer) == self.settings.n_block:
                        chunk = np.asarray(roi_buffer, dtype=np.float64)
                        roi_buffer.clear()
                        if len(chunk) >= 4:
                            result = demodulate(
                                chunk,
                                self.settings.trigger_frequency_hz,
                                self.settings.f_expected,
                                self.settings.f_search_halfwidth,
                                self.settings.window,
                            )
                            demod_results.append(result)
                            self.demod_result.emit(result)

                    elapsed = time.monotonic() - start_wall
                    self.progress.emit(len(all_frames), self.total_frames, elapsed)

            frames = (
                np.stack(all_frames, axis=0)
                if all_frames
                else np.empty((0, self.backend.frame_width()), dtype=np.uint16)
            )
            roi_ts = np.asarray(roi_all, dtype=np.float64)

            self.run_finished.emit(
                frames,
                {
                    "roi_timeseries": roi_ts,
                    "demod_results": demod_results,
                },
            )
            if self.sif_path:
                self.backend.save_as_sif(self.sif_path)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("AcquisitionWorker error")
            self.error.emit(str(exc))
        finally:
            self.backend.abort()
            self.worker_finished.emit()


class LiveSpectrumWorker(QThread):
    """Continuously streams frames for a large live-spectrum view."""

    frame_acquired = pyqtSignal(object)
    roi_sample = pyqtSignal(float, float)
    error = pyqtSignal(str)
    worker_finished = pyqtSignal()

    def __init__(
        self,
        backend: CameraBackend,
        exposure_s: float,
        trigger_frequency_hz: float,
        pixel_start: int,
        pixel_end: int,
        burst_frames: int = 64,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.exposure_s = float(exposure_s)
        self.trigger_frequency_hz = float(trigger_frequency_hz)
        self.pixel_start = int(pixel_start)
        self.pixel_end = int(pixel_end)
        self.burst_frames = int(max(4, burst_frames))
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.backend.abort()

    def run(self) -> None:
        timeout_ms = DemodulationSettings(
            exposure_s=self.exposure_s,
            trigger_frequency_hz=self.trigger_frequency_hz,
            pixel_start=self.pixel_start,
            pixel_end=self.pixel_end,
            roi_method="sum",
            n_block=4,
            f_expected=0.0,
            f_search_halfwidth=0.0,
            window="none",
        ).frame_timeout_ms()
        sample_index = 0
        try:
            self.setPriority(self.Priority.HighPriority)
            while self._running:
                self.backend.setup_kinetic(
                    self.exposure_s,
                    max(self.burst_frames, _LIVE_ACQUISITION_FRAMES),
                    TriggerMode.EXTERNAL,
                )
                self.backend.start()
                consecutive_timeouts = 0
                acquired = 0
                acquisition_frames = max(self.burst_frames, _LIVE_ACQUISITION_FRAMES)
                rearm_attempts = 0
                while self._running and acquired < acquisition_frames:
                    if not self._running:
                        break
                    if not self.backend.wait_next_frame(timeout_ms):
                        pending = _read_pending_frames(
                            self.backend,
                            acquisition_frames - acquired,
                        )
                        if pending is not None:
                            consecutive_timeouts = 0
                            rearm_attempts = 0
                            ready = pending
                        else:
                            consecutive_timeouts += 1
                            if consecutive_timeouts >= 3:
                                diagnostics = self.backend.acquisition_diagnostics()
                                remaining = acquisition_frames - acquired
                                if rearm_attempts < _MAX_REARM_ATTEMPTS and remaining > 0:
                                    rearm_attempts += 1
                                    self.error.emit(
                                        _rearm_message(
                                            timeout_ms,
                                            rearm_attempts,
                                            _MAX_REARM_ATTEMPTS,
                                            diagnostics,
                                        )
                                    )
                                    _rearm_acquisition(
                                        self.backend,
                                        self.exposure_s,
                                        remaining,
                                    )
                                    consecutive_timeouts = 0
                                    continue
                                self.error.emit(
                                    _timeout_failure_message(timeout_ms, diagnostics)
                                )
                                self._running = False
                                break
                            continue
                    else:
                        consecutive_timeouts = 0
                        rearm_attempts = 0
                        ready = _read_ready_frames(
                            self.backend,
                            acquisition_frames - acquired,
                        )
                    for frame in ready:
                        acquired += 1
                        self.frame_acquired.emit(frame.copy())
                        roi_sum = float(
                            integrate_roi(
                                frame.reshape(1, -1),
                                self.pixel_start,
                                self.pixel_end,
                                "sum",
                            )[0]
                        )
                        elapsed_s = sample_index / max(self.trigger_frequency_hz, 1e-6)
                        self.roi_sample.emit(elapsed_s, roi_sum)
                        sample_index += 1
        except Exception as exc:  # noqa: BLE001 - worker must report and exit cleanly.
            LOG.exception("LiveSpectrumWorker error")
            self.error.emit(str(exc))
        finally:
            self.backend.abort()
            self.worker_finished.emit()
