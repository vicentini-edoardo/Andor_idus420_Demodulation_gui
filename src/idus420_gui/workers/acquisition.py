"""Worker threads that keep blocking camera operations off the GUI thread."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from idus420_gui.camera.base import CameraBackend, TriggerMode
from idus420_gui.processing.demodulation import DemodResult, demodulate
from idus420_gui.processing.roi import integrate_roi
from idus420_gui.workers._frame_io import pump_frames

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
    """Acquires kinetic frames and emits demodulated results.

    In continuous mode the worker keeps a *rolling* window of the most recent
    ``n_block`` ROI samples and re-runs the FFT each time fresh frames arrive
    (throttled to the GUI update rate).  This gives a live, smoothly updating
    alignment view instead of one result per non-overlapping block.  In
    one-shot mode (``continuous=False``) it collects a single block of
    ``n_block`` frames, emits one result, and exits.
    """

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
        parent: QObject | None = None,
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
            n_block = self.settings.n_block
            acquisition_frames = (
                n_block
                if not self.continuous
                else max(n_block * 64, _LIVE_ACQUISITION_FRAMES)
            )
            # Rolling window of the most recent ROI samples (continuous mode);
            # discrete accumulator for the one-shot block (non-continuous mode).
            roi_window: deque[float] = deque(maxlen=n_block)
            block_frames: list[np.ndarray] = []
            last_preview_s = 0.0
            last_demod_s = 0.0
            while self._running:
                self.backend.setup_kinetic(
                    self.settings.exposure_s,
                    acquisition_frames,
                    TriggerMode.EXTERNAL,
                )
                self.backend.start()
                for ready in pump_frames(
                    self.backend,
                    total_frames=acquisition_frames,
                    exposure_s=self.settings.exposure_s,
                    timeout_ms=timeout_ms,
                    is_running=lambda: self._running,
                    emit_error=self.error.emit,
                    on_fatal=self.stop,
                ):
                    for frame in ready:
                        now = time.monotonic()
                        if now - last_preview_s >= _GUI_UPDATE_INTERVAL_S:
                            self.frame_acquired.emit(frame.copy())
                            last_preview_s = now
                        if self.continuous:
                            # Push the new ROI sample into the rolling window and
                            # re-demodulate the full window (throttled) once it fills.
                            roi_window.append(
                                float(
                                    integrate_roi(
                                        frame.reshape(1, -1),
                                        self.settings.pixel_start,
                                        self.settings.pixel_end,
                                        self.settings.roi_method,
                                    )[0]
                                )
                            )
                            if (
                                len(roi_window) == n_block
                                and now - last_demod_s >= _GUI_UPDATE_INTERVAL_S
                            ):
                                roi_ts = np.fromiter(
                                    roi_window, dtype=np.float64, count=n_block
                                )
                                self.block_complete.emit(roi_ts)
                                result = demodulate(
                                    roi_ts,
                                    self.settings.trigger_frequency_hz,
                                    self.settings.f_expected,
                                    self.settings.f_search_halfwidth,
                                    self.settings.window,
                                )
                                self.demod_result.emit(result)
                                last_demod_s = now
                        else:
                            block_frames.append(frame.copy())
                            if len(block_frames) == n_block:
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
        parent: QObject | None = None,
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
        frame_times: list[float] = []  # per-frame timestamps (s) relative to start
        try:
            timeout_ms = self.settings.frame_timeout_ms()
            self.backend.setup_kinetic(
                self.settings.exposure_s,
                self.total_frames,
                TriggerMode.EXTERNAL,
            )
            self.backend.start()
            start_wall = time.monotonic()
            for ready in pump_frames(
                self.backend,
                total_frames=self.total_frames,
                exposure_s=self.settings.exposure_s,
                timeout_ms=timeout_ms,
                is_running=lambda: self._running,
                emit_error=self.error.emit,
            ):
                for frame in ready:
                    all_frames.append(frame.copy())
                    frame_times.append(time.monotonic() - start_wall)
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
                    "frame_times_s": np.asarray(frame_times, dtype=np.float64),
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
    roi_sample = pyqtSignal(float, float, float)
    error = pyqtSignal(str)
    worker_finished = pyqtSignal()

    def __init__(
        self,
        backend: CameraBackend,
        exposure_s: float,
        trigger_frequency_hz: float,
        pixel_start: int,
        pixel_end: int,
        pixel_start2: int = 0,
        pixel_end2: int = 0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.exposure_s = float(exposure_s)
        self.trigger_frequency_hz = float(trigger_frequency_hz)
        self.pixel_start = int(pixel_start)
        self.pixel_end = int(pixel_end)
        self.pixel_start2 = int(pixel_start2)
        self.pixel_end2 = int(pixel_end2)
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
            acquisition_frames = _LIVE_ACQUISITION_FRAMES
            while self._running:
                self.backend.setup_kinetic(
                    self.exposure_s,
                    acquisition_frames,
                    TriggerMode.EXTERNAL,
                )
                self.backend.start()
                for ready in pump_frames(
                    self.backend,
                    total_frames=acquisition_frames,
                    exposure_s=self.exposure_s,
                    timeout_ms=timeout_ms,
                    is_running=lambda: self._running,
                    emit_error=self.error.emit,
                    on_fatal=self.stop,
                ):
                    for frame in ready:
                        self.frame_acquired.emit(frame.copy())
                        frame_row = frame.reshape(1, -1)
                        roi_mean = float(
                            integrate_roi(
                                frame_row,
                                self.pixel_start,
                                self.pixel_end,
                                "mean",
                            )[0]
                        )
                        roi_mean2 = float(
                            integrate_roi(
                                frame_row,
                                self.pixel_start2,
                                self.pixel_end2,
                                "mean",
                            )[0]
                        )
                        elapsed_s = sample_index / max(self.trigger_frequency_hz, 1e-6)
                        self.roi_sample.emit(elapsed_s, roi_mean, roi_mean2)
                        sample_index += 1
        except Exception as exc:  # noqa: BLE001 - worker must report and exit cleanly.
            LOG.exception("LiveSpectrumWorker error")
            self.error.emit(str(exc))
        finally:
            self.backend.abort()
            self.worker_finished.emit()
