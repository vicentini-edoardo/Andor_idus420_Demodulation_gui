"""Worker threads that keep blocking camera operations off the GUI thread."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from idus420_gui.camera.base import CameraBackend, TriggerMode
from idus420_gui.processing.demodulation import DemodResult, demodulate
from idus420_gui.processing.roi import integrate_roi

# Frames acquired per camera kinetic series during continuous demodulation.
# The rolling ROI window persists across chunks, so a large chunk keeps the
# stream contiguous and limits how often a re-setup gap lands inside a window.
_STREAM_CHUNK_FRAMES = 4096


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
        """Conservative per-frame timeout: 3× the kinetic cycle time, minimum 2 s."""
        cycle_s = 1.0 / max(self.trigger_frequency_hz, 1e-6)
        return int(max(2000, cycle_s * 3 * 1000))


class DemodulationWorker(QThread):
    """Streams frames into a rolling ROI window and emits demodulated results.

    In continuous mode the worker keeps a single sliding FIFO buffer of ROI
    samples (``n_block`` long).  Each new frame appends one sample; the window
    fills, then rolls, and demodulation re-runs every ``hop_frames`` frames so
    the FFT, time series, and peak history update as a continuous stream rather
    than in disjoint blocks.  ``continuous=False`` keeps the legacy single-block
    behaviour used by one-shot callers and tests.
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
        hop_frames: int = 0,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend
        self.settings = settings
        self.continuous = continuous
        # Frames between rolling-window updates; 0 selects an automatic cadence.
        self._hop_frames = int(hop_frames)
        self._running = True

    def stop(self) -> None:
        """Request a clean stop; abort unblocks any pending wait_next_frame call."""
        self._running = False
        self.backend.abort()

    def run(self) -> None:
        try:
            if self.continuous:
                self._run_continuous()
            else:
                self._run_single_block()
        except Exception as exc:  # noqa: BLE001 - worker must report and exit cleanly.
            self.error.emit(str(exc))
        finally:
            self.backend.abort()
            self.worker_finished.emit()

    def _run_continuous(self) -> None:
        s = self.settings
        n = max(4, s.n_block)
        hop = self._hop_frames if self._hop_frames > 0 else max(1, n // 8)
        chunk = max(_STREAM_CHUNK_FRAMES, n * 2)
        timeout_ms = s.frame_timeout_ms()
        # Persisted across acquisition chunks so the demodulation window stays
        # contiguous instead of restarting every block.
        roi_window: deque[float] = deque(maxlen=n)
        since_update = 0
        while self._running:
            self.backend.setup_kinetic(s.exposure_s, chunk, TriggerMode.EXTERNAL)
            self.backend.start()
            for _ in range(chunk):
                if not self._running:
                    break
                if not self.backend.wait_next_frame(timeout_ms):
                    self.error.emit(
                        f"No triggers detected after {timeout_ms / 1000:.1f} s — "
                        "check external trigger cabling."
                    )
                    self._running = False
                    break
                frame = self.backend.get_oldest_frame()
                roi_window.append(
                    float(
                        integrate_roi(
                            frame.reshape(1, -1),
                            s.pixel_start,
                            s.pixel_end,
                            s.roi_method,
                        )[0]
                    )
                )
                since_update += 1
                if since_update >= hop:
                    since_update = 0
                    self.frame_acquired.emit(frame.copy())
                    self._emit_window(roi_window)

    def _emit_window(self, roi_window: deque[float]) -> None:
        ts = np.fromiter(roi_window, dtype=np.float64, count=len(roi_window))
        self.block_complete.emit(ts)
        if ts.size >= 4:
            result = demodulate(
                ts,
                self.settings.trigger_frequency_hz,
                self.settings.f_expected,
                self.settings.f_search_halfwidth,
                self.settings.window,
            )
            self.demod_result.emit(result)

    def _run_single_block(self) -> None:
        s = self.settings
        timeout_ms = s.frame_timeout_ms()
        self.backend.setup_kinetic(s.exposure_s, s.n_block, TriggerMode.EXTERNAL)
        self.backend.start()
        # Accumulate only the frames actually acquired instead of preallocating
        # the full block, which can be very large.
        block_frames: list[np.ndarray] = []
        for idx in range(s.n_block):
            if not self._running:
                break
            if not self.backend.wait_next_frame(timeout_ms):
                self.error.emit(
                    f"No triggers detected after {timeout_ms / 1000:.1f} s — "
                    "check external trigger cabling."
                )
                self._running = False
                break
            frame = self.backend.get_oldest_frame()
            block_frames.append(frame)
            if idx == 0 or idx == s.n_block - 1:
                self.frame_acquired.emit(frame.copy())
        acquired = len(block_frames)
        if not acquired:
            return
        block = np.stack(block_frames, axis=0)
        roi_ts = integrate_roi(block, s.pixel_start, s.pixel_end, s.roi_method)
        self.block_complete.emit(roi_ts.copy())
        if acquired >= 4:
            result = demodulate(
                roi_ts,
                s.trigger_frequency_hz,
                s.f_expected,
                s.f_search_halfwidth,
                s.window,
            )
            self.demod_result.emit(result)


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
        all_frames: list[np.ndarray] = []
        demod_results: list[DemodResult] = []
        roi_all: list[float] = []      # full ROI time series (one value per frame)
        roi_buffer: list[float] = []   # running ROI accumulator for real-time blocking
        try:
            timeout_ms = self.settings.frame_timeout_ms()
            self.backend.setup_kinetic(
                self.settings.exposure_s,
                self.total_frames,
                TriggerMode.EXTERNAL,
            )
            self.backend.start()
            start_wall = time.monotonic()
            while self._running and len(all_frames) < self.total_frames:
                if not self.backend.wait_next_frame(timeout_ms):
                    self.error.emit(
                        f"No triggers detected after {timeout_ms / 1000:.1f} s — "
                        "check external trigger cabling."
                    )
                    break
                # Use get_oldest_frame for consistent one-frame-at-a-time retrieval;
                # this avoids the mixed-semantics bug with get_new_frames_batch.
                frame = self.backend.get_oldest_frame()
                all_frames.append(frame)
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
                roi_all.append(roi_val)
                roi_buffer.append(roi_val)

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
            # Reuse the ROI values already computed per frame in the loop above
            # rather than re-integrating every frame a second time.
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
            while self._running:
                self.backend.setup_kinetic(
                    self.exposure_s,
                    self.burst_frames,
                    TriggerMode.EXTERNAL,
                )
                self.backend.start()
                for _ in range(self.burst_frames):
                    if not self._running:
                        break
                    if not self.backend.wait_next_frame(timeout_ms):
                        self.error.emit(
                            f"No triggers detected after {timeout_ms / 1000:.1f} s — "
                            "check external trigger cabling."
                        )
                        self._running = False
                        break
                    frame = self.backend.get_oldest_frame().copy()
                    self.frame_acquired.emit(frame)
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
            self.error.emit(str(exc))
        finally:
            self.backend.abort()
            self.worker_finished.emit()
