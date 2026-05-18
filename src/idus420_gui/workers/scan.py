"""Scan worker: orchestrates per-point stage movement + Andor + SNOM acquisition."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from idus420_gui.camera.base import CameraBackend, TriggerMode
from idus420_gui.motion.base import ScanGrid, SnomSample, StageBackend, StagePoint
from idus420_gui.processing.demodulation import DemodResult, demodulate
from idus420_gui.processing.roi import integrate_roi
from idus420_gui.workers.acquisition import DemodulationSettings


@dataclass
class PointResult:
    """All data collected at a single scan point."""

    point: StagePoint
    actual_xyz_nm: tuple[float, float, float]
    frames: np.ndarray                   # (n_frames, frame_width) uint16
    roi_timeseries: np.ndarray           # (n_frames,) float64
    demod_results: list[DemodResult]
    snom_samples: list[SnomSample]       # one sample interleaved after each Andor frame


@dataclass
class ScanResult:
    """All data from a completed (or aborted) scan."""

    grid: ScanGrid
    settings: DemodulationSettings
    point_results: list[PointResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ScanWorker(QThread):
    """Drives a 2-D raster scan: moves stage, acquires Andor frames, reads SNOM signals."""

    point_started = pyqtSignal(object)           # StagePoint
    point_finished = pyqtSignal(int, int)        # current_point_index, total_points
    snom_sample_ready = pyqtSignal(object)       # SnomSample (for live display)
    point_data_ready = pyqtSignal(int, object)   # point_index, PointResult
    scan_finished = pyqtSignal(object)           # ScanResult
    error = pyqtSignal(str)
    worker_finished = pyqtSignal()

    def __init__(
        self,
        camera_backend: CameraBackend,
        stage_backend: StageBackend,
        grid: ScanGrid,
        settings: DemodulationSettings,
        metadata: dict[str, Any] | None = None,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera = camera_backend
        self.stage = stage_backend
        self.grid = grid
        self.settings = settings
        self.metadata = metadata or {}
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.camera.abort()

    def run(self) -> None:
        result = ScanResult(
            grid=self.grid,
            settings=self.settings,
            metadata=dict(self.metadata),
        )
        try:
            self.stage.connect(self.metadata.get("snom_host", "nea-server"))
            t0_scan = time.monotonic()
            total = self.grid.total_points()
            point_index = 0

            for point in self.grid.points():
                if not self._running:
                    break

                self.point_started.emit(point)
                self.stage.goto_xy_nm(point.x_nm, point.y_nm)
                actual_xyz = self.stage.read_xyz_nm()

                n_frames = int(self.settings.n_block)
                timeout_ms = self.settings.frame_timeout_ms()
                self.camera.setup_kinetic(
                    self.settings.exposure_s,
                    n_frames,
                    TriggerMode.EXTERNAL,
                )
                self.camera.start()

                frames: list[np.ndarray] = []
                snom_samples: list[SnomSample] = []
                roi_buffer: list[float] = []
                demod_results: list[DemodResult] = []
                t0_point = time.monotonic()

                while self._running and len(frames) < n_frames:
                    if not self.camera.wait_next_frame(timeout_ms):
                        self.error.emit(
                            f"Point ({point.ix},{point.iy}): no triggers after "
                            f"{timeout_ms / 1000:.1f} s — check external trigger."
                        )
                        self._running = False
                        break

                    frame = self.camera.get_oldest_frame()
                    frames.append(frame)

                    t_sample = time.monotonic() - t0_scan
                    sample = self.stage.read_sample(t_sample)
                    snom_samples.append(sample)
                    self.snom_sample_ready.emit(sample)

                    roi_val = float(
                        integrate_roi(
                            frame.reshape(1, -1),
                            self.settings.pixel_start,
                            self.settings.pixel_end,
                            self.settings.roi_method,
                        )[0]
                    )
                    roi_buffer.append(roi_val)

                    if len(roi_buffer) == self.settings.n_block:
                        chunk = np.asarray(roi_buffer, dtype=np.float64)
                        roi_buffer.clear()
                        if len(chunk) >= 4:
                            try:
                                dr = demodulate(
                                    chunk,
                                    self.settings.trigger_frequency_hz,
                                    self.settings.f_expected,
                                    self.settings.f_search_halfwidth,
                                    self.settings.window,
                                )
                                demod_results.append(dr)
                            except Exception:  # noqa: BLE001
                                pass

                frames_arr = (
                    np.stack(frames, axis=0)
                    if frames
                    else np.empty((0, self.camera.frame_width()), dtype=np.uint16)
                )
                roi_arr = np.asarray(
                    [
                        float(
                            integrate_roi(
                                f.reshape(1, -1),
                                self.settings.pixel_start,
                                self.settings.pixel_end,
                                self.settings.roi_method,
                            )[0]
                        )
                        for f in frames
                    ],
                    dtype=np.float64,
                ) if frames else np.empty(0, dtype=np.float64)

                pt_result = PointResult(
                    point=point,
                    actual_xyz_nm=actual_xyz,
                    frames=frames_arr,
                    roi_timeseries=roi_arr,
                    demod_results=demod_results,
                    snom_samples=snom_samples,
                )
                result.point_results.append(pt_result)
                self.point_data_ready.emit(point_index, pt_result)
                point_index += 1
                self.point_finished.emit(point_index, total)

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.camera.abort()
            try:
                self.stage.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self.scan_finished.emit(result)
            self.worker_finished.emit()
