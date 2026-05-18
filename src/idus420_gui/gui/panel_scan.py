"""2-D raster scan panel: controls SNOM stage + Andor acquisition."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from idus420_gui.camera.base import CameraBackend
from idus420_gui.gui import theme
from idus420_gui.gui.panel_demod import DemodPanel
from idus420_gui.io.save import save_scan_h5
from idus420_gui.motion.base import ScanGrid
from idus420_gui.workers.scan import PointResult, ScanResult, ScanWorker

try:
    from idus420_gui.motion.nea_snom import NEA_TOOLS_AVAILABLE, NeaSnomBackend
except ImportError:
    NEA_TOOLS_AVAILABLE = False
    NeaSnomBackend = None  # type: ignore[assignment,misc]

_SETTINGS_KEY_PREFIX = "scan_panel"


class ScanPanel(QWidget):
    """2-D XY raster scan panel using the SNOM Sample motor + Andor camera."""

    log_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, demod_source: DemodPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: CameraBackend | None = None
        self.demod_source = demod_source
        self.worker: ScanWorker | None = None

        self._scan_map_data: np.ndarray | None = None  # (ny, nx) peak amplitudes

        self._build_ui()
        self._restore_settings()

        if not NEA_TOOLS_AVAILABLE:
            self._show_error(
                "SNOM stage backend unavailable: nea_tools / nest_asyncio not installed. "
                "Install with:  pip install 'idus420_gui[snom]'  —  "
                "You can edit scan parameters but cannot start a scan."
            )

    def set_backend(self, backend: CameraBackend | None) -> None:
        self.backend = backend

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ---- Left: controls ----
        ctrl_container = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_container)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)
        ctrl_layout.setSpacing(8)

        # Error banner (hidden by default)
        self.error_banner = QLabel()
        self.error_banner.setWordWrap(True)
        self.error_banner.setStyleSheet(
            f"color: {theme.ACCENT_ERR};"
            f"background-color: {theme.SURFACE_ALT};"
            f"border: 1px solid {theme.ACCENT_ERR};"
            "border-radius: 4px;"
            "padding: 6px;"
        )
        self.error_banner.hide()
        ctrl_layout.addWidget(self.error_banner)

        # SNOM connection
        conn_box = QGroupBox("SNOM Connection")
        conn_grid = QGridLayout(conn_box)
        conn_grid.setSpacing(6)
        conn_grid.setColumnStretch(1, 1)
        self.snom_host = QLineEdit("nea-server")
        conn_grid.addWidget(_lbl("Host"), 0, 0)
        conn_grid.addWidget(self.snom_host, 0, 1)
        ctrl_layout.addWidget(conn_box)

        # Scan grid
        grid_box = QGroupBox("Scan Grid")
        gg = QGridLayout(grid_box)
        gg.setSpacing(6)
        gg.setColumnStretch(1, 1)
        gg.setColumnStretch(3, 1)

        self.x_start = QDoubleSpinBox()
        self.x_start.setRange(-1e9, 1e9)
        self.x_start.setDecimals(1)
        self.x_start.setValue(50000.0)   # default: 50 µm
        self.x_start.setSuffix(" nm")

        self.x_step = QDoubleSpinBox()
        self.x_step.setRange(0.1, 1e9)
        self.x_step.setDecimals(1)
        self.x_step.setValue(1000.0)
        self.x_step.setSuffix(" nm")

        self.nx = QSpinBox()
        self.nx.setRange(1, 10000)
        self.nx.setValue(5)

        self.y_start = QDoubleSpinBox()
        self.y_start.setRange(-1e9, 1e9)
        self.y_start.setDecimals(1)
        self.y_start.setValue(50000.0)   # default: 50 µm
        self.y_start.setSuffix(" nm")

        self.y_step = QDoubleSpinBox()
        self.y_step.setRange(0.1, 1e9)
        self.y_step.setDecimals(1)
        self.y_step.setValue(1000.0)
        self.y_step.setSuffix(" nm")

        self.ny = QSpinBox()
        self.ny.setRange(1, 10000)
        self.ny.setValue(5)

        self.angle = QDoubleSpinBox()
        self.angle.setRange(-180.0, 180.0)
        self.angle.setDecimals(1)
        self.angle.setValue(0.0)
        self.angle.setSuffix(" °")

        self.total_label = QLabel("Total: 25 points")

        row = 0
        gg.addWidget(_lbl("X start"), row, 0)
        gg.addWidget(self.x_start, row, 1)
        gg.addWidget(_lbl("X step"), row, 2)
        gg.addWidget(self.x_step, row, 3)
        row += 1
        gg.addWidget(_lbl("X points"), row, 0)
        gg.addWidget(self.nx, row, 1)
        row += 1
        gg.addWidget(_lbl("Y start"), row, 0)
        gg.addWidget(self.y_start, row, 1)
        gg.addWidget(_lbl("Y step"), row, 2)
        gg.addWidget(self.y_step, row, 3)
        row += 1
        gg.addWidget(_lbl("Y points"), row, 0)
        gg.addWidget(self.ny, row, 1)
        row += 1
        gg.addWidget(_lbl("Angle"), row, 0)
        gg.addWidget(self.angle, row, 1)
        row += 1
        gg.addWidget(self.total_label, row, 0, 1, 4)
        ctrl_layout.addWidget(grid_box)

        # Scan order
        order_box = QGroupBox("Scan Order")
        order_lay = QHBoxLayout(order_box)
        self.order_combo = QComboBox()
        self.order_combo.addItem("Snake (boustrophedon)", "snake")
        self.order_combo.addItem("Raster left-to-right", "raster_lr")
        order_lay.addWidget(self.order_combo)
        ctrl_layout.addWidget(order_box)

        # Per-point acquisition
        acq_box = QGroupBox("Per-Point Acquisition")
        acq_g = QGridLayout(acq_box)
        acq_g.setSpacing(6)
        acq_g.setColumnStretch(1, 1)

        self.frames_per_point = QSpinBox()
        self.frames_per_point.setRange(4, 100_000_000)
        self.frames_per_point.setValue(100)
        acq_g.addWidget(_lbl("Frames / point"), 0, 0)
        acq_g.addWidget(self.frames_per_point, 0, 1)
        ctrl_layout.addWidget(acq_box)

        # Output
        out_box = QGroupBox("Output")
        out_g = QGridLayout(out_box)
        out_g.setSpacing(6)
        out_g.setColumnStretch(1, 1)

        self.output_dir = QLineEdit(str(Path.cwd()))
        choose_btn = QPushButton("Browse")
        choose_btn.setFixedWidth(70)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.output_dir)
        dir_row.addWidget(choose_btn)
        self.stem = QLineEdit("snom_scan")

        out_g.addWidget(_lbl("Output dir"), 0, 0)
        out_g.addLayout(dir_row, 0, 1)
        out_g.addWidget(_lbl("Filename stem"), 1, 0)
        out_g.addWidget(self.stem, 1, 1)
        ctrl_layout.addWidget(out_box)

        # Controls + progress
        run_box = QGroupBox("Run")
        run_lay = QVBoxLayout(run_box)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Scan")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        run_lay.addLayout(btn_row)

        self.progress = QProgressBar()
        self.status_label = QLabel("Idle")
        run_lay.addWidget(self.progress)
        run_lay.addWidget(self.status_label)
        ctrl_layout.addWidget(run_box)

        ctrl_layout.addStretch(1)
        ctrl_container.setMinimumWidth(300)
        ctrl_container.setMaximumWidth(540)
        splitter.addWidget(ctrl_container)

        # ---- Right: scan map ----
        plot_container = QWidget()
        plot_lay = QVBoxLayout(plot_container)
        plot_lay.setContentsMargins(4, 4, 4, 4)
        self.map_widget = pg.GraphicsLayoutWidget()
        self.map_widget.setBackground(theme.BG)
        self.map_plot = self.map_widget.addPlot(title="Scan map (demod peak amplitude)")
        theme._style_plot_item(self.map_plot)  # noqa: SLF001
        self.map_plot.setAspectLocked(True)
        self.map_image = pg.ImageItem()
        self.map_plot.addItem(self.map_image)
        self.map_colorbar = pg.ColorBarItem(
            colorMap="viridis",
            label="Peak amplitude",
        )
        self.map_colorbar.setImageItem(self.map_image, insert_in=self.map_plot)
        plot_lay.addWidget(self.map_widget)
        splitter.addWidget(plot_container)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 800])
        outer.addWidget(splitter, stretch=1)

        # Signal wiring
        choose_btn.clicked.connect(self._choose_directory)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        for w in (self.nx, self.ny):
            w.valueChanged.connect(self._update_total_label)

    # ------------------------------------------------------------------
    # Scan control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not NEA_TOOLS_AVAILABLE or NeaSnomBackend is None:
            self._show_error(
                "Cannot start scan: SNOM stage backend unavailable. "
                "Install with:  pip install 'idus420_gui[snom]'"
            )
            return
        if not self.backend:
            self._show_error(
                "No camera backend connected — cannot start scan."
            )
            return
        if not self.demod_source._validate_roi():  # noqa: SLF001
            return
        self._clear_error()

        nx, ny = self.nx.value(), self.ny.value()
        grid = ScanGrid(
            x_start_nm=self.x_start.value(),
            y_start_nm=self.y_start.value(),
            x_step_nm=self.x_step.value(),
            y_step_nm=self.y_step.value(),
            nx=nx,
            ny=ny,
            order=self.order_combo.currentData(),
            angle_deg=self.angle.value(),
        )
        settings = self.demod_source.settings()
        # Override n_block to match user-requested frames per point.
        from dataclasses import replace  # noqa: PLC0415
        settings = replace(settings, n_block=self.frames_per_point.value())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pending_stem = f"{self.stem.text()}_{timestamp}"
        self._pending_out_dir = Path(self.output_dir.text())

        metadata: dict[str, Any] = {
            "software": "idus420_gui",
            "timestamp": timestamp,
            "snom_host": self.snom_host.text(),
            "detector_size": self.backend.detector_size() if self.backend else None,
            "serial_number": self.backend.serial_number() if self.backend else None,
            "sdk_version": self.backend.sdk_version() if self.backend else None,
            "roi_pixel_start": settings.pixel_start,
            "roi_pixel_end": settings.pixel_end,
            "roi_method": settings.roi_method,
            "exposure_s": settings.exposure_s,
            "trigger_frequency_hz": settings.trigger_frequency_hz,
            "n_block": settings.n_block,
            "f_expected_hz": settings.f_expected,
            "f_search_halfwidth_hz": settings.f_search_halfwidth,
            "window": settings.window,
        }
        metadata["snom_host"] = self.snom_host.text()

        stage = NeaSnomBackend()

        self._scan_map_data = np.zeros((ny, nx), dtype=np.float64)
        self.map_image.setImage(self._scan_map_data.T)
        self.progress.setMaximum(grid.total_points())
        self.progress.setValue(0)

        self.worker = ScanWorker(self.backend, stage, grid, settings, metadata)
        self.worker.point_started.connect(self._on_point_started)
        self.worker.point_finished.connect(self._on_point_finished)
        self.worker.point_data_ready.connect(self._on_point_data)
        self.worker.scan_finished.connect(self._on_scan_finished)
        self.worker.error.connect(self._show_error)
        self.worker.worker_finished.connect(lambda: self._set_running_ui(False))

        self._set_running_ui(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_point_started(self, point: object) -> None:
        self.status_label.setText(
            f"Moving to point ({point.ix}, {point.iy})  "  # type: ignore[attr-defined]
            f"x={point.x_nm:.0f} nm  y={point.y_nm:.0f} nm"  # type: ignore[attr-defined]
        )

    def _on_point_finished(self, current: int, total: int) -> None:
        self.progress.setValue(current)
        self.status_label.setText(f"Completed {current} / {total} points")

    def _on_point_data(self, point_index: int, result: PointResult) -> None:
        if self._scan_map_data is None:
            return
        iy, ix = result.point.iy, result.point.ix
        if result.demod_results:
            amp = result.demod_results[-1].peak_amplitude
        else:
            amp = float(np.mean(result.roi_timeseries)) if len(result.roi_timeseries) else 0.0
        self._scan_map_data[iy, ix] = amp
        self.map_image.setImage(self._scan_map_data.T)

    def _on_scan_finished(self, result: ScanResult) -> None:
        if not result.point_results:
            self.log_message.emit("Scan finished with no data — nothing saved.")
            return
        out_dir = getattr(self, "_pending_out_dir", Path.cwd())
        stem = getattr(self, "_pending_stem", "snom_scan")
        path = out_dir / f"{stem}.h5"

        metadata = dict(result.metadata)
        try:
            save_scan_h5(path, result, metadata)
            self.log_message.emit(f"Scan saved to {path}")
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"Scan save failed: {exc}")
        self._save_settings()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_error(self, msg: str) -> None:
        self.error_banner.setText(msg)
        self.error_banner.show()
        self.log_message.emit(msg)

    def _clear_error(self) -> None:
        self.error_banner.hide()
        self.error_banner.clear()

    def _set_running_ui(self, running: bool) -> None:
        for w in [
            self.snom_host,
            self.x_start, self.x_step, self.nx,
            self.y_start, self.y_step, self.ny,
            self.angle,
            self.order_combo,
            self.frames_per_point,
            self.output_dir, self.stem,
            self.start_btn,
        ]:
            w.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.running_changed.emit(running)

    def _update_total_label(self) -> None:
        self.total_label.setText(f"Total: {self.nx.value() * self.ny.value()} points")

    def _choose_directory(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output directory", self.output_dir.text())
        if d:
            self.output_dir.setText(d)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_settings(self) -> None:
        s = QSettings("idus420_gui", "ScanPanel")
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/snom_host", self.snom_host.text())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/x_start", self.x_start.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/x_step", self.x_step.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/nx", self.nx.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/y_start", self.y_start.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/y_step", self.y_step.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/ny", self.ny.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/angle", self.angle.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/order", self.order_combo.currentIndex())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/frames_per_point", self.frames_per_point.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/output_dir", self.output_dir.text())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/stem", self.stem.text())

    def _restore_settings(self) -> None:
        s = QSettings("idus420_gui", "ScanPanel")

        def _f(key: str) -> str | None:
            return s.value(f"{_SETTINGS_KEY_PREFIX}/{key}")  # type: ignore[return-value]

        if (v := _f("snom_host")) is not None:
            self.snom_host.setText(str(v))
        if (v := _f("x_start")) is not None:
            self.x_start.setValue(float(v))
        if (v := _f("x_step")) is not None:
            self.x_step.setValue(float(v))
        if (v := _f("nx")) is not None:
            self.nx.setValue(int(v))
        if (v := _f("y_start")) is not None:
            self.y_start.setValue(float(v))
        if (v := _f("y_step")) is not None:
            self.y_step.setValue(float(v))
        if (v := _f("ny")) is not None:
            self.ny.setValue(int(v))
        if (v := _f("angle")) is not None:
            self.angle.setValue(float(v))
        if (v := _f("order")) is not None:
            self.order_combo.setCurrentIndex(int(v))
        if (v := _f("frames_per_point")) is not None:
            self.frames_per_point.setValue(int(v))
        if (v := _f("output_dir")) is not None:
            self.output_dir.setText(str(v))
        if (v := _f("stem")) is not None:
            self.stem.setText(str(v))

        self._update_total_label()


def _lbl(text: str) -> QLabel:
    return QLabel(text)
