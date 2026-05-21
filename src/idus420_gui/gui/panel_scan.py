"""2-D raster scan panel: controls SNOM stage + Andor acquisition."""

from __future__ import annotations

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

# harmonic key → (plot title, colormap, value label, index into demod_results)
_DEMOD_CHANNELS: dict[str, tuple[str, str, str, int]] = {
    "0w": ("Demod amp @ 0ω (DC)", "viridis", "Mean intensity", 0),
    "1w": ("Demod amp @ 1ω",      "viridis", "Peak amplitude", 1),
    "2w": ("Demod amp @ 2ω",      "viridis", "Peak amplitude", 2),
    "3w": ("Demod amp @ 3ω",      "viridis", "Peak amplitude", 3),
}

# channel key → (plot title, colormap, value label, extractor from SnomSample)
_SNOM_CHANNELS: dict[str, tuple[str, str, str, object]] = {
    "Z":   ("SNOM Z",   "viridis", "Z (nm)",      lambda s: s.xyz_nm[2]),
    "M1A": ("SNOM M1A", "viridis", "Amplitude",   lambda s: s.m_amp[1]),
    "M1P": ("SNOM M1P", "CET-C1",  "Phase (rad)", lambda s: s.m_phase[1]),
    "M2A": ("SNOM M2A", "viridis", "Amplitude",   lambda s: s.m_amp[2]),
    "M2P": ("SNOM M2P", "CET-C1",  "Phase (rad)", lambda s: s.m_phase[2]),
}


class ScanPanel(QWidget):
    """2-D XY raster scan panel using the SNOM Sample motor + Andor camera."""

    log_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, demod_source: DemodPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: CameraBackend | None = None
        self.demod_source = demod_source
        self.worker: ScanWorker | None = None
        self._stage: object | None = None  # NeaSnomBackend kept alive across scans

        self._map_demod: dict[str, np.ndarray] | None = None
        self._demod_slot_keys: list[str] = ["0w", "1w"]
        self._map_snom: dict[str, np.ndarray] | None = None
        self._snom_slot_keys: list[str] = ["M1A", "M1P"]
        self._scan_is_line: bool = False
        self._scan_line_coords: np.ndarray | None = None  # nm positions along varying axis

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

        # ---- Right: scan maps (2×2 grid) ----
        plot_container = QWidget()
        plot_lay = QVBoxLayout(plot_container)
        plot_lay.setContentsMargins(4, 4, 4, 4)
        self.map_widget = pg.GraphicsLayoutWidget()
        self.map_widget.setBackground(theme.BG)

        # Two generic demod slots (row 0) — harmonic chosen by selector below
        for slot_i, default_key in enumerate(self._demod_slot_keys):
            title, cmap, label, _ = _DEMOD_CHANNELS[default_key]
            plot = self.map_widget.addPlot(row=0, col=slot_i, title=title)
            theme._style_plot_item(plot)  # noqa: SLF001
            plot.setAspectLocked(True)
            img = pg.ImageItem()
            plot.addItem(img)
            try:
                cb = pg.ColorBarItem(colorMap=cmap, label=label)
            except Exception:  # noqa: BLE001
                cb = pg.ColorBarItem(colorMap="viridis", label=label)
            cb.setImageItem(img, insert_in=plot)
            setattr(self, f"_demod_plot_{slot_i}", plot)
            setattr(self, f"_demod_image_{slot_i}", img)
            setattr(self, f"_demod_cb_{slot_i}", cb)

        # Two generic SNOM slots (row 1) — channel chosen by selector below
        for slot_i, default_key in enumerate(self._snom_slot_keys):
            title, cmap, label, _ = _SNOM_CHANNELS[default_key]
            plot = self.map_widget.addPlot(row=1, col=slot_i, title=title)
            theme._style_plot_item(plot)  # noqa: SLF001
            plot.setAspectLocked(True)
            img = pg.ImageItem()
            plot.addItem(img)
            try:
                cb = pg.ColorBarItem(colorMap=cmap, label=label)
            except Exception:  # noqa: BLE001
                cb = pg.ColorBarItem(colorMap="viridis", label=label)
            cb.setImageItem(img, insert_in=plot)
            setattr(self, f"_snom_plot_{slot_i}", plot)
            setattr(self, f"_snom_image_{slot_i}", img)
            setattr(self, f"_snom_cb_{slot_i}", cb)

        plot_lay.addWidget(self.map_widget)

        # Demod channel selectors — one per slot, live-switchable during scan
        demod_sel_row = QHBoxLayout()
        demod_sel_row.setSpacing(8)
        self.demod_combo_0 = QComboBox()
        self.demod_combo_1 = QComboBox()
        for key, (ch_title, _c, _l, _idx) in _DEMOD_CHANNELS.items():
            self.demod_combo_0.addItem(ch_title, key)
            self.demod_combo_1.addItem(ch_title, key)
        self.demod_combo_0.setCurrentIndex(
            list(_DEMOD_CHANNELS.keys()).index(self._demod_slot_keys[0])
        )
        self.demod_combo_1.setCurrentIndex(
            list(_DEMOD_CHANNELS.keys()).index(self._demod_slot_keys[1])
        )
        demod_sel_row.addStretch(1)
        demod_sel_row.addWidget(QLabel("Demod plot 1:"))
        demod_sel_row.addWidget(self.demod_combo_0)
        demod_sel_row.addSpacing(24)
        demod_sel_row.addWidget(QLabel("Demod plot 2:"))
        demod_sel_row.addWidget(self.demod_combo_1)
        demod_sel_row.addStretch(1)
        plot_lay.addLayout(demod_sel_row)

        self.demod_combo_0.currentIndexChanged.connect(self._on_demod_channel_changed)
        self.demod_combo_1.currentIndexChanged.connect(self._on_demod_channel_changed)

        # SNOM channel selectors — one per slot, live-switchable during scan
        snom_sel_row = QHBoxLayout()
        snom_sel_row.setSpacing(8)
        self.snom_combo_0 = QComboBox()
        self.snom_combo_1 = QComboBox()
        for key, (ch_title, _c, _l, _e) in _SNOM_CHANNELS.items():
            self.snom_combo_0.addItem(ch_title, key)
            self.snom_combo_1.addItem(ch_title, key)
        self.snom_combo_0.setCurrentIndex(
            list(_SNOM_CHANNELS.keys()).index(self._snom_slot_keys[0])
        )
        self.snom_combo_1.setCurrentIndex(
            list(_SNOM_CHANNELS.keys()).index(self._snom_slot_keys[1])
        )
        snom_sel_row.addStretch(1)
        snom_sel_row.addWidget(QLabel("SNOM plot 1:"))
        snom_sel_row.addWidget(self.snom_combo_0)
        snom_sel_row.addSpacing(24)
        snom_sel_row.addWidget(QLabel("SNOM plot 2:"))
        snom_sel_row.addWidget(self.snom_combo_1)
        snom_sel_row.addStretch(1)
        plot_lay.addLayout(snom_sel_row)

        self.snom_combo_0.currentIndexChanged.connect(self._on_snom_channel_changed)
        self.snom_combo_1.currentIndexChanged.connect(self._on_snom_channel_changed)

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

        # Reuse the stage backend across scans — disconnect/reconnect between
        # scans corrupts the neaspec session state.
        if self._stage is None:
            self._stage = NeaSnomBackend()
        stage = self._stage

        nan = np.full((ny, nx), np.nan, dtype=np.float64)
        self._map_demod = {k: nan.copy() for k in _DEMOD_CHANNELS}
        self._demod_slot_keys = [
            self.demod_combo_0.currentData(),
            self.demod_combo_1.currentData(),
        ]
        self._map_snom = {k: nan.copy() for k in _SNOM_CHANNELS}
        self._snom_slot_keys = [
            self.snom_combo_0.currentData(),
            self.snom_combo_1.currentData(),
        ]

        self._scan_is_line = (nx == 1 or ny == 1)
        self._rebuild_plots(nx, ny, grid)
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

    def closeEvent(self, event: object) -> None:
        if self._stage is not None:
            try:
                self._stage.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._stage = None
        super().closeEvent(event)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Plot mode switching (image vs line)
    # ------------------------------------------------------------------

    def _rebuild_plots(self, nx: int, ny: int, grid: ScanGrid) -> None:
        """Switch each subplot between ImageItem (2D) and PlotDataItem (1D line)."""
        demod_specs = [
            (
                f"_demod_plot_{i}", f"_demod_image_{i}", f"_demod_cb_{i}", f"_demod_line_{i}",
                *_DEMOD_CHANNELS[key][:3],
            )
            for i, key in enumerate(self._demod_slot_keys)
        ]
        snom_specs = [
            (
                f"_snom_plot_{i}", f"_snom_image_{i}", f"_snom_cb_{i}", f"_snom_line_{i}",
                *_SNOM_CHANNELS[key][:3],
            )
            for i, key in enumerate(self._snom_slot_keys)
        ]
        all_specs = demod_specs + snom_specs

        if self._scan_is_line:
            # Build position axis along the varying dimension (nm)
            if ny > 1:
                step = grid.y_step_nm
                n = ny
                x_label = "Y position (nm)"
            else:
                step = grid.x_step_nm
                n = nx
                x_label = "X position (nm)"
            self._scan_line_coords = np.arange(n) * step

            for attr_plot, _attr_img, attr_cb, attr_line, title, _, y_label in all_specs:
                plot: pg.PlotItem = getattr(self, attr_plot)
                plot.clear()
                plot.setAspectLocked(False)
                plot.setTitle(title)
                plot.setLabel("bottom", x_label)
                plot.setLabel("left", y_label)
                line = plot.plot(
                    self._scan_line_coords,
                    np.full(n, np.nan),
                    pen=pg.mkPen(color="#4fc3f7", width=2),
                    symbol="o",
                    symbolSize=4,
                    symbolBrush="#4fc3f7",
                )
                setattr(self, attr_line, line)
                # Hide colorbar — not applicable in line mode
                cb = getattr(self, attr_cb)
                cb.hide()
        else:
            for attr_plot, attr_img, attr_cb, _attr_line, title, _cmap, _y_label in all_specs:
                plot = getattr(self, attr_plot)
                plot.clear()
                plot.setAspectLocked(True)
                plot.setTitle(title)
                img = pg.ImageItem()
                plot.addItem(img)
                cb = getattr(self, attr_cb)
                cb.setImageItem(img, insert_in=plot)
                cb.show()
                setattr(self, attr_img, img)
                img.setImage(np.zeros((ny, nx)).T)

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
        if self._map_demod is None:
            return
        iy, ix = result.point.iy, result.point.ix
        dr = result.demod_results
        for key, (_t, _c, _l, idx) in _DEMOD_CHANNELS.items():
            self._map_demod[key][iy, ix] = (
                dr[idx].peak_amplitude
                if len(dr) > idx and dr[idx] is not None else np.nan
            )
        if result.snom_samples and self._map_snom is not None:
            for key, (_t, _c, _l, extract) in _SNOM_CHANNELS.items():
                self._map_snom[key][iy, ix] = float(
                    np.nanmean([extract(s) for s in result.snom_samples])  # type: ignore[operator]
                )
        self._render_demod_slots()
        self._render_snom_slots()

    def _render_demod_slots(self) -> None:
        """Repaint both demod plot slots from the stored harmonic maps."""
        if self._map_demod is None:
            return
        for i, key in enumerate(self._demod_slot_keys):
            arr = self._map_demod[key]
            if self._scan_is_line:
                line = getattr(self, f"_demod_line_{i}", None)
                if line is None:
                    continue
                coords = self._scan_line_coords
                data = arr[:, 0] if arr.shape[1] == 1 else arr[0, :]
                line.setData(coords, data)
            else:
                img = getattr(self, f"_demod_image_{i}", None)
                if img is not None:
                    img.setImage(arr.T)

    def _restyle_demod_slots(self) -> None:
        """Update titles, colormaps and colorbar labels for both demod slots."""
        for i, key in enumerate(self._demod_slot_keys):
            title, cmap, label, _ = _DEMOD_CHANNELS[key]
            plot: pg.PlotItem = getattr(self, f"_demod_plot_{i}")
            plot.setTitle(title)
            cb = getattr(self, f"_demod_cb_{i}")
            try:
                cb.setColorMap(cmap)
            except Exception:  # noqa: BLE001
                cb.setColorMap("viridis")
            cb.setLabel(label)

    def _on_demod_channel_changed(self) -> None:
        """Handle live change of either demod channel selector."""
        self._demod_slot_keys = [
            self.demod_combo_0.currentData(),
            self.demod_combo_1.currentData(),
        ]
        self._restyle_demod_slots()
        if self._map_demod is not None:
            self._render_demod_slots()

    def _render_snom_slots(self) -> None:
        """Repaint both SNOM plot slots from the stored channel maps."""
        if self._map_snom is None:
            return
        for i, key in enumerate(self._snom_slot_keys):
            arr = self._map_snom[key]
            if self._scan_is_line:
                line = getattr(self, f"_snom_line_{i}", None)
                if line is None:
                    continue
                coords = self._scan_line_coords
                if arr.shape[1] == 1:
                    data = arr[:, 0]
                else:
                    data = arr[0, :]
                line.setData(coords, data)
            else:
                img = getattr(self, f"_snom_image_{i}", None)
                if img is not None:
                    img.setImage(arr.T)

    def _restyle_snom_slots(self) -> None:
        """Update titles, colormaps and colorbar labels for both SNOM slots."""
        for i, key in enumerate(self._snom_slot_keys):
            title, cmap, label, _ = _SNOM_CHANNELS[key]
            plot: pg.PlotItem = getattr(self, f"_snom_plot_{i}")
            plot.setTitle(title)
            cb = getattr(self, f"_snom_cb_{i}")
            try:
                cb.setColorMap(cmap)
            except Exception:  # noqa: BLE001
                cb.setColorMap("viridis")
            cb.setLabel(label)

    def _on_snom_channel_changed(self) -> None:
        """Handle live change of either SNOM channel selector."""
        self._snom_slot_keys = [
            self.snom_combo_0.currentData(),
            self.snom_combo_1.currentData(),
        ]
        self._restyle_snom_slots()
        if self._map_snom is not None:
            self._render_snom_slots()

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
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/demod_combo_0", self.demod_combo_0.currentIndex())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/demod_combo_1", self.demod_combo_1.currentIndex())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/snom_combo_0", self.snom_combo_0.currentIndex())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/snom_combo_1", self.snom_combo_1.currentIndex())

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

        if (v := _f("demod_combo_0")) is not None:
            self.demod_combo_0.setCurrentIndex(int(v))
        if (v := _f("demod_combo_1")) is not None:
            self.demod_combo_1.setCurrentIndex(int(v))
        self._demod_slot_keys = [
            self.demod_combo_0.currentData(),
            self.demod_combo_1.currentData(),
        ]
        self._restyle_demod_slots()
        if (v := _f("snom_combo_0")) is not None:
            self.snom_combo_0.setCurrentIndex(int(v))
        if (v := _f("snom_combo_1")) is not None:
            self.snom_combo_1.setCurrentIndex(int(v))
        self._snom_slot_keys = [
            self.snom_combo_0.currentData(),
            self.snom_combo_1.currentData(),
        ]
        self._restyle_snom_slots()
        self._update_total_label()


def _lbl(text: str) -> QLabel:
    return QLabel(text)
