"""Demodulation Alignment panel."""

from __future__ import annotations

from typing import Literal, cast

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from idus420_gui.camera.base import CameraBackend
from idus420_gui.gui import theme
from idus420_gui.gui._helpers import PanelSettings, roi_error_message, stop_worker
from idus420_gui.gui.widgets import ReadoutLabel
from idus420_gui.processing.demodulation import DemodResult
from idus420_gui.workers.acquisition import DemodulationSettings, DemodulationWorker

_SETTINGS_KEY_PREFIX = "demod_panel"


class DemodPanel(QWidget):
    """Continuous alignment/demodulation view."""

    log_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: CameraBackend | None = None
        self._detector_width: int = 100_000
        self.worker: DemodulationWorker | None = None
        self.peak_history: list[float] = []
        self._build_ui()
        self._restore_settings()

    def set_backend(self, backend: CameraBackend | None) -> None:
        self.backend = backend
        self.set_frame_width(
            backend.frame_width() if backend is not None and backend.is_connected() else 100_000
        )

    def set_frame_width(self, width: int) -> None:
        self._detector_width = max(1, int(width))
        self.roi_start.setMaximum(self._detector_width - 1)
        self.roi_end.setMaximum(self._detector_width - 1)
        self.roi_start.setValue(min(self.roi_start.value(), self._detector_width - 1))
        self.roi_end.setValue(min(self.roi_end.value(), self._detector_width - 1))
        self._update_roi_region()

    def set_exposure(self, value: float) -> None:
        self.exposure_spin.setValue(value)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self.readout = ReadoutLabel("Peak: -- | Frequency: -- | SNR: --")
        outer.addWidget(self.readout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # --- Controls pane (single column) ---
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        controls_box = QGroupBox("Demodulation Parameters")
        grid = QGridLayout(controls_box)
        grid.setSpacing(6)
        grid.setColumnStretch(1, 1)

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setDecimals(6)
        self.exposure_spin.setRange(0.000001, 1000)
        self.exposure_spin.setValue(0.001)
        self.exposure_spin.setSuffix(" s")

        self.trigger_spin = QDoubleSpinBox()
        self.trigger_spin.setRange(0.001, 1_000_000)
        self.trigger_spin.setValue(500.0)
        self.trigger_spin.setSuffix(" Hz")

        self.roi_start = QSpinBox()
        self.roi_start.setRange(0, 100_000)
        self.roi_start.setValue(480)

        self.roi_end = QSpinBox()
        self.roi_end.setRange(0, 100_000)
        self.roi_end.setValue(560)

        self.roi_method = QComboBox()
        self.roi_method.addItems(["sum", "mean"])

        self.n_block = QSpinBox()
        self.n_block.setRange(4, 1_000_000)
        self.n_block.setValue(512)

        self.expected = QDoubleSpinBox()
        self.expected.setRange(0, 1_000_000)
        self.expected.setValue(37.0)
        self.expected.setSuffix(" Hz")

        self.search = QDoubleSpinBox()
        self.search.setRange(0, 1_000_000)
        self.search.setValue(5.0)
        self.search.setSuffix(" Hz")

        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "blackman", "none"])

        self.resolution_label = QLabel("--")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        # Single-column parameter grid (label | widget)
        row = 0
        grid.addWidget(_lbl("Exposure time"), row, 0)
        grid.addWidget(self.exposure_spin, row, 1)
        row += 1
        grid.addWidget(_lbl("Trigger freq."), row, 0)
        grid.addWidget(self.trigger_spin, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI start (px)"), row, 0)
        grid.addWidget(self.roi_start, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI end (px)"), row, 0)
        grid.addWidget(self.roi_end, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI method"), row, 0)
        grid.addWidget(self.roi_method, row, 1)
        row += 1
        grid.addWidget(_lbl("Frames / FFT block"), row, 0)
        grid.addWidget(self.n_block, row, 1)
        row += 1
        grid.addWidget(_lbl("Expected freq."), row, 0)
        grid.addWidget(self.expected, row, 1)
        row += 1
        grid.addWidget(_lbl("Search ½-width"), row, 0)
        grid.addWidget(self.search, row, 1)
        row += 1
        grid.addWidget(_lbl("Window"), row, 0)
        grid.addWidget(self.window_combo, row, 1)
        row += 1
        grid.addWidget(_lbl("Freq. resolution"), row, 0)
        grid.addWidget(self.resolution_label, row, 1)
        row += 1

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        grid.addLayout(buttons, row, 0, 1, 2)

        controls_layout.addWidget(controls_box)
        controls_layout.addStretch(1)
        controls_container.setMinimumWidth(240)
        controls_container.setMaximumWidth(320)

        splitter.addWidget(controls_container)

        # --- Plot pane (3-row stack) ---
        # Row 0: live spectrum (full width)
        # Row 1: ROI time series (left, small) | FFT magnitude (right, large)
        # Row 2: running peak amplitude (full width)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(theme.BG)

        self.spectrum_plot = self.plot_widget.addPlot(
            row=0, col=0, colspan=2, title="Live spectrum"
        )
        self.time_plot = self.plot_widget.addPlot(row=1, col=0, title="ROI time series")
        self.fft_plot = self.plot_widget.addPlot(row=1, col=1, title="FFT magnitude")
        self.history_plot = self.plot_widget.addPlot(
            row=2, col=0, colspan=2, title="Running peak amplitude"
        )

        # FFT wider than time series; row heights balanced
        self.plot_widget.ci.layout.setColumnStretchFactor(0, 1)
        self.plot_widget.ci.layout.setColumnStretchFactor(1, 3)
        self.plot_widget.ci.layout.setRowStretchFactor(0, 2)
        self.plot_widget.ci.layout.setRowStretchFactor(1, 3)
        self.plot_widget.ci.layout.setRowStretchFactor(2, 2)

        for pi in (self.spectrum_plot, self.time_plot, self.fft_plot, self.history_plot):
            theme.style_plot_item(pi)

        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen(theme.CURVE_YELLOW, width=1.5)
        )
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(theme.CURVE_CYAN, width=1.5))
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen(theme.CURVE_MAGENTA, width=1.5))
        self.history_curve = self.history_plot.plot(
            pen=pg.mkPen(theme.CURVE_GREEN, width=1.5)
        )

        # ROI shaded region on live spectrum
        roi_color = QColor(theme.ACCENT)
        roi_color.setAlpha(40)
        self.roi_region = pg.LinearRegionItem(
            values=(self.roi_start.value(), self.roi_end.value()),
            movable=False,
            brush=pg.mkBrush(roi_color),
            pen=pg.mkPen(theme.ACCENT, width=1),
        )
        self.spectrum_plot.addItem(self.roi_region)

        # Vertical peak-frequency line on FFT plot
        self.fft_peak_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen(theme.ACCENT_WARN, width=1, style=Qt.PenStyle.DashLine),
        )
        self.fft_plot.addItem(self.fft_peak_line)

        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 900])

        outer.addWidget(splitter, stretch=1)

        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.trigger_spin.valueChanged.connect(self._update_resolution)
        self.n_block.valueChanged.connect(self._update_resolution)
        self.roi_start.valueChanged.connect(self._update_roi_region)
        self.roi_end.valueChanged.connect(self._update_roi_region)
        self._update_resolution()

    def start(self) -> None:
        if not self.backend:
            self.log_message.emit("No camera backend is connected.")
            return
        if not self.validate_roi():
            return
        # Ensure any previous worker has fully stopped before starting a new one.
        if self.worker is not None and self.worker.isRunning():
            stop_worker(self.worker)
        self.peak_history.clear()
        self.worker = DemodulationWorker(self.backend, self.settings(), continuous=True)
        self.worker.frame_acquired.connect(lambda frame: self.spectrum_curve.setData(frame))
        self.worker.block_complete.connect(lambda ts: self.time_curve.setData(ts))
        self.worker.demod_result.connect(self._handle_result)
        self.worker.error.connect(self.log_message.emit)
        self.worker.worker_finished.connect(self._on_worker_finished)
        self._set_running_ui(True)
        self.worker.start()

    def stop(self) -> None:
        # The Stop UI state is applied when the worker confirms it has finished.
        if self.worker:
            self.worker.stop()

    def _on_worker_finished(self) -> None:
        # Ignore a late signal from a worker that has already been replaced.
        if self.sender() is not self.worker:
            return
        self.worker = None
        self._set_running_ui(False)

    def settings(self) -> DemodulationSettings:
        return DemodulationSettings(
            exposure_s=self.exposure_spin.value(),
            trigger_frequency_hz=self.trigger_spin.value(),
            pixel_start=self.roi_start.value(),
            pixel_end=self.roi_end.value(),
            roi_method=cast(Literal["sum", "mean"], self.roi_method.currentText()),
            n_block=self.n_block.value(),
            f_expected=self.expected.value(),
            f_search_halfwidth=self.search.value(),
            window=cast(
                Literal["hann", "blackman", "none"], self.window_combo.currentText()
            ),
        )

    def validate_roi(self) -> bool:
        """Validate the ROI; emit a log message and return ``False`` if invalid."""
        error = roi_error_message(
            self.roi_start.value(), self.roi_end.value(), self._detector_width
        )
        if error:
            self.log_message.emit(error)
            return False
        return True

    def _handle_result(self, result: DemodResult) -> None:
        self.fft_curve.setData(result.f_axis, result.spectrum)
        self.fft_peak_line.setPos(float(result.peak_frequency))
        self.peak_history.append(float(result.peak_amplitude))
        self.peak_history = self.peak_history[-600:]
        self.history_curve.setData(self.peak_history)
        self.readout.setText(
            f"Peak: {result.peak_amplitude:.4g} | "
            f"Frequency: {result.peak_frequency:.4g} Hz | "
            f"SNR: {result.snr:.3g}"
        )

    def _update_roi_region(self) -> None:
        self.roi_region.setRegion((self.roi_start.value(), self.roi_end.value()))

    def _set_running_ui(self, running: bool) -> None:
        for widget in [
            self.exposure_spin,
            self.trigger_spin,
            self.roi_start,
            self.roi_end,
            self.roi_method,
            self.n_block,
            self.expected,
            self.search,
            self.window_combo,
            self.start_button,
        ]:
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if not running:
            self._save_settings()
        self.running_changed.emit(running)

    def _update_resolution(self) -> None:
        resolution = self.trigger_spin.value() / self.n_block.value()
        self.resolution_label.setText(f"{resolution:.6g} Hz/bin")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_settings(self) -> None:
        s = PanelSettings("DemodPanel", _SETTINGS_KEY_PREFIX)
        s.set("exposure_s", self.exposure_spin.value())
        s.set("trigger_hz", self.trigger_spin.value())
        s.set("roi_start", self.roi_start.value())
        s.set("roi_end", self.roi_end.value())
        s.set("roi_method", self.roi_method.currentText())
        s.set("n_block", self.n_block.value())
        s.set("f_expected", self.expected.value())
        s.set("f_search_hw", self.search.value())
        s.set("window", self.window_combo.currentText())

    def _restore_settings(self) -> None:
        s = PanelSettings("DemodPanel", _SETTINGS_KEY_PREFIX)
        self.exposure_spin.setValue(s.get_float("exposure_s", 0.001))
        self.trigger_spin.setValue(s.get_float("trigger_hz", 500.0))
        self.roi_start.setValue(s.get_int("roi_start", 480))
        self.roi_end.setValue(s.get_int("roi_end", 560))
        idx = self.roi_method.findText(s.get_str("roi_method", "sum"))
        if idx >= 0:
            self.roi_method.setCurrentIndex(idx)
        self.n_block.setValue(s.get_int("n_block", 512))
        self.expected.setValue(s.get_float("f_expected", 37.0))
        self.search.setValue(s.get_float("f_search_hw", 5.0))
        idx = self.window_combo.findText(s.get_str("window", "hann"))
        if idx >= 0:
            self.window_combo.setCurrentIndex(idx)
        self._update_resolution()
        self._update_roi_region()


def _lbl(text: str) -> QLabel:
    return QLabel(text)
