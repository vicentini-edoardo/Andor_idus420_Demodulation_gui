"""Large live-spectrum panel."""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
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
from idus420_gui.workers.acquisition import LiveSpectrumWorker

_SETTINGS_KEY_PREFIX = "live_panel"


class LiveSpectrumPanel(QWidget):
    """Dedicated big-screen live spectrum view."""

    log_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: CameraBackend | None = None
        self.worker: LiveSpectrumWorker | None = None
        self._detector_width = 100_000
        self._syncing_roi = False
        self._history_t: deque[float] = deque()
        self._history_roi: deque[float] = deque()
        self._last_roi_sum: float | None = None
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

        self.readout = ReadoutLabel("Spectrum: -- | ROI sum: --")
        outer.addWidget(self.readout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        controls_box = QGroupBox("Live Spectrum")
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

        self.burst_spin = QSpinBox()
        self.burst_spin.setRange(4, 4096)
        self.burst_spin.setValue(64)

        self.roi_start = QSpinBox()
        self.roi_start.setRange(0, 100_000)
        self.roi_start.setValue(480)

        self.roi_end = QSpinBox()
        self.roi_end.setRange(0, 100_000)
        self.roi_end.setValue(560)

        self.history_points_spin = QSpinBox()
        self.history_points_spin.setRange(10, 100_000)
        self.history_points_spin.setValue(1000)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.autorange_button = QPushButton("Auto Range")

        row = 0
        grid.addWidget(_lbl("Exposure time"), row, 0)
        grid.addWidget(self.exposure_spin, row, 1)
        row += 1
        grid.addWidget(_lbl("Trigger freq."), row, 0)
        grid.addWidget(self.trigger_spin, row, 1)
        row += 1
        grid.addWidget(_lbl("Frames / burst"), row, 0)
        grid.addWidget(self.burst_spin, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI start (px)"), row, 0)
        grid.addWidget(self.roi_start, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI end (px)"), row, 0)
        grid.addWidget(self.roi_end, row, 1)
        row += 1
        grid.addWidget(_lbl("History max points"), row, 0)
        grid.addWidget(self.history_points_spin, row, 1)
        row += 1

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        grid.addLayout(buttons, row, 0, 1, 2)
        row += 1
        grid.addWidget(self.autorange_button, row, 0, 1, 2)

        controls_layout.addWidget(controls_box)
        controls_layout.addStretch(1)
        controls_container.setMinimumWidth(240)
        controls_container.setMaximumWidth(380)

        splitter.addWidget(controls_container)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(theme.BG)

        self.spectrum_plot = self.plot_widget.addPlot(row=0, col=0, title="Live spectrum")
        self.history_plot = self.plot_widget.addPlot(
            row=1,
            col=0,
            title="ROI summed counts vs time",
        )
        self.plot_widget.ci.layout.setRowStretchFactor(0, 3)
        self.plot_widget.ci.layout.setRowStretchFactor(1, 2)

        for pi in (self.spectrum_plot, self.history_plot):
            theme.style_plot_item(pi)

        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen(theme.CURVE_YELLOW, width=2)
        )
        self.history_curve = self.history_plot.plot(
            pen=pg.mkPen(theme.CURVE_CYAN, width=1.8)
        )
        self.history_plot.setLabel("bottom", "Time (s)")
        self.history_plot.setLabel("left", "ROI sum (counts)")

        roi_color = QColor(theme.ACCENT)
        roi_color.setAlpha(40)
        self.roi_region = pg.LinearRegionItem(
            values=(self.roi_start.value(), self.roi_end.value()),
            movable=True,
            brush=pg.mkBrush(roi_color),
            pen=pg.mkPen(theme.ACCENT, width=1),
        )
        self.spectrum_plot.addItem(self.roi_region)

        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 900])

        outer.addWidget(splitter, stretch=1)

        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.autorange_button.clicked.connect(self._auto_range)
        self.roi_start.valueChanged.connect(self._update_roi_region)
        self.roi_end.valueChanged.connect(self._update_roi_region)
        self.roi_region.sigRegionChanged.connect(self._update_roi_spinboxes_from_region)
        self.history_points_spin.valueChanged.connect(self._trim_history)

    def start(self) -> None:
        if not self.backend:
            self.log_message.emit("No camera backend is connected.")
            return
        if not self._validate_roi():
            return
        # Ensure any previous worker has fully stopped before starting a new one.
        if self.worker is not None and self.worker.isRunning():
            stop_worker(self.worker)
        self._history_t.clear()
        self._history_roi.clear()
        self._last_roi_sum = None
        self.history_curve.clear()
        self.history_plot.enableAutoRange()
        self.history_plot.autoRange()
        self.worker = LiveSpectrumWorker(
            self.backend,
            self.exposure_spin.value(),
            self.trigger_spin.value(),
            self.roi_start.value(),
            self.roi_end.value(),
            self.burst_spin.value(),
        )
        self.worker.frame_acquired.connect(self._update_frame)
        self.worker.roi_sample.connect(self._update_roi_history)
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

    def _update_frame(self, frame: object) -> None:
        arr = np.asarray(frame, dtype=np.uint16)
        self.spectrum_curve.setData(arr)
        y_min = int(arr.min()) if arr.size else 0
        y_max = int(arr.max()) if arr.size else 0
        self._set_readout(arr.size, y_min, y_max)

    def _update_roi_history(self, elapsed_s: float, roi_sum: float) -> None:
        self._last_roi_sum = roi_sum
        self._history_t.append(float(elapsed_s))
        self._history_roi.append(float(roi_sum))
        self._trim_history()
        spectrum_data = self.spectrum_curve.getData()[1]
        if spectrum_data is None:
            self._set_readout(0, 0, 0)
        else:
            arr = np.asarray(spectrum_data)
            self._set_readout(arr.size, int(arr.min()), int(arr.max()))

    def _trim_history(self) -> None:
        limit = self.history_points_spin.value()
        while len(self._history_t) > limit:
            self._history_t.popleft()
        while len(self._history_roi) > limit:
            self._history_roi.popleft()
        if self._history_t and self._history_roi:
            self._refresh_history_curve()

    def _refresh_history_curve(self) -> None:
        t = np.asarray(self._history_t, dtype=np.float64)
        roi = np.asarray(self._history_roi, dtype=np.float64)
        self.history_curve.setData(t, roi)
        self._update_history_view(t, roi)

    def _update_history_view(self, t: np.ndarray, roi: np.ndarray) -> None:
        if t.size == 0 or roi.size == 0:
            return
        if t.size == 1:
            x0 = max(0.0, float(t[0]) - 0.5)
            x1 = float(t[0]) + 0.5
        else:
            x0 = float(t[0])
            x1 = float(t[-1])
            if x1 <= x0:
                x1 = x0 + 1e-6
        y0 = float(np.min(roi))
        y1 = float(np.max(roi))
        if y1 <= y0:
            pad = max(abs(y0) * 0.05, 1.0)
            y0 -= pad
            y1 += pad
        else:
            pad = (y1 - y0) * 0.08
            y0 -= pad
            y1 += pad
        self.history_plot.setXRange(x0, x1, padding=0.0)
        self.history_plot.setYRange(y0, y1, padding=0.0)

    def _set_readout(self, n_px: int, y_min: int, y_max: int) -> None:
        roi_text = f"{self._last_roi_sum:.0f}" if self._last_roi_sum is not None else "--"
        self.readout.setText(
            f"Spectrum: {n_px} px | min {y_min} | max {y_max} | ROI sum: {roi_text}"
        )

    def _validate_roi(self) -> bool:
        error = roi_error_message(
            self.roi_start.value(), self.roi_end.value(), self._detector_width
        )
        if error:
            self.log_message.emit(error)
            return False
        return True

    def _update_roi_region(self) -> None:
        if self._syncing_roi:
            return
        self._syncing_roi = True
        self.roi_region.setRegion((self.roi_start.value(), self.roi_end.value()))
        self._syncing_roi = False

    def _update_roi_spinboxes_from_region(self) -> None:
        if self._syncing_roi:
            return
        lower, upper = self.roi_region.getRegion()
        lower_i = max(0, min(int(round(lower)), self._detector_width - 1))
        upper_i = max(0, min(int(round(upper)), self._detector_width - 1))
        self._syncing_roi = True
        self.roi_start.setValue(lower_i)
        self.roi_end.setValue(upper_i)
        self._syncing_roi = False
        self._update_roi_region()

    def _auto_range(self) -> None:
        self.spectrum_plot.enableAutoRange()
        self.spectrum_plot.autoRange()
        self.history_plot.enableAutoRange()
        self.history_plot.autoRange()

    def _set_running_ui(self, running: bool) -> None:
        for widget in [
            self.exposure_spin,
            self.trigger_spin,
            self.burst_spin,
            self.roi_start,
            self.roi_end,
            self.history_points_spin,
            self.start_button,
        ]:
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.autorange_button.setEnabled(True)
        self.roi_region.setMovable(not running)
        if not running:
            self._save_settings()
        self.running_changed.emit(running)

    def _save_settings(self) -> None:
        s = PanelSettings("LiveSpectrumPanel", _SETTINGS_KEY_PREFIX)
        s.set("exposure_s", self.exposure_spin.value())
        s.set("trigger_hz", self.trigger_spin.value())
        s.set("burst_frames", self.burst_spin.value())
        s.set("roi_start", self.roi_start.value())
        s.set("roi_end", self.roi_end.value())
        s.set("history_max_points", self.history_points_spin.value())

    def _restore_settings(self) -> None:
        s = PanelSettings("LiveSpectrumPanel", _SETTINGS_KEY_PREFIX)
        self.exposure_spin.setValue(s.get_float("exposure_s", 0.001))
        self.trigger_spin.setValue(s.get_float("trigger_hz", 500.0))
        self.burst_spin.setValue(s.get_int("burst_frames", 64))
        self.roi_start.setValue(s.get_int("roi_start", 480))
        self.roi_end.setValue(s.get_int("roi_end", 560))
        self.history_points_spin.setValue(s.get_int("history_max_points", 1000))
        self._update_roi_region()


def _lbl(text: str) -> QLabel:
    return QLabel(text)
