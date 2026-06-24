"""Large live-spectrum panel."""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
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
        self._syncing_roi2 = False
        self._history_t: deque[float] = deque()
        self._history_roi: deque[float] = deque()
        self._history_roi2: deque[float] = deque()
        self._last_roi_sum: float | None = None
        self._last_roi_sum2: float | None = None
        self._build_ui()
        self._restore_settings()

    def set_backend(self, backend: CameraBackend | None) -> None:
        self.backend = backend
        self.set_frame_width(
            backend.frame_width() if backend is not None and backend.is_connected() else 100_000
        )

    def set_frame_width(self, width: int) -> None:
        self._detector_width = max(1, int(width))
        for spin in (self.roi_start, self.roi_end, self.roi_start2, self.roi_end2):
            spin.setMaximum(self._detector_width - 1)
        for spin in (self.roi_start, self.roi_end, self.roi_start2, self.roi_end2):
            spin.setValue(min(spin.value(), self._detector_width - 1))
        self._update_roi_region()
        self._update_roi_region2()

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

        self.roi_start2 = QSpinBox()
        self.roi_start2.setRange(0, 100_000)
        self.roi_start2.setValue(600)

        self.roi_end2 = QSpinBox()
        self.roi_end2.setRange(0, 100_000)
        self.roi_end2.setValue(680)

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
        grid.addWidget(_lbl("ROI1 start (px)"), row, 0)
        grid.addWidget(self.roi_start, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI1 end (px)"), row, 0)
        grid.addWidget(self.roi_end, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI2 start (px)"), row, 0)
        grid.addWidget(self.roi_start2, row, 1)
        row += 1
        grid.addWidget(_lbl("ROI2 end (px)"), row, 0)
        grid.addWidget(self.roi_end2, row, 1)
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
            theme._style_plot_item(pi)  # noqa: SLF001

        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen(theme.CURVE_YELLOW, width=2)
        )
        self.history_curve = self.history_plot.plot(
            pen=pg.mkPen(theme.CURVE_CYAN, width=1.8)
        )
        self.history_plot.setLabel("bottom", "Time (s)")
        self.history_plot.setLabel("left", "ROI1 sum (counts)", color=theme.CURVE_CYAN)

        # --- twin right-axis for ROI2 ---
        self.history_plot.showAxis("right")
        self.history_vb2 = pg.ViewBox()
        self.history_plot.scene().addItem(self.history_vb2)
        self.history_plot.getAxis("right").linkToView(self.history_vb2)
        self.history_vb2.setXLink(self.history_plot)
        self.history_curve2 = pg.PlotCurveItem(
            pen=pg.mkPen(theme.CURVE_MAGENTA, width=1.8)
        )
        self.history_vb2.addItem(self.history_curve2)
        self.history_plot.getAxis("right").setLabel(
            "ROI2 sum (counts)", color=theme.CURVE_MAGENTA
        )

        def _sync_history_vb2() -> None:
            self.history_vb2.setGeometry(self.history_plot.vb.sceneBoundingRect())
            self.history_vb2.linkedViewChanged(
                self.history_plot.vb, self.history_vb2.XAxis
            )

        self._sync_history_vb2 = _sync_history_vb2
        self.history_plot.vb.sigResized.connect(_sync_history_vb2)
        # initial sync so geometry is correct before first resize event
        _sync_history_vb2()

        # --- ROI1 overlay (cyan-tinted) ---
        roi_color = QColor(theme.ACCENT)
        roi_color.setAlpha(40)
        self.roi_region = pg.LinearRegionItem(
            values=(self.roi_start.value(), self.roi_end.value()),
            movable=True,
            brush=pg.mkBrush(roi_color),
            pen=pg.mkPen(theme.ACCENT, width=1),
        )
        self.spectrum_plot.addItem(self.roi_region)

        # --- ROI2 overlay (magenta-tinted) ---
        roi_color2 = QColor(theme.CURVE_MAGENTA)
        roi_color2.setAlpha(40)
        self.roi_region2 = pg.LinearRegionItem(
            values=(self.roi_start2.value(), self.roi_end2.value()),
            movable=True,
            brush=pg.mkBrush(roi_color2),
            pen=pg.mkPen(theme.CURVE_MAGENTA, width=1),
        )
        self.spectrum_plot.addItem(self.roi_region2)

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
        self.roi_start2.valueChanged.connect(self._update_roi_region2)
        self.roi_end2.valueChanged.connect(self._update_roi_region2)
        self.roi_region2.sigRegionChanged.connect(self._update_roi_spinboxes_from_region2)
        self.history_points_spin.valueChanged.connect(self._trim_history)

    def start(self) -> None:
        if not self.backend:
            self.log_message.emit("No camera backend is connected.")
            return
        if not self._validate_roi():
            return
        self._history_t.clear()
        self._history_roi.clear()
        self._history_roi2.clear()
        self._last_roi_sum = None
        self._last_roi_sum2 = None
        self.history_curve.clear()
        self.history_curve2.clear()
        self.history_plot.enableAutoRange()
        self.history_plot.autoRange()
        self.worker = LiveSpectrumWorker(
            self.backend,
            self.exposure_spin.value(),
            self.trigger_spin.value(),
            self.roi_start.value(),
            self.roi_end.value(),
            self.burst_spin.value(),
            self.roi_start2.value(),
            self.roi_end2.value(),
        )
        self.worker.frame_acquired.connect(self._update_frame)
        self.worker.roi_sample.connect(self._update_roi_history)
        self.worker.error.connect(self.log_message.emit)
        self.worker.worker_finished.connect(lambda: self._set_running_ui(False))
        self._set_running_ui(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
        self._set_running_ui(False)

    def _update_frame(self, frame: object) -> None:
        arr = np.asarray(frame, dtype=np.uint16)
        self.spectrum_curve.setData(arr)
        y_min = int(arr.min()) if arr.size else 0
        y_max = int(arr.max()) if arr.size else 0
        self._set_readout(arr.size, y_min, y_max)

    def _update_roi_history(self, elapsed_s: float, roi_sum: float, roi_sum2: float) -> None:
        self._last_roi_sum = roi_sum
        self._last_roi_sum2 = roi_sum2
        self._history_t.append(float(elapsed_s))
        self._history_roi.append(float(roi_sum))
        self._history_roi2.append(float(roi_sum2))
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
        while len(self._history_roi2) > limit:
            self._history_roi2.popleft()
        if self._history_t and self._history_roi:
            self._refresh_history_curve()

    def _refresh_history_curve(self) -> None:
        t = np.asarray(self._history_t, dtype=np.float64)
        roi = np.asarray(self._history_roi, dtype=np.float64)
        roi2 = np.asarray(self._history_roi2, dtype=np.float64)
        self.history_curve.setData(t, roi)
        self.history_curve2.setData(t, roi2)
        self._update_history_view(t, roi, roi2)

    def _update_history_view(
        self, t: np.ndarray, roi: np.ndarray, roi2: np.ndarray
    ) -> None:
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

        def _y_range(arr: np.ndarray) -> tuple[float, float]:
            y0 = float(np.min(arr))
            y1 = float(np.max(arr))
            if y1 <= y0:
                pad = max(abs(y0) * 0.05, 1.0)
                return y0 - pad, y1 + pad
            pad = (y1 - y0) * 0.08
            return y0 - pad, y1 + pad

        self.history_plot.setXRange(x0, x1, padding=0.0)
        y0, y1 = _y_range(roi)
        self.history_plot.setYRange(y0, y1, padding=0.0)
        if roi2.size > 0:
            y0_2, y1_2 = _y_range(roi2)
            self.history_vb2.setYRange(y0_2, y1_2, padding=0.0)
        # keep right ViewBox geometry in sync
        self._sync_history_vb2()

    def _set_readout(self, n_px: int, y_min: int, y_max: int) -> None:
        roi_text = f"{self._last_roi_sum:.0f}" if self._last_roi_sum is not None else "--"
        roi2_text = f"{self._last_roi_sum2:.0f}" if self._last_roi_sum2 is not None else "--"
        self.readout.setText(
            f"Spectrum: {n_px} px | min {y_min} | max {y_max}"
            f" | ROI1 sum: {roi_text} | ROI2 sum: {roi2_text}"
        )

    def _validate_roi(self) -> bool:
        start = self.roi_start.value()
        end = self.roi_end.value()
        if start > end:
            self.log_message.emit("ROI1 start must be ≤ ROI1 end.")
            return False
        if end >= self._detector_width:
            self.log_message.emit(
                f"ROI1 end {end} exceeds detector width {self._detector_width}."
            )
            return False
        start2 = self.roi_start2.value()
        end2 = self.roi_end2.value()
        if start2 > end2:
            self.log_message.emit("ROI2 start must be ≤ ROI2 end.")
            return False
        if end2 >= self._detector_width:
            self.log_message.emit(
                f"ROI2 end {end2} exceeds detector width {self._detector_width}."
            )
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

    def _update_roi_region2(self) -> None:
        if self._syncing_roi2:
            return
        self._syncing_roi2 = True
        self.roi_region2.setRegion((self.roi_start2.value(), self.roi_end2.value()))
        self._syncing_roi2 = False

    def _update_roi_spinboxes_from_region2(self) -> None:
        if self._syncing_roi2:
            return
        lower, upper = self.roi_region2.getRegion()
        lower_i = max(0, min(int(round(lower)), self._detector_width - 1))
        upper_i = max(0, min(int(round(upper)), self._detector_width - 1))
        self._syncing_roi2 = True
        self.roi_start2.setValue(lower_i)
        self.roi_end2.setValue(upper_i)
        self._syncing_roi2 = False
        self._update_roi_region2()

    def _auto_range(self) -> None:
        self.spectrum_plot.enableAutoRange()
        self.spectrum_plot.autoRange()
        self.history_plot.enableAutoRange()
        self.history_plot.autoRange()
        self.history_vb2.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.history_vb2.autoRange()

    def _set_running_ui(self, running: bool) -> None:
        for widget in [
            self.exposure_spin,
            self.trigger_spin,
            self.burst_spin,
            self.roi_start,
            self.roi_end,
            self.roi_start2,
            self.roi_end2,
            self.history_points_spin,
            self.start_button,
        ]:
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.autorange_button.setEnabled(True)
        self.roi_region.setMovable(not running)
        self.roi_region2.setMovable(not running)
        if not running:
            self._save_settings()
        self.running_changed.emit(running)

    def _save_settings(self) -> None:
        s = QSettings("idus420_gui", "LiveSpectrumPanel")
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/exposure_s", self.exposure_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/trigger_hz", self.trigger_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/burst_frames", self.burst_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/roi_start", self.roi_start.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/roi_end", self.roi_end.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/roi_start2", self.roi_start2.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/roi_end2", self.roi_end2.value())
        s.setValue(
            f"{_SETTINGS_KEY_PREFIX}/history_max_points",
            self.history_points_spin.value(),
        )

    def _restore_settings(self) -> None:
        s = QSettings("idus420_gui", "LiveSpectrumPanel")

        def fval(key: str, default: float) -> float:
            v = s.value(f"{_SETTINGS_KEY_PREFIX}/{key}")
            return float(v) if v is not None else default

        def ival(key: str, default: int) -> int:
            v = s.value(f"{_SETTINGS_KEY_PREFIX}/{key}")
            return int(v) if v is not None else default

        self.exposure_spin.setValue(fval("exposure_s", 0.001))
        self.trigger_spin.setValue(fval("trigger_hz", 500.0))
        self.burst_spin.setValue(ival("burst_frames", 64))
        self.roi_start.setValue(ival("roi_start", 480))
        self.roi_end.setValue(ival("roi_end", 560))
        self.roi_start2.setValue(ival("roi_start2", 600))
        self.roi_end2.setValue(ival("roi_end2", 680))
        self.history_points_spin.setValue(ival("history_max_points", 1000))
        self._update_roi_region()
        self._update_roi_region2()


def _lbl(text: str) -> QLabel:
    return QLabel(text)
