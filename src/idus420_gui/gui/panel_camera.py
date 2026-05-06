"""Camera Settings panel."""

from __future__ import annotations

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from idus420_gui.camera.andor import AndorIDusBackend
from idus420_gui.camera.base import (
    CameraBackend,
    CameraConfig,
    CameraError,
    ReadMode,
    ShutterMode,
    SingleTrackConfig,
    TempStatus,
)
from idus420_gui.camera.mock import MockBackend
from idus420_gui.gui.widgets import StatusLed

_SETTINGS_KEY_PREFIX = "camera_panel"


class CameraPanel(QWidget):
    """Connects/configures the active camera backend."""

    backend_changed = pyqtSignal(object)
    connection_changed = pyqtSignal(bool)
    temperature_changed = pyqtSignal(float, object)
    log_message = pyqtSignal(str)
    exposure_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: CameraBackend | None = None
        self._pending_changes = False
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        # Top-level layout: two columns side by side
        # Left column: Connection + Cooling stacked
        # Right column: Static Configuration (taller)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # --- Connection group ---
        connection_box = QGroupBox("Connection")
        conn_grid = QGridLayout(connection_box)
        conn_grid.setSpacing(6)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Mock", "Andor iDus"])
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        self.info_label = QLabel("Not connected")
        self.info_label.setWordWrap(True)

        conn_grid.addWidget(QLabel("Backend"), 0, 0)
        conn_grid.addWidget(self.backend_combo, 0, 1)
        conn_grid.addWidget(self.connect_button, 1, 0)
        conn_grid.addWidget(self.disconnect_button, 1, 1)
        conn_grid.addWidget(QLabel("Info"), 2, 0)
        conn_grid.addWidget(self.info_label, 2, 1)
        conn_grid.setColumnStretch(1, 1)

        left_col.addWidget(connection_box)

        # --- Cooling group ---
        cooling_box = QGroupBox("Cooling")
        cool_grid = QGridLayout(cooling_box)
        cool_grid.setSpacing(6)

        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(-95, 20)
        self.temp_spin.setValue(-60)
        self.temp_spin.setSuffix(" °C")
        self.cooler_on_button = QPushButton("Cooler ON")
        self.cooler_off_button = QPushButton("Cooler OFF")
        self.temp_label = QLabel("-- °C")
        self.temp_led = StatusLed()

        temp_row = QHBoxLayout()
        temp_row.addWidget(self.temp_label)
        temp_row.addWidget(self.temp_led)
        temp_row.addStretch(1)

        cool_grid.addWidget(QLabel("Target (°C)"), 0, 0)
        cool_grid.addWidget(self.temp_spin, 0, 1)
        cool_grid.addWidget(self.cooler_on_button, 1, 0)
        cool_grid.addWidget(self.cooler_off_button, 1, 1)
        cool_grid.addWidget(QLabel("Current"), 2, 0)
        cool_grid.addLayout(temp_row, 2, 1)
        cool_grid.setColumnStretch(1, 1)

        left_col.addWidget(cooling_box)
        left_col.addStretch(1)

        outer.addLayout(left_col)

        # --- Static Configuration group (right column) ---
        config_box = QGroupBox("Static Configuration")
        config_grid = QGridLayout(config_box)
        config_grid.setSpacing(6)
        config_grid.setColumnStretch(1, 1)
        config_grid.setColumnStretch(3, 1)

        self.read_mode_combo = QComboBox()
        self.read_mode_combo.addItems(["FVB", "Single-Track"])

        self.st_center_spin = QSpinBox()
        self.st_center_spin.setRange(1, 9999)
        self.st_center_spin.setValue(128)
        self.st_center_spin.setSuffix(" px")

        self.st_height_spin = QSpinBox()
        self.st_height_spin.setRange(1, 9999)
        self.st_height_spin.setValue(10)
        self.st_height_spin.setSuffix(" px")

        self._st_center_label = QLabel("Center row")
        self._st_height_label = QLabel("Track height")

        self.hs_combo = QComboBox()
        self.vs_combo = QComboBox()
        self.preamp_combo = QComboBox()

        self.shutter_combo = QComboBox()
        self.shutter_combo.addItems(["Permanently Open", "Auto", "Permanently Closed"])

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setDecimals(6)
        self.exposure_spin.setRange(0.000001, 1000.0)
        self.exposure_spin.setValue(0.001)
        self.exposure_spin.setSuffix(" s")

        self.apply_button = QPushButton("Apply")
        self.actual_label = QLabel("Actual timings: --")
        self.actual_label.setWordWrap(True)

        # Row 0: Read mode | HS speed
        config_grid.addWidget(QLabel("Read mode"), 0, 0)
        config_grid.addWidget(self.read_mode_combo, 0, 1)
        config_grid.addWidget(QLabel("HS speed (MHz)"), 0, 2)
        config_grid.addWidget(self.hs_combo, 0, 3)

        # Row 1: Center row | VS speed
        config_grid.addWidget(self._st_center_label, 1, 0)
        config_grid.addWidget(self.st_center_spin, 1, 1)
        config_grid.addWidget(QLabel("VS speed (µs)"), 1, 2)
        config_grid.addWidget(self.vs_combo, 1, 3)

        # Row 2: Track height | Pre-amp gain
        config_grid.addWidget(self._st_height_label, 2, 0)
        config_grid.addWidget(self.st_height_spin, 2, 1)
        config_grid.addWidget(QLabel("Pre-amp gain"), 2, 2)
        config_grid.addWidget(self.preamp_combo, 2, 3)

        # Row 3: Shutter | Exposure
        config_grid.addWidget(QLabel("Shutter"), 3, 0)
        config_grid.addWidget(self.shutter_combo, 3, 1)
        config_grid.addWidget(QLabel("Exposure (s)"), 3, 2)
        config_grid.addWidget(self.exposure_spin, 3, 3)

        # Row 4: Apply button (spans all cols) + actual timings
        config_grid.addWidget(self.apply_button, 4, 0, 1, 2)
        config_grid.addWidget(self.actual_label, 4, 2, 1, 2)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        right_col.addWidget(config_box)
        right_col.addStretch(1)

        outer.addLayout(right_col, stretch=1)

        # --- Signals ---
        self.connect_button.clicked.connect(self._connect_backend)
        self.disconnect_button.clicked.connect(self._disconnect_backend)
        self.cooler_on_button.clicked.connect(self._cooler_on)
        self.cooler_off_button.clicked.connect(self._cooler_off)
        self.apply_button.clicked.connect(self._apply_config)
        self.exposure_spin.valueChanged.connect(self.exposure_changed.emit)
        self.read_mode_combo.currentIndexChanged.connect(self._on_read_mode_changed)

        for widget in [
            self.hs_combo, self.vs_combo, self.preamp_combo,
            self.shutter_combo, self.read_mode_combo,
        ]:
            widget.currentIndexChanged.connect(self._mark_pending)
        for spin in [self.exposure_spin, self.st_center_spin, self.st_height_spin]:
            spin.valueChanged.connect(self._mark_pending)

        self._on_read_mode_changed()
        self._set_config_enabled(False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def poll_temperature(self) -> None:
        """Read temperature from backend and update labels/LED."""
        if not self.backend or not self.backend.is_connected():
            return
        try:
            temp, status = self.backend.get_temperature()
        except CameraError as exc:
            self.log_message.emit(str(exc))
            return
        self.temp_label.setText(f"{temp:.1f} °C ({status.value})")
        self.temp_led.set_state("green" if status is TempStatus.STABILIZED else "orange")
        self.temperature_changed.emit(temp, status)

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_read_mode_changed(self, _index: int = 0) -> None:
        is_st = self.read_mode_combo.currentText() == "Single-Track"
        self.st_center_spin.setVisible(is_st)
        self.st_height_spin.setVisible(is_st)
        self._st_center_label.setVisible(is_st)
        self._st_height_label.setVisible(is_st)

    def _mark_pending(self) -> None:
        if not self._pending_changes:
            self._pending_changes = True
            self.apply_button.setText("Apply ●")
        self.apply_button.setProperty("pending", True)
        self.apply_button.style().unpolish(self.apply_button)
        self.apply_button.style().polish(self.apply_button)

    def _clear_pending(self) -> None:
        self._pending_changes = False
        self.apply_button.setText("Apply")
        self.apply_button.setProperty("pending", False)
        self.apply_button.style().unpolish(self.apply_button)
        self.apply_button.style().polish(self.apply_button)

    def _connect_backend(self) -> None:
        try:
            self.backend = self._make_backend()
            self.backend.connect()
            self.backend_changed.emit(self.backend)
            self._populate_camera_values()
            serial = self.backend.serial_number()
            xpix, ypix = self.backend.detector_size()
            self.info_label.setText(
                f"Serial {serial}; detector {xpix} x {ypix}; SDK {self.backend.sdk_version()}"
            )
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.backend_combo.setEnabled(False)
            self._set_config_enabled(True)
            self.connection_changed.emit(True)
            self.log_message.emit("Camera connected.")
        except Exception as exc:  # noqa: BLE001 - slot-level user feedback.
            self.log_message.emit(f"Connection failed: {exc}")
            self.backend = None
            self.connection_changed.emit(False)

    def _disconnect_backend(self) -> None:
        if self.backend:
            try:
                self.backend.disconnect()
            except CameraError as exc:
                self.log_message.emit(str(exc))
        self.backend = None
        self.info_label.setText("Not connected")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.backend_combo.setEnabled(True)
        self._set_config_enabled(False)
        self.connection_changed.emit(False)
        self.log_message.emit("Camera disconnected.")

    def _make_backend(self) -> CameraBackend:
        if self.backend_combo.currentText() == "Andor iDus":
            return AndorIDusBackend()
        return MockBackend()

    def _populate_camera_values(self) -> None:
        if not self.backend:
            return
        t_min, t_max = self.backend.temperature_range()
        self.temp_spin.setRange(t_min, t_max)
        _, ypix = self.backend.detector_size()
        self.st_center_spin.setRange(1, ypix)
        self.st_center_spin.setValue(ypix // 2)
        self.st_height_spin.setRange(1, ypix)
        self.hs_combo.clear()
        self.hs_combo.addItems([f"{value:g}" for value in self.backend.list_hs_speeds()])
        self.vs_combo.clear()
        self.vs_combo.addItems([f"{value:g}" for value in self.backend.list_vs_speeds()])
        self.preamp_combo.clear()
        self.preamp_combo.addItems([f"{value:g}" for value in self.backend.list_preamp_gains()])
        self._clear_pending()

    def _cooler_on(self) -> None:
        if not self.backend:
            return
        try:
            self.backend.set_target_temperature(self.temp_spin.value())
            self.backend.cooler_on()
            self.poll_temperature()
            self.log_message.emit("Cooler enabled.")
        except CameraError as exc:
            self.log_message.emit(str(exc))

    def _cooler_off(self) -> None:
        if not self.backend:
            return
        try:
            self.backend.cooler_off()
            self.poll_temperature()
            self.log_message.emit("Cooler disabled.")
        except CameraError as exc:
            self.log_message.emit(str(exc))

    def _apply_config(self) -> None:
        if not self.backend:
            return
        try:
            self.backend.configure(self.current_config())
            timings = self.backend.query_timings()
            t = timings
            period = t.readout_s if t.readout_s and t.readout_s > 0 else t.kinetic_s
            f_max = (1.0 / period) if period and period > 0 else float("nan")
            self.actual_label.setText(
                "Actual timings: "
                f"exp {t.exposure_s:.6g} s, acc {t.accumulate_s:.6g} s, "
                f"kin {t.kinetic_s:.6g} s | "
                f"max ext trigger {f_max:.4g} Hz"
            )
            self._clear_pending()
            self._save_settings()
            self.log_message.emit("Camera settings applied.")
        except CameraError as exc:
            self.log_message.emit(str(exc))

    def current_config(self) -> CameraConfig:
        shutter = {
            "Permanently Open": ShutterMode.OPEN,
            "Auto": ShutterMode.AUTO,
            "Permanently Closed": ShutterMode.CLOSED,
        }[self.shutter_combo.currentText()]
        read_mode = (
            ReadMode.SINGLE_TRACK
            if self.read_mode_combo.currentText() == "Single-Track"
            else ReadMode.FVB
        )
        return CameraConfig(
            hs_speed_index=self.hs_combo.currentIndex(),
            vs_speed_index=self.vs_combo.currentIndex(),
            preamp_gain_index=self.preamp_combo.currentIndex(),
            shutter_mode=shutter,
            exposure_s=self.exposure_spin.value(),
            read_mode=read_mode,
            single_track=SingleTrackConfig(
                center_row=self.st_center_spin.value(),
                height=self.st_height_spin.value(),
            ),
        )

    def _set_config_enabled(self, enabled: bool) -> None:
        for widget in [
            self.temp_spin,
            self.cooler_on_button,
            self.cooler_off_button,
            self.hs_combo,
            self.vs_combo,
            self.preamp_combo,
            self.shutter_combo,
            self.exposure_spin,
            self.apply_button,
            self.read_mode_combo,
            self.st_center_spin,
            self.st_height_spin,
        ]:
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_settings(self) -> None:
        s = QSettings("idus420_gui", "CameraPanel")
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/exposure_s", self.exposure_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/shutter", self.shutter_combo.currentText())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/backend", self.backend_combo.currentText())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/target_temp", self.temp_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/read_mode", self.read_mode_combo.currentText())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/st_center", self.st_center_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/st_height", self.st_height_spin.value())

    def _restore_settings(self) -> None:
        s = QSettings("idus420_gui", "CameraPanel")
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/exposure_s")) is not None:
            self.exposure_spin.setValue(float(val))
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/shutter")) is not None:
            idx = self.shutter_combo.findText(str(val))
            if idx >= 0:
                self.shutter_combo.setCurrentIndex(idx)
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/backend")) is not None:
            idx = self.backend_combo.findText(str(val))
            if idx >= 0:
                self.backend_combo.setCurrentIndex(idx)
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/target_temp")) is not None:
            self.temp_spin.setValue(int(val))
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/read_mode")) is not None:
            idx = self.read_mode_combo.findText(str(val))
            if idx >= 0:
                self.read_mode_combo.setCurrentIndex(idx)
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/st_center")) is not None:
            self.st_center_spin.setValue(int(val))
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/st_height")) is not None:
            self.st_height_spin.setValue(int(val))
        self._clear_pending()
