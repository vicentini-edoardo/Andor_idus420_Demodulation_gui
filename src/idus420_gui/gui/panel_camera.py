"""Camera Settings panel."""

from __future__ import annotations

import json

import numpy as np
from PyQt6.QtCore import QSettings, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
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
    READ_MODE_LABELS,
    SHUTTER_MODE_LABELS,
    CameraBackend,
    CameraConfig,
    CameraError,
    CropConfig,
    ReadMode,
    SingleTrackConfig,
    TempStatus,
)
from idus420_gui.camera.mock import MockBackend
from idus420_gui.gui.widgets import StatusLed
from idus420_gui.spectrograph import (
    AndorShamrockBackend,
    MockSpectroBackend,
    SpectroBackend,
    SpectroError,
)

_SETTINGS_KEY_PREFIX = "camera_panel"


class CameraPanel(QWidget):
    """Connects/configures the active camera backend."""

    backend_changed = pyqtSignal(object)
    connection_changed = pyqtSignal(bool)
    temperature_changed = pyqtSignal(float, object)
    log_message = pyqtSignal(str)
    exposure_changed = pyqtSignal(float)
    frame_geometry_changed = pyqtSignal(int)
    # Emits calibrated wavelength axis (np.ndarray) when available, or None to
    # revert downstream plots to pixel index.
    wavelength_axis_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: CameraBackend | None = None
        self.spectro_backend: SpectroBackend | None = None
        self._pending_changes = False
        self._last_calibration: np.ndarray | None = None
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        # Top-level layout: camera panels row on top, spectrograph below
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

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

        top_row.addLayout(left_col)

        # --- Static Configuration group (right column) ---
        config_box = QGroupBox("Static Configuration")
        config_grid = QGridLayout(config_box)
        config_grid.setSpacing(6)
        config_grid.setColumnStretch(1, 1)
        config_grid.setColumnStretch(3, 1)

        self.read_mode_combo = QComboBox()
        self.read_mode_combo.addItems(list(READ_MODE_LABELS))

        self.st_center_spin = QSpinBox()
        self.st_center_spin.setRange(1, 9999)
        self.st_center_spin.setValue(128)
        self.st_center_spin.setSuffix(" px")

        self.st_height_spin = QSpinBox()
        self.st_height_spin.setRange(1, 9999)
        self.st_height_spin.setValue(10)
        self.st_height_spin.setSuffix(" px")

        self.hbin_spin = QSpinBox()
        self.hbin_spin.setRange(1, 1024)
        self.hbin_spin.setValue(1)
        self.hbin_spin.setSuffix(" px")

        self._st_center_label = QLabel("Center row")
        self._st_height_label = QLabel("Track height")
        self._hbin_label = QLabel("H bin")

        self.crop_check = QCheckBox("Enable Isolated Crop Mode")
        self.crop_check.setToolTip(
            "Reduces the CCD area for higher throughput via SetIsolatedCropMode.\n"
            "On the iDus only FVB read mode is supported."
        )

        self.crop_height_spin = QSpinBox()
        self.crop_height_spin.setRange(1, 9999)
        self.crop_height_spin.setValue(50)
        self.crop_height_spin.setSuffix(" px")

        self.crop_width_spin = QSpinBox()
        self.crop_width_spin.setRange(1, 9999)
        self.crop_width_spin.setValue(1024)
        self.crop_width_spin.setSuffix(" px")

        self.crop_vbin_spin = QSpinBox()
        self.crop_vbin_spin.setRange(1, 9999)
        self.crop_vbin_spin.setValue(1)
        self.crop_vbin_spin.setSuffix(" px")

        self._crop_height_label = QLabel("Crop height")
        self._crop_width_label = QLabel("Crop width")
        self._crop_vbin_label = QLabel("Crop V bin")

        self.hs_combo = QComboBox()
        self.vs_combo = QComboBox()
        self.preamp_combo = QComboBox()

        self.shutter_combo = QComboBox()
        self.shutter_combo.addItems(list(SHUTTER_MODE_LABELS))

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

        # Row 3: H bin | Exposure
        config_grid.addWidget(self._hbin_label, 3, 0)
        config_grid.addWidget(self.hbin_spin, 3, 1)
        config_grid.addWidget(QLabel("Exposure (s)"), 3, 2)
        config_grid.addWidget(self.exposure_spin, 3, 3)

        # Row 4: Shutter
        config_grid.addWidget(QLabel("Shutter"), 4, 0)
        config_grid.addWidget(self.shutter_combo, 4, 1)

        # Row 5: Crop mode enable (spans all cols)
        config_grid.addWidget(self.crop_check, 5, 0, 1, 4)

        # Row 6: Crop height | Crop width
        config_grid.addWidget(self._crop_height_label, 6, 0)
        config_grid.addWidget(self.crop_height_spin, 6, 1)
        config_grid.addWidget(self._crop_width_label, 6, 2)
        config_grid.addWidget(self.crop_width_spin, 6, 3)

        # Row 7: Crop vbin
        config_grid.addWidget(self._crop_vbin_label, 7, 0)
        config_grid.addWidget(self.crop_vbin_spin, 7, 1)

        # Row 8: Apply button (spans all cols) + actual timings
        config_grid.addWidget(self.apply_button, 8, 0, 1, 2)
        config_grid.addWidget(self.actual_label, 8, 2, 1, 2)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        right_col.addWidget(config_box)
        right_col.addStretch(1)

        top_row.addLayout(right_col, stretch=1)
        outer.addLayout(top_row)

        # --- Spectrograph (below camera panels) ---
        spectro_box = QGroupBox("Spectrograph")
        sg = QGridLayout(spectro_box)
        sg.setSpacing(6)
        sg.setColumnStretch(1, 1)

        self.spectro_backend_combo = QComboBox()
        self.spectro_backend_combo.addItems(["Mock", "Andor Shamrock"])
        self.spectro_backend_combo.setCurrentText("Andor Shamrock")
        self.spectro_connect_button = QPushButton("Connect")
        self.spectro_disconnect_button = QPushButton("Disconnect")
        self.spectro_disconnect_button.setEnabled(False)
        self.spectro_info_label = QLabel("Not connected")
        self.spectro_info_label.setWordWrap(True)

        self.spectro_grating_combo = QComboBox()
        self.spectro_grating_combo.setEnabled(False)

        self.spectro_wl_spin = QDoubleSpinBox()
        self.spectro_wl_spin.setRange(0.0, 9999.0)
        self.spectro_wl_spin.setDecimals(1)
        self.spectro_wl_spin.setValue(600.0)
        self.spectro_wl_spin.setSuffix(" nm")
        self.spectro_wl_spin.setEnabled(False)
        self.spectro_set_wl_button = QPushButton("Set WL")
        self.spectro_set_wl_button.setEnabled(False)

        self.spectro_range_label = QLabel("--")

        self.spectro_pixel_width_spin = QDoubleSpinBox()
        self.spectro_pixel_width_spin.setRange(0.1, 999.0)
        self.spectro_pixel_width_spin.setDecimals(1)
        self.spectro_pixel_width_spin.setValue(26.0)
        self.spectro_pixel_width_spin.setSuffix(" µm")
        self.spectro_pixel_width_spin.setToolTip(
            "Physical pixel pitch of the detector (iDus 420 = 26 µm)."
        )

        self.spectro_calibrate_button = QPushButton("Get Calibration")
        self.spectro_calibrate_button.setEnabled(False)
        self.spectro_cal_label = QLabel("--")
        self.spectro_cal_label.setWordWrap(True)

        row = 0
        sg.addWidget(QLabel("Backend"), row, 0)
        sg.addWidget(self.spectro_backend_combo, row, 1)
        row += 1
        conn_btns = QHBoxLayout()
        conn_btns.addWidget(self.spectro_connect_button)
        conn_btns.addWidget(self.spectro_disconnect_button)
        sg.addLayout(conn_btns, row, 0, 1, 2)
        row += 1
        sg.addWidget(QLabel("Status"), row, 0)
        sg.addWidget(self.spectro_info_label, row, 1)
        row += 1
        sg.addWidget(QLabel("Grating"), row, 0)
        sg.addWidget(self.spectro_grating_combo, row, 1)
        row += 1
        sg.addWidget(QLabel("Centre WL"), row, 0)
        wl_row = QHBoxLayout()
        wl_row.addWidget(self.spectro_wl_spin)
        wl_row.addWidget(self.spectro_set_wl_button)
        sg.addLayout(wl_row, row, 1)
        row += 1
        sg.addWidget(QLabel("Range"), row, 0)
        sg.addWidget(self.spectro_range_label, row, 1)
        row += 1
        sg.addWidget(QLabel("Pixel width"), row, 0)
        sg.addWidget(self.spectro_pixel_width_spin, row, 1)
        row += 1
        sg.addWidget(self.spectro_calibrate_button, row, 0, 1, 2)
        row += 1
        sg.addWidget(QLabel("Calibration"), row, 0)
        sg.addWidget(self.spectro_cal_label, row, 1)

        outer.addWidget(spectro_box)
        outer.addStretch(1)

        # --- Signals ---
        self.connect_button.clicked.connect(self._connect_backend)
        self.disconnect_button.clicked.connect(self._disconnect_backend)
        self.cooler_on_button.clicked.connect(self._cooler_on)
        self.cooler_off_button.clicked.connect(self._cooler_off)
        self.apply_button.clicked.connect(self._apply_config)
        self.exposure_spin.valueChanged.connect(self.exposure_changed.emit)
        self.read_mode_combo.currentIndexChanged.connect(self._on_read_mode_changed)
        self.backend_combo.currentIndexChanged.connect(self._update_hbin_limits)
        self.crop_check.stateChanged.connect(self._on_crop_changed)

        for widget in [
            self.hs_combo, self.vs_combo, self.preamp_combo,
            self.shutter_combo, self.read_mode_combo,
        ]:
            widget.currentIndexChanged.connect(self._mark_pending)
        for spin in [
            self.exposure_spin, self.st_center_spin, self.st_height_spin, self.hbin_spin,
            self.crop_height_spin, self.crop_width_spin, self.crop_vbin_spin,
        ]:
            spin.valueChanged.connect(self._mark_pending)

        self._on_read_mode_changed()
        self._on_crop_changed()
        self._update_hbin_limits()
        self._set_config_enabled(False)

        self.spectro_connect_button.clicked.connect(self._spectro_connect)
        self.spectro_disconnect_button.clicked.connect(self._spectro_disconnect)
        self.spectro_grating_combo.currentIndexChanged.connect(self._spectro_grating_changed)
        self.spectro_set_wl_button.clicked.connect(self._spectro_set_wavelength)
        self.spectro_calibrate_button.clicked.connect(self._spectro_get_calibration)

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
        self._hbin_label.setText("ST H bin" if is_st else "FVB H bin")

    def _on_crop_changed(self, _state: int = 0) -> None:
        is_crop = self.crop_check.isChecked()
        for w in [
            self._crop_height_label, self.crop_height_spin,
            self._crop_width_label, self.crop_width_spin,
            self._crop_vbin_label, self.crop_vbin_spin,
        ]:
            w.setVisible(is_crop)
        self._update_hbin_limits()

    def _update_hbin_limits(self) -> None:
        crop_active = self.crop_check.isChecked()
        if self.backend_combo.currentText() == "Andor iDus" and not crop_active:
            self.hbin_spin.setRange(1, 1)
            self.hbin_spin.setToolTip(
                "The iDus backend only supports horizontal bin = 1 in FVB and Single-Track."
            )
            self.hbin_spin.setValue(1)
            return
        current_max = self.hbin_spin.maximum()
        if current_max <= 1:
            self.hbin_spin.setRange(1, 1024)
        self.hbin_spin.setToolTip(
            "Horizontal binning within the crop area." if crop_active else ""
        )

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
            # Push the restored/default settings to the hardware immediately so
            # the camera does not run on SDK power-on defaults (e.g. a closed
            # shutter) until the user happens to click Apply.
            self._apply_config()
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
        xpix, ypix = self.backend.detector_size()
        self.st_center_spin.setRange(1, ypix)
        self.st_center_spin.setValue(ypix // 2)
        self.st_height_spin.setRange(1, ypix)
        self.crop_height_spin.setRange(1, ypix)
        self.crop_width_spin.setRange(1, xpix)
        self.crop_width_spin.setValue(xpix)
        self.crop_vbin_spin.setRange(1, ypix)
        if isinstance(self.backend, AndorIDusBackend) and not self.crop_check.isChecked():
            self.hbin_spin.setRange(1, 1)
            self.hbin_spin.setValue(1)
            self.hbin_spin.setToolTip(
                "The iDus backend only supports horizontal bin = 1 in FVB and Single-Track."
            )
        else:
            self.hbin_spin.setRange(1, xpix)
            self.hbin_spin.setToolTip("")
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

    # ------------------------------------------------------------------
    # Spectrograph
    # ------------------------------------------------------------------

    def _spectro_connect(self) -> None:
        try:
            spc: SpectroBackend = (
                AndorShamrockBackend()
                if self.spectro_backend_combo.currentText() == "Andor Shamrock"
                else MockSpectroBackend()
            )
            spc.connect()
            self.spectro_backend = spc
            gratings = spc.list_gratings()
            self.spectro_grating_combo.blockSignals(True)
            self.spectro_grating_combo.clear()
            for g in gratings:
                label = f"{g.index}: {g.lines_per_mm:.0f} l/mm  blaze {g.blaze}"
                self.spectro_grating_combo.addItem(label, userData=g.index)
            current = spc.get_grating()
            self.spectro_grating_combo.setCurrentIndex(current - 1)
            self.spectro_grating_combo.blockSignals(False)
            wl = spc.get_wavelength()
            self.spectro_wl_spin.setValue(wl)
            self._spectro_update_range()
            self.spectro_info_label.setText("Connected")
            self._spectro_set_enabled(True)
            self.spectro_connect_button.setEnabled(False)
            self.spectro_disconnect_button.setEnabled(True)
            self.spectro_backend_combo.setEnabled(False)
            self.log_message.emit("Spectrograph connected.")
        except (SpectroError, Exception) as exc:  # noqa: BLE001
            self.log_message.emit(f"Spectrograph connection failed: {exc}")
            self.spectro_backend = None

    def _spectro_disconnect(self) -> None:
        if self.spectro_backend:
            try:
                self.spectro_backend.disconnect()
            except SpectroError as exc:
                self.log_message.emit(str(exc))
        self.spectro_backend = None
        self.spectro_info_label.setText("Not connected")
        self.spectro_range_label.setText("--")
        if self._last_calibration is not None:
            n = len(self._last_calibration)
            lo = float(np.nanmin(self._last_calibration))
            hi = float(np.nanmax(self._last_calibration))
            self.spectro_cal_label.setText(f"{n} px  |  {lo:.1f} – {hi:.1f} nm  (offline)")
        else:
            self.spectro_cal_label.setText("--")
        self.spectro_grating_combo.clear()
        self._spectro_set_enabled(False)
        self.spectro_connect_button.setEnabled(True)
        self.spectro_disconnect_button.setEnabled(False)
        self.spectro_backend_combo.setEnabled(True)
        self.log_message.emit("Spectrograph disconnected.")

    def _spectro_set_enabled(self, enabled: bool) -> None:
        for w in [
            self.spectro_grating_combo,
            self.spectro_wl_spin,
            self.spectro_set_wl_button,
            self.spectro_calibrate_button,
        ]:
            w.setEnabled(enabled)

    def _spectro_grating_changed(self, _index: int) -> None:
        if not self.spectro_backend:
            return
        idx = self.spectro_grating_combo.currentData()
        if idx is None:
            return
        try:
            self.spectro_backend.set_grating(int(idx))
            wl = self.spectro_backend.get_wavelength()
            self.spectro_wl_spin.setValue(wl)
            self._spectro_update_range()
            self.log_message.emit(f"Grating set to slot {idx}.")
            # Re-calibrate automatically if a calibration was already active.
            if self._last_calibration is not None:
                self._spectro_get_calibration()
        except SpectroError as exc:
            self.log_message.emit(str(exc))

    def _spectro_set_wavelength(self) -> None:
        if not self.spectro_backend:
            return
        nm = self.spectro_wl_spin.value()
        try:
            self.spectro_backend.set_wavelength(nm)
            self._spectro_update_range()
            self.log_message.emit(f"Centre wavelength set to {nm:.1f} nm.")
            if self._last_calibration is not None:
                self._spectro_get_calibration()
        except SpectroError as exc:
            self.log_message.emit(str(exc))

    def _spectro_update_range(self) -> None:
        if not self.spectro_backend:
            return
        try:
            lo, hi = self.spectro_backend.get_wavelength_limits()
            self.spectro_range_label.setText(f"{lo:.0f} – {hi:.0f} nm")
            self.spectro_wl_spin.setRange(lo, hi)
        except SpectroError as exc:
            self.log_message.emit(str(exc))

    def _spectro_get_calibration(self) -> None:
        if not self.spectro_backend:
            return
        hbin = self.hbin_spin.value()
        if self.backend and self.backend.is_connected():
            n_pixels = self.backend.frame_width()
        else:
            n_pixels = 1024 // hbin
        pixel_um = self.spectro_pixel_width_spin.value() * hbin
        try:
            axis = self.spectro_backend.get_calibration(n_pixels, pixel_um)
            lo = float(np.nanmin(axis))
            hi = float(np.nanmax(axis))
            self.spectro_cal_label.setText(f"{n_pixels} px  |  {lo:.1f} – {hi:.1f} nm")
            self._last_calibration = axis
            self._save_calibration()
            self.wavelength_axis_changed.emit(axis)
            self.log_message.emit(
                f"Wavelength calibration: {n_pixels} px, {lo:.1f}–{hi:.1f} nm."
            )
        except SpectroError as exc:
            self.log_message.emit(str(exc))

    def _apply_config(self) -> None:
        if not self.backend:
            return
        try:
            self.backend.configure(self.current_config())
            timings = self.backend.query_timings()
            self.frame_geometry_changed.emit(self.backend.frame_width())
            t = timings
            period = max(t.exposure_s, t.accumulate_s, t.kinetic_s)
            f_max = (1.0 / period) if period and period > 0 else float("nan")
            self.actual_label.setText(
                "Actual timings: "
                f"exp {t.exposure_s:.6g} s, acc {t.accumulate_s:.6g} s, "
                f"kin {t.kinetic_s:.6g} s | "
                f"max ext trigger {f_max:.4g} Hz"
            )
            self._clear_pending()
            self._save_settings()
            if (
                self.spectro_backend
                and self.spectro_backend.is_connected()
                and self._last_calibration is not None
            ):
                self._spectro_get_calibration()
            self.log_message.emit("Camera settings applied.")
        except CameraError as exc:
            self.log_message.emit(str(exc))

    def current_config(self) -> CameraConfig:
        shutter = SHUTTER_MODE_LABELS[self.shutter_combo.currentText()]
        read_mode = READ_MODE_LABELS.get(
            self.read_mode_combo.currentText(), ReadMode.FVB
        )
        return CameraConfig(
            hs_speed_index=self.hs_combo.currentIndex(),
            vs_speed_index=self.vs_combo.currentIndex(),
            preamp_gain_index=self.preamp_combo.currentIndex(),
            shutter_mode=shutter,
            exposure_s=self.exposure_spin.value(),
            read_mode=read_mode,
            fvb_horizontal_bin=self.hbin_spin.value(),
            single_track=SingleTrackConfig(
                center_row=self.st_center_spin.value(),
                height=self.st_height_spin.value(),
                horizontal_bin=self.hbin_spin.value(),
            ),
            crop=CropConfig(
                active=self.crop_check.isChecked(),
                crop_height=self.crop_height_spin.value(),
                crop_width=self.crop_width_spin.value(),
                vbin=self.crop_vbin_spin.value(),
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
            self.hbin_spin,
            self.crop_check,
            self.crop_height_spin,
            self.crop_width_spin,
            self.crop_vbin_spin,
        ]:
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_calibration(self) -> None:
        if self._last_calibration is None:
            return
        s = QSettings("idus420_gui", "CameraPanel")
        s.setValue(
            f"{_SETTINGS_KEY_PREFIX}/calibration",
            json.dumps(self._last_calibration.tolist()),
        )

    def _restore_calibration(self) -> None:
        s = QSettings("idus420_gui", "CameraPanel")
        val = s.value(f"{_SETTINGS_KEY_PREFIX}/calibration")
        if val is None:
            return
        try:
            axis = np.asarray(json.loads(str(val)), dtype=np.float64)
        except Exception:
            return
        self._last_calibration = axis
        n = len(axis)
        lo = float(np.nanmin(axis))
        hi = float(np.nanmax(axis))
        self.spectro_cal_label.setText(f"{n} px  |  {lo:.1f} – {hi:.1f} nm  (restored)")
        self.wavelength_axis_changed.emit(axis)

    def _save_settings(self) -> None:
        s = QSettings("idus420_gui", "CameraPanel")
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/exposure_s", self.exposure_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/shutter", self.shutter_combo.currentText())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/backend", self.backend_combo.currentText())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/target_temp", self.temp_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/read_mode", self.read_mode_combo.currentText())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/st_center", self.st_center_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/st_height", self.st_height_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/horizontal_bin", self.hbin_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/crop_active", self.crop_check.isChecked())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/crop_height", self.crop_height_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/crop_width", self.crop_width_spin.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/crop_vbin", self.crop_vbin_spin.value())

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
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/horizontal_bin")) is not None:
            self.hbin_spin.setValue(int(val))
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/crop_active")) is not None:
            self.crop_check.setChecked(val is True or val == "true")
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/crop_height")) is not None:
            self.crop_height_spin.setValue(int(val))
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/crop_width")) is not None:
            self.crop_width_spin.setValue(int(val))
        if (val := s.value(f"{_SETTINGS_KEY_PREFIX}/crop_vbin")) is not None:
            self.crop_vbin_spin.setValue(int(val))
        self._clear_pending()
        QTimer.singleShot(0, self._restore_calibration)
