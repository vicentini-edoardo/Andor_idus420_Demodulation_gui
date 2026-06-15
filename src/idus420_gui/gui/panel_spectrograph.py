"""Spectrometer panel: grating selection and central wavelength control."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from idus420_gui.gui._helpers import PanelSettings
from idus420_gui.spectrograph.andor import AndorShamrockBackend
from idus420_gui.spectrograph.base import SpectrographBackend, SpectrographError
from idus420_gui.spectrograph.mock import MockSpectrograph

_SETTINGS_KEY_PREFIX = "spectrograph_panel"


class SpectrographPanel(QWidget):
    """Connects to the spectrograph and controls grating + central wavelength."""

    connection_changed = pyqtSignal(bool)
    calibration_changed = pyqtSignal(object)  # np.ndarray | None
    log_message = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: SpectrographBackend | None = None
        self._number_pixels = 1024
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
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
        self.backend_combo.addItems(["Mock", "Andor Shamrock"])
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        self.info_label = QLabel("Not connected")
        self.info_label.setWordWrap(True)

        conn_grid.addWidget(QLabel("Device"), 0, 0)
        conn_grid.addWidget(self.backend_combo, 0, 1)
        conn_grid.addWidget(self.connect_button, 1, 0)
        conn_grid.addWidget(self.disconnect_button, 1, 1)
        conn_grid.addWidget(QLabel("Info"), 2, 0)
        conn_grid.addWidget(self.info_label, 2, 1)
        conn_grid.setColumnStretch(1, 1)

        left_col.addWidget(connection_box)
        left_col.addStretch(1)
        outer.addLayout(left_col)

        # --- Grating / wavelength group ---
        grating_box = QGroupBox("Grating & Central Wavelength")
        grid = QGridLayout(grating_box)
        grid.setSpacing(6)
        grid.setColumnStretch(1, 1)

        self.grating_combo = QComboBox()

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setDecimals(3)
        self.wavelength_spin.setRange(0.0, 5000.0)
        self.wavelength_spin.setValue(500.0)
        self.wavelength_spin.setSuffix(" nm")

        self.pixel_width_spin = QDoubleSpinBox()
        self.pixel_width_spin.setDecimals(2)
        self.pixel_width_spin.setRange(0.1, 100.0)
        self.pixel_width_spin.setValue(26.0)
        self.pixel_width_spin.setSuffix(" µm")
        self.pixel_width_spin.setToolTip(
            "Detector pixel width, used to compute the wavelength calibration.\n"
            "The Andor iDus 420 has 26 µm pixels."
        )

        self.apply_button = QPushButton("Set Grating / Wavelength")
        self.status_label = QLabel("Grating: -- | Centre: --")
        self.status_label.setWordWrap(True)

        row = 0
        grid.addWidget(QLabel("Grating"), row, 0)
        grid.addWidget(self.grating_combo, row, 1)
        row += 1
        grid.addWidget(QLabel("Central wavelength"), row, 0)
        grid.addWidget(self.wavelength_spin, row, 1)
        row += 1
        grid.addWidget(QLabel("Pixel width"), row, 0)
        grid.addWidget(self.pixel_width_spin, row, 1)
        row += 1
        grid.addWidget(self.apply_button, row, 0, 1, 2)
        row += 1
        grid.addWidget(self.status_label, row, 0, 1, 2)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        right_col.addWidget(grating_box)
        right_col.addStretch(1)
        outer.addLayout(right_col, stretch=1)

        # --- Signals ---
        self.connect_button.clicked.connect(self._connect_backend)
        self.disconnect_button.clicked.connect(self._disconnect_backend)
        self.apply_button.clicked.connect(self._apply)
        self.grating_combo.currentIndexChanged.connect(self._on_grating_combo_changed)

        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_frame_width(self, width: int) -> None:
        """Track the camera's output pixel count for the wavelength calibration."""
        self._number_pixels = max(1, int(width))
        if self.backend and self.backend.is_connected():
            self._push_geometry_and_emit()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _make_backend(self) -> SpectrographBackend:
        if self.backend_combo.currentText() == "Andor Shamrock":
            return AndorShamrockBackend()
        return MockSpectrograph()

    def _connect_backend(self) -> None:
        try:
            self.backend = self._make_backend()
            self.backend.connect()
            self._populate_gratings()
            serial = self.backend.serial_number()
            self.info_label.setText(
                f"Serial {serial}; SDK {self.backend.sdk_version()}"
            )
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.backend_combo.setEnabled(False)
            self._set_controls_enabled(True)
            self.connection_changed.emit(True)
            self._push_geometry_and_emit()
            self.log_message.emit("Spectrograph connected.")
        except Exception as exc:  # noqa: BLE001 - slot-level user feedback.
            self.log_message.emit(f"Spectrograph connection failed: {exc}")
            self.backend = None
            self.connection_changed.emit(False)

    def _disconnect_backend(self) -> None:
        if self.backend:
            try:
                self.backend.disconnect()
            except SpectrographError as exc:
                self.log_message.emit(str(exc))
        self.backend = None
        self.info_label.setText("Not connected")
        self.status_label.setText("Grating: -- | Centre: --")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.backend_combo.setEnabled(True)
        self._set_controls_enabled(False)
        self.connection_changed.emit(False)
        self.calibration_changed.emit(None)
        self.log_message.emit("Spectrograph disconnected.")

    def _populate_gratings(self) -> None:
        if not self.backend:
            return
        self.grating_combo.blockSignals(True)
        self.grating_combo.clear()
        for grating in self.backend.list_gratings():
            self.grating_combo.addItem(grating.label(), grating.index)
        current = self.backend.current_grating()
        idx = self.grating_combo.findData(current)
        if idx >= 0:
            self.grating_combo.setCurrentIndex(idx)
        self.grating_combo.blockSignals(False)
        self._refresh_wavelength_limits()
        self.wavelength_spin.setValue(self.backend.get_wavelength())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _selected_grating(self) -> int:
        data = self.grating_combo.currentData()
        return int(data) if data is not None else 1

    def _on_grating_combo_changed(self, _index: int) -> None:
        # Update the allowed wavelength range to match the selected grating,
        # but do not move the turret until the user clicks Apply.
        self._refresh_wavelength_limits()

    def _refresh_wavelength_limits(self) -> None:
        if not self.backend:
            return
        try:
            lo, hi = self.backend.wavelength_limits(self._selected_grating())
        except SpectrographError as exc:
            self.log_message.emit(str(exc))
            return
        value = self.wavelength_spin.value()
        self.wavelength_spin.setRange(lo, hi)
        self.wavelength_spin.setValue(min(max(value, lo), hi))

    def _apply(self) -> None:
        if not self.backend:
            return
        try:
            self.backend.set_grating(self._selected_grating())
            self.backend.set_wavelength(self.wavelength_spin.value())
            self._push_geometry_and_emit()
            self._save_settings()
            self.log_message.emit("Spectrograph settings applied.")
        except SpectrographError as exc:
            self.log_message.emit(str(exc))

    def _push_geometry_and_emit(self) -> None:
        """Recompute the wavelength calibration and broadcast it."""
        if not self.backend or not self.backend.is_connected():
            return
        try:
            self.backend.set_pixel_geometry(
                self.pixel_width_spin.value(), self._number_pixels
            )
            calibration = self.backend.get_calibration(self._number_pixels)
            grating = self.backend.current_grating()
            wavelength = self.backend.get_wavelength()
        except SpectrographError as exc:
            self.log_message.emit(str(exc))
            return
        self.status_label.setText(
            f"Grating: {grating} | Centre: {wavelength:g} nm | "
            f"{calibration[0]:.2f}–{calibration[-1]:.2f} nm"
        )
        self.calibration_changed.emit(np.asarray(calibration, dtype=np.float64))

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in [
            self.grating_combo,
            self.wavelength_spin,
            self.pixel_width_spin,
            self.apply_button,
        ]:
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_settings(self) -> None:
        s = PanelSettings("SpectrographPanel", _SETTINGS_KEY_PREFIX)
        s.set("backend", self.backend_combo.currentText())
        s.set("grating", self._selected_grating())
        s.set("wavelength_nm", self.wavelength_spin.value())
        s.set("pixel_width_um", self.pixel_width_spin.value())

    def _restore_settings(self) -> None:
        s = PanelSettings("SpectrographPanel", _SETTINGS_KEY_PREFIX)
        idx = self.backend_combo.findText(s.get_str("backend", "Mock"))
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)
        self.wavelength_spin.setValue(s.get_float("wavelength_nm", 500.0))
        self.pixel_width_spin.setValue(s.get_float("pixel_width_um", 26.0))
        # The active grating is read back from the device on connect, so it is
        # not restored here.
