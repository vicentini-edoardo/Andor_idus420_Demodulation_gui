"""Main application window."""

from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget

from idus420_gui.camera.base import AcquisitionStatus, CameraBackend, TempStatus
from idus420_gui.gui.panel_acquire import AcquisitionPanel
from idus420_gui.gui.panel_camera import CameraPanel
from idus420_gui.gui.panel_demod import DemodPanel
from idus420_gui.gui.panel_live import LiveSpectrumPanel
from idus420_gui.gui.widgets import LogView


class MainWindow(QMainWindow):
    """Top-level window with camera, demodulation, and acquisition tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Andor iDus 420 Acquisition")
        self.backend: CameraBackend | None = None
        self.acquisition_running = False
        self.temperature_stable = False

        self.tabs = QTabWidget()
        self.camera_panel = CameraPanel()
        self.live_panel = LiveSpectrumPanel()
        self.demod_panel = DemodPanel()
        self.acquire_panel = AcquisitionPanel(self.demod_panel)
        self.tabs.addTab(self.camera_panel, "Camera Settings")
        self.tabs.addTab(self.live_panel, "Live Spectrum")
        self.tabs.addTab(self.demod_panel, "Demodulation Alignment")
        self.tabs.addTab(self.acquire_panel, "Acquisition")
        self.setCentralWidget(self.tabs)

        self.connection_label = QLabel("Disconnected")
        self.connection_label.setObjectName("connection_label")
        self.connection_label.setProperty("connected", "false")
        self.temperature_label = QLabel("Temp: --")
        self.acquisition_label = QLabel("Idle")
        self.acquisition_label.setObjectName("acquisition_label")
        self.acquisition_label.setProperty("running", "false")
        self.log_view = LogView()
        self.statusBar().addWidget(self.connection_label)
        self.statusBar().addPermanentWidget(self.temperature_label)
        self.statusBar().addPermanentWidget(self.acquisition_label)
        self.log_view.setWindowTitle("Log")
        self.statusBar().showMessage("Ready")

        self.camera_panel.backend_changed.connect(self._set_backend)
        self.camera_panel.connection_changed.connect(self._connection_changed)
        self.camera_panel.temperature_changed.connect(self._temperature_changed)
        self.camera_panel.log_message.connect(self.log)
        self.camera_panel.exposure_changed.connect(self.live_panel.set_exposure)
        self.camera_panel.exposure_changed.connect(self.demod_panel.set_exposure)
        self.camera_panel.frame_geometry_changed.connect(self.live_panel.set_frame_width)
        self.camera_panel.frame_geometry_changed.connect(self.demod_panel.set_frame_width)
        self.live_panel.log_message.connect(self.log)
        self.demod_panel.log_message.connect(self.log)
        self.acquire_panel.log_message.connect(self.log)
        self.live_panel.running_changed.connect(self._running_changed)
        self.demod_panel.running_changed.connect(self._running_changed)
        self.acquire_panel.running_changed.connect(self._running_changed)

        # Acquisition-status poll: runs only when NOT acquiring (1 Hz).
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self._poll_status)
        self.status_timer.start()

        # Temperature poll: paused during acquisition to avoid SDK contention (5 s).
        self.temperature_timer = QTimer(self)
        self.temperature_timer.setInterval(5000)
        self.temperature_timer.timeout.connect(self.camera_panel.poll_temperature)
        self.temperature_timer.start()

        self._update_tab_state()

    def log(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        self.log_view.append_line(message)
        if message.lower().startswith(("connection failed", "no triggers", "save")):
            return
        if "failed" in message.lower() or "error" in message.lower():
            QMessageBox.warning(self, "Camera Error", message)

    def _set_backend(self, backend: CameraBackend) -> None:
        self.backend = backend
        self.live_panel.set_backend(backend)
        self.demod_panel.set_backend(backend)
        self.acquire_panel.set_backend(backend)

    def _connection_changed(self, connected: bool) -> None:
        if not connected:
            self.backend = None
            self.live_panel.set_backend(None)
            self.demod_panel.set_backend(None)
            self.acquire_panel.set_backend(None)
            self.temperature_stable = False
        self.connection_label.setText("Connected" if connected else "Disconnected")
        self.connection_label.setProperty("connected", "true" if connected else "false")
        self.connection_label.style().unpolish(self.connection_label)
        self.connection_label.style().polish(self.connection_label)
        self._update_tab_state()

    def _temperature_changed(self, temp: float, status: TempStatus) -> None:
        self.temperature_stable = status is TempStatus.STABILIZED
        self.temperature_label.setText(f"Temp: {temp:.1f} C ({status.value})")
        self._update_tab_state()

    def _running_changed(self, running: bool) -> None:
        self.acquisition_running = running
        self.acquisition_label.setText("Running" if running else "Idle")
        self.acquisition_label.setProperty("running", "true" if running else "false")
        self.acquisition_label.style().unpolish(self.acquisition_label)
        self.acquisition_label.style().polish(self.acquisition_label)
        # Pause both polls during acquisition to avoid SDK contention.
        if running:
            self.status_timer.stop()
            self.temperature_timer.stop()
        else:
            self.status_timer.start()
            self.temperature_timer.start()
        self._update_tab_state()

    def _poll_status(self) -> None:
        self.camera_panel.poll_temperature()
        if self.backend and self.backend.is_connected():
            self.acquisition_label.setText(self.backend.status().value)
        else:
            self.acquisition_label.setText(AcquisitionStatus.IDLE.value)

    def _update_tab_state(self) -> None:
        connected = self.backend is not None and self.backend.is_connected()
        enabled = connected and not self.acquisition_running
        self.tabs.setTabEnabled(1, enabled or self.acquisition_running)
        self.tabs.setTabEnabled(2, enabled or self.acquisition_running)
        self.tabs.setTabEnabled(3, enabled or self.acquisition_running)

    def closeEvent(self, event: object) -> None:
        if self.live_panel.worker:
            self.live_panel.worker.stop()
            self.live_panel.worker.wait(2000)
        if self.demod_panel.worker:
            self.demod_panel.worker.stop()
            self.demod_panel.worker.wait(2000)
        if self.acquire_panel.worker:
            self.acquire_panel.worker.stop()
            self.acquire_panel.worker.wait(2000)
        if self.backend:
            self.backend.disconnect()
        super().closeEvent(event)  # type: ignore[arg-type]
