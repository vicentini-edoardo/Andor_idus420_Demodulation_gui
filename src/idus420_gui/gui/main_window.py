"""Main application window."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from idus420_gui.camera.base import AcquisitionStatus, CameraBackend, TempStatus
from idus420_gui.gui.panel_acquire import AcquisitionPanel
from idus420_gui.gui.panel_camera import CameraPanel
from idus420_gui.gui.panel_demod import DemodPanel
from idus420_gui.gui.panel_live import LiveSpectrumPanel
from idus420_gui.gui.panel_scan import ScanPanel
from idus420_gui.gui.widgets import LogView

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_git_command(repo_dir: Path, cmd: list[str], run=subprocess.run) -> str:
    result = run(cmd, capture_output=True, text=True, cwd=repo_dir, timeout=30)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"{' '.join(cmd)} failed with exit code {result.returncode}")
    return output


def _run_git_update(repo_dir: Path, branch: str, run=subprocess.run) -> tuple[str, bool]:
    remote_ref = f"origin/{branch}"
    before_head = _run_git_command(repo_dir, ["git", "rev-parse", "HEAD"], run)
    _run_git_command(repo_dir, ["git", "fetch", "--prune", "origin"], run)

    if _run_git_command(repo_dir, ["git", "branch", "--show-current"], run) != branch:
        try:
            _run_git_command(repo_dir, ["git", "checkout", branch], run)
        except RuntimeError:
            _run_git_command(
                repo_dir, ["git", "checkout", "-b", branch, "--track", remote_ref], run
            )

    _run_git_command(
        repo_dir, ["git", "branch", "--set-upstream-to", remote_ref, branch], run
    )
    _run_git_command(repo_dir, ["git", "pull", "--ff-only"], run)
    after_head = _run_git_command(repo_dir, ["git", "rev-parse", "HEAD"], run)
    if before_head == after_head:
        return "Already up to date.", False
    return (
        _run_git_command(
            repo_dir,
            ["git", "log", f"{before_head}..{after_head}", "--pretty=format:• %s", "--no-merges"],
            run,
        )
        or "Updated.",
        True,
    )


class _GitFetchWorker(QThread):
    # (returncode, branch_list, info) — info=current_branch on success, error msg on failure
    finished = pyqtSignal(int, object, str)

    def run(self) -> None:
        try:
            r = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "fetch", "--prune"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                self.finished.emit(r.returncode, [], (r.stdout + r.stderr).strip())
                return

            rb = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "branch", "-r", "--format=%(refname:short)"],
                capture_output=True, text=True,
            )
            branches: list[str] = []
            for line in rb.stdout.splitlines():
                line = line.strip()
                if not line or "->" in line:
                    continue
                prefix = "origin/"
                b = line[len(prefix):] if line.startswith(prefix) else line
                if b not in branches:
                    branches.append(b)

            cur = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "branch", "--show-current"],
                capture_output=True, text=True,
            ).stdout.strip()

            if cur in branches:
                branches.remove(cur)
                branches.insert(0, cur)

            self.finished.emit(0, branches, cur)
        except Exception as exc:
            self.finished.emit(-1, [], str(exc))


class _GitSwitchWorker(QThread):
    finished = pyqtSignal(int, str)  # returncode, output/changelog

    def __init__(self, branch: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._branch = branch

    def run(self) -> None:
        try:
            output, _changed = _run_git_update(_REPO_ROOT, self._branch)
            self.finished.emit(0, output)
        except Exception as exc:
            self.finished.emit(-1, str(exc))


class _BranchDialog(QDialog):
    def __init__(self, branches: list[str], current: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Branch")
        self.setMinimumWidth(300)

        self._combo = QComboBox()
        for b in branches:
            self._combo.addItem(b)
        idx = self._combo.findText(current)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

        btn_ok = QPushButton("Update")
        btn_ok.setProperty("accent", True)
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select branch to switch to and pull:"))
        layout.addWidget(self._combo)
        layout.addLayout(btn_row)

    def selected_branch(self) -> str:
        return self._combo.currentText()


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
        self.scan_panel = ScanPanel(self.demod_panel)
        self.tabs.addTab(self.camera_panel, "Camera Settings")
        self.tabs.addTab(self.live_panel, "Live Spectrum")
        self.tabs.addTab(self.demod_panel, "Demodulation Alignment")
        self.tabs.addTab(self.acquire_panel, "Acquisition")
        self.tabs.addTab(self.scan_panel, "Scan")
        self.setCentralWidget(self.tabs)

        self.connection_label = QLabel("Disconnected")
        self.connection_label.setObjectName("connection_label")
        self.connection_label.setProperty("connected", "false")
        self.temperature_label = QLabel("Temp: --")
        self.acquisition_label = QLabel("Idle")
        self.acquisition_label.setObjectName("acquisition_label")
        self.acquisition_label.setProperty("running", "false")
        self.log_view = LogView()
        self._fetch_worker: _GitFetchWorker | None = None
        self._switch_worker: _GitSwitchWorker | None = None
        self.statusBar().addWidget(self.connection_label)
        # The Update button does `git pull` + restart, which only makes sense
        # when the app runs from a source checkout; hide it for installed copies.
        if (_REPO_ROOT / ".git").exists():
            self._update_btn = QPushButton("Update")
            self._update_btn.setToolTip("git pull and restart")
            self._update_btn.clicked.connect(self._update_app)
            self.statusBar().addPermanentWidget(self._update_btn)
        else:
            self._update_btn = None
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
        self.camera_panel.wavelength_axis_changed.connect(self.live_panel.set_wavelength_axis)
        self.camera_panel.wavelength_axis_changed.connect(self.demod_panel.set_wavelength_axis)
        self.camera_panel.wavelength_axis_changed.connect(self.acquire_panel.set_wavelength_axis)
        self.camera_panel.wavelength_axis_changed.connect(self.scan_panel.set_wavelength_axis)
        self.live_panel.log_message.connect(self.log)
        self.demod_panel.log_message.connect(self.log)
        self.demod_panel.rp_trigger_synced.connect(self.live_panel.set_trigger_frequency)
        self.acquire_panel.log_message.connect(self.log)
        self.live_panel.running_changed.connect(self._running_changed)
        self.demod_panel.running_changed.connect(self._running_changed)
        self.acquire_panel.running_changed.connect(self._running_changed)
        self.scan_panel.running_changed.connect(self._running_changed)
        self.scan_panel.log_message.connect(self.log)

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
        # A modal dialog would block the GUI thread; during a run, workers can
        # emit transient errors (e.g. re-arm notices), so surface them only in
        # the status bar / log and skip the interrupting popup.
        if self.acquisition_running:
            return
        if "failed" in message.lower() or "error" in message.lower():
            QMessageBox.warning(self, "Camera Error", message)

    def _set_backend(self, backend: CameraBackend) -> None:
        self.backend = backend
        self.live_panel.set_backend(backend)
        self.demod_panel.set_backend(backend)
        self.acquire_panel.set_backend(backend)
        self.scan_panel.set_backend(backend)

    def _connection_changed(self, connected: bool) -> None:
        if not connected:
            self.backend = None
            self.live_panel.set_backend(None)
            self.demod_panel.set_backend(None)
            self.acquire_panel.set_backend(None)
            self.scan_panel.set_backend(None)
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
        self.demod_panel.set_external_running(running)
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
        # Temperature is polled separately on its own 5 s timer to limit SDK
        # contention; this 1 Hz status poll only refreshes acquisition state.
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
        # Scan tab is always accessible so parameters can be edited before a
        # camera is connected; ScanPanel.start() guards against missing
        # camera/stage backends itself.
        self.tabs.setTabEnabled(4, True)

    def _update_app(self) -> None:
        if self._update_btn is None:
            return
        self._update_btn.setEnabled(False)
        self._update_btn.setText("Fetching…")
        self._fetch_worker = _GitFetchWorker()
        self._fetch_worker.finished.connect(self._on_fetch_done)
        self._fetch_worker.start()

    def _on_fetch_done(self, returncode: int, branches: object, info: str) -> None:
        if self._update_btn is not None:
            self._update_btn.setEnabled(True)
            self._update_btn.setText("Update")
        if returncode != 0:
            QMessageBox.warning(self, "Update", f"Fetch failed:\n{info}")
            return
        branch_list: list = branches  # type: ignore[assignment]
        if not branch_list:
            QMessageBox.information(self, "Update", "No remote branches found.")
            return
        dlg = _BranchDialog(branch_list, info, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dlg.selected_branch()
        if self._update_btn is not None:
            self._update_btn.setEnabled(False)
            self._update_btn.setText("Updating…")
        self._switch_worker = _GitSwitchWorker(selected, self)
        self._switch_worker.finished.connect(self._on_switch_done)
        self._switch_worker.start()

    def _on_switch_done(self, returncode: int, output: str) -> None:
        if self._update_btn is not None:
            self._update_btn.setEnabled(True)
            self._update_btn.setText("Update")
        msg = QMessageBox(self)
        msg.setWindowTitle("Update")
        if returncode != 0:
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Update failed.")
            msg.setInformativeText(output)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        elif output == "Already up to date.":
            msg.setText("Already up to date.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        else:
            msg.setText("What's new:")
            msg.setInformativeText(output)
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg.button(QMessageBox.StandardButton.Yes).setText("Restart now")
            msg.button(QMessageBox.StandardButton.No).setText("Later")
            if msg.exec() == QMessageBox.StandardButton.Yes:
                self._restart()

    def _restart(self) -> None:
        subprocess.Popen([sys.executable, "-m", "idus420_gui"])
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

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
        if self.scan_panel.worker:
            self.scan_panel.worker.stop()
            self.scan_panel.worker.wait(5000)
        if self.backend:
            self.backend.disconnect()
        super().closeEvent(event)  # type: ignore[arg-type]
