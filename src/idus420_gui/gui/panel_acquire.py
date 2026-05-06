"""Fixed-duration acquisition panel."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
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
from idus420_gui.io.save import save_h5, save_npz, save_txt
from idus420_gui.workers.acquisition import AcquisitionWorker, DemodulationSettings

_SETTINGS_KEY_PREFIX = "acquire_panel"


class AcquisitionPanel(QWidget):
    """One-shot acquisition and save workflow."""

    log_message = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, demod_source: DemodPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.backend: CameraBackend | None = None
        self.demod_source = demod_source
        self.worker: AcquisitionWorker | None = None
        self._run_start_time: float = 0.0
        self._build_ui()
        self._restore_settings()

    def set_backend(self, backend: CameraBackend | None) -> None:
        self.backend = backend

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # --- Controls pane ---
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        controls_box = QGroupBox("Acquisition")
        grid = QGridLayout(controls_box)
        grid.setSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.duration_s = QSpinBox()
        self.duration_s.setRange(1, 86400)
        self.duration_s.setValue(10)
        self.duration_s.setSuffix(" s")

        self.total_frames = QSpinBox()
        self.total_frames.setRange(1, 100_000_000)
        self.total_frames.setValue(5000)

        self.use_frames = QCheckBox("Use total frames instead of duration")

        self.output_dir = QLineEdit(str(Path.cwd()))
        self.choose_dir = QPushButton("Browse")
        self.choose_dir.setFixedWidth(70)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self.output_dir)
        dir_row.addWidget(self.choose_dir)

        self.stem = QLineEdit("idus420_run")

        # Format checkboxes stacked vertically
        self.save_npz_cb = QCheckBox(".npz")
        self.save_npz_cb.setChecked(True)
        self.save_h5_cb = QCheckBox(".h5")
        self.save_txt_cb = QCheckBox(".txt  (tab-separated, metadata header)")
        self.save_sif_cb = QCheckBox(".sif via Andor SDK")

        format_col = QVBoxLayout()
        format_col.setSpacing(3)
        format_col.addWidget(self.save_npz_cb)
        format_col.addWidget(self.save_h5_cb)
        format_col.addWidget(self.save_txt_cb)
        format_col.addWidget(self.save_sif_cb)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)

        self.progress = QProgressBar()
        self.elapsed_label = QLabel("--")

        row = 0
        grid.addWidget(_lbl("Duration"), row, 0)
        grid.addWidget(self.duration_s, row, 1)
        grid.addWidget(_lbl("Total frames"), row, 2)
        grid.addWidget(self.total_frames, row, 3)
        row += 1
        grid.addWidget(self.use_frames, row, 0, 1, 4)
        row += 1
        grid.addWidget(_lbl("Output dir"), row, 0)
        grid.addLayout(dir_row, row, 1, 1, 3)
        row += 1
        grid.addWidget(_lbl("Filename stem"), row, 0)
        grid.addWidget(self.stem, row, 1, 1, 3)
        row += 1
        grid.addWidget(_lbl("Formats"), row, 0)
        grid.addLayout(format_col, row, 1, 1, 3)
        row += 1
        grid.addLayout(button_row, row, 0, 1, 4)
        row += 1
        grid.addWidget(_lbl("Progress"), row, 0)
        grid.addWidget(self.progress, row, 1, 1, 3)
        row += 1
        grid.addWidget(_lbl("Status"), row, 0)
        grid.addWidget(self.elapsed_label, row, 1, 1, 3)

        controls_layout.addWidget(controls_box)
        controls_layout.addStretch(1)
        controls_container.setMinimumWidth(300)
        controls_container.setMaximumWidth(520)

        splitter.addWidget(controls_container)

        # --- Plot pane ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(theme.BG)

        self.spectrum_plot = self.plot_widget.addPlot(row=0, col=0, title="Live spectrum")
        self.history_plot = self.plot_widget.addPlot(
            row=0, col=1, title="Running peak amplitude"
        )

        for pi in (self.spectrum_plot, self.history_plot):
            theme._style_plot_item(pi)  # noqa: SLF001

        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen(theme.CURVE_YELLOW, width=1.5)
        )
        self.history_curve = self.history_plot.plot(
            pen=pg.mkPen(theme.CURVE_GREEN, width=1.5)
        )

        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 900])

        outer.addWidget(splitter, stretch=1)

        self.choose_dir.clicked.connect(self._choose_directory)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)

    def start(self) -> None:
        if not self.backend:
            self.log_message.emit("No camera backend is connected.")
            return
        if not self.demod_source._validate_roi():  # noqa: SLF001 - intentional internal access
            return
        settings: DemodulationSettings = self.demod_source.settings()
        total_frames = self.total_frames.value() if self.use_frames.isChecked() else None
        total_seconds = None if self.use_frames.isChecked() else float(self.duration_s.value())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.output_dir.text())
        stem = f"{self.stem.text()}_{timestamp}"
        self._pending_stem = stem
        self._pending_timestamp = timestamp
        sif_path = str(out_dir / f"{stem}.sif") if self.save_sif_cb.isChecked() else None
        self.worker = AcquisitionWorker(
            self.backend, settings, total_seconds, total_frames, sif_path=sif_path
        )
        self.worker.frame_acquired.connect(lambda frame: self.spectrum_curve.setData(frame))
        self.worker.progress.connect(self._handle_progress)
        self.worker.demod_result.connect(self._handle_demod_result)
        self.worker.run_finished.connect(self._save_run)
        self.worker.error.connect(self.log_message.emit)
        self.worker.worker_finished.connect(lambda: self._set_running_ui(False))
        self._peak_history: list[float] = []
        self._set_running_ui(True)
        import time
        self._run_start_time = time.monotonic()
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()

    def _save_run(self, frames: object, processed: object) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = getattr(self, "_pending_stem", None) or now
        timestamp = getattr(self, "_pending_timestamp", None) or now
        out_dir = Path(self.output_dir.text())
        settings = self.demod_source.settings()
        metadata = {
            "software": "idus420_gui",
            "timestamp": timestamp,
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
        roi_ts = processed["roi_timeseries"]  # type: ignore[index]
        results = processed["demod_results"]  # type: ignore[index]
        if self.save_npz_cb.isChecked():
            path = out_dir / f"{stem}.npz"
            save_npz(path, frames, roi_ts, results, metadata)  # type: ignore[arg-type]
            self.log_message.emit(f"Saved {path}")
        if self.save_h5_cb.isChecked():
            path = out_dir / f"{stem}.h5"
            save_h5(path, frames, roi_ts, results, metadata)  # type: ignore[arg-type]
            self.log_message.emit(f"Saved {path}")
        if self.save_txt_cb.isChecked():
            path = out_dir / f"{stem}.txt"
            save_txt(path, frames, roi_ts, results, metadata)  # type: ignore[arg-type]
            self.log_message.emit(f"Saved {path}")
        if self.save_sif_cb.isChecked():
            self.log_message.emit(f"Saved {out_dir / f'{stem}.sif'}")
        self._save_settings()

    def _handle_progress(self, acquired: int, total: int, elapsed_s: float) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(acquired)
        if acquired > 0 and elapsed_s > 0:
            rate = acquired / elapsed_s
            remaining_s = (total - acquired) / rate if rate > 0 else 0.0
            self.elapsed_label.setText(
                f"{acquired} / {total} frames  —  "
                f"elapsed {elapsed_s:.0f} s, ETA {remaining_s:.0f} s"
            )
        else:
            self.elapsed_label.setText(f"{acquired} / {total} frames")

    def _handle_demod_result(self, result: object) -> None:
        self._peak_history.append(float(result.peak_amplitude))  # type: ignore[attr-defined]
        self._peak_history = self._peak_history[-600:]
        self.history_curve.setData(self._peak_history)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Output directory",
            self.output_dir.text(),
        )
        if directory:
            self.output_dir.setText(directory)

    def _set_running_ui(self, running: bool) -> None:
        for widget in [
            self.duration_s,
            self.total_frames,
            self.use_frames,
            self.output_dir,
            self.choose_dir,
            self.stem,
            self.save_npz_cb,
            self.save_h5_cb,
            self.save_txt_cb,
            self.save_sif_cb,
            self.start_button,
        ]:
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.running_changed.emit(running)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_settings(self) -> None:
        s = QSettings("idus420_gui", "AcquirePanel")
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/output_dir", self.output_dir.text())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/stem", self.stem.text())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/duration_s", self.duration_s.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/total_frames", self.total_frames.value())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/use_frames", self.use_frames.isChecked())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/save_npz", self.save_npz_cb.isChecked())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/save_h5", self.save_h5_cb.isChecked())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/save_txt", self.save_txt_cb.isChecked())
        s.setValue(f"{_SETTINGS_KEY_PREFIX}/save_sif", self.save_sif_cb.isChecked())

    def _restore_settings(self) -> None:
        s = QSettings("idus420_gui", "AcquirePanel")

        def bval(key: str, default: bool) -> bool:
            v = s.value(f"{_SETTINGS_KEY_PREFIX}/{key}")
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            return str(v).lower() in {"true", "1", "yes"}

        if (v := s.value(f"{_SETTINGS_KEY_PREFIX}/output_dir")) is not None:
            self.output_dir.setText(str(v))
        if (v := s.value(f"{_SETTINGS_KEY_PREFIX}/stem")) is not None:
            self.stem.setText(str(v))
        if (v := s.value(f"{_SETTINGS_KEY_PREFIX}/duration_s")) is not None:
            self.duration_s.setValue(int(v))
        if (v := s.value(f"{_SETTINGS_KEY_PREFIX}/total_frames")) is not None:
            self.total_frames.setValue(int(v))
        self.use_frames.setChecked(bval("use_frames", False))
        self.save_npz_cb.setChecked(bval("save_npz", True))
        self.save_h5_cb.setChecked(bval("save_h5", False))
        self.save_txt_cb.setChecked(bval("save_txt", False))
        self.save_sif_cb.setChecked(bval("save_sif", False))


def _lbl(text: str) -> QLabel:
    return QLabel(text)
