from __future__ import annotations

import json
import time
from pathlib import Path

from PyQt6.QtCore import QSettings

from idus420_gui.camera.mock import MockBackend
from idus420_gui.gui.panel_acquire import AcquisitionPanel
from idus420_gui.gui.panel_demod import DemodPanel


class _Signal:
    def connect(self, _slot) -> None:  # type: ignore[no-untyped-def]
        pass


class _Worker:
    frame_acquired = _Signal()
    progress = _Signal()
    demod_result = _Signal()
    run_finished = _Signal()
    error = _Signal()
    worker_finished = _Signal()

    def __init__(self, _backend, settings, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        self.settings = settings
        self.started = False

    def start(self) -> None:
        self.started = True


def _write_state(path: Path, shift_hz: float = 37.0) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "source": "redpitaya_ttl_frequency_divider",
        "connected": True,
        "hardware_confirmed": True,
        "sequence": 1,
        "updated_at": time.time(),
        "mode": "pulse",
        "output_mode": "modulated",
        "period_stable": True,
        "trigger_frequency_hz": 500.0,
        "frequency_shift_hz": shift_hz,
        "expected_peak_hz": abs(shift_hz),
        "control": 1,
        "harmonic_n": 1,
        "trig_phase_step": 1125899915,
        "phase_step_offset": round(shift_hz * 1000),
        "osc_half_period": 0,
    }), encoding="utf-8")


def test_acquisition_snapshots_rp_state_before_worker_start(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    QSettings("idus420_gui", "AcquirePanel").clear()
    QSettings("idus420_gui", "DemodPanel").clear()
    backend = MockBackend()
    backend.connect()
    demod = DemodPanel()
    panel = AcquisitionPanel(demod)
    qtbot.addWidget(demod)
    qtbot.addWidget(panel)
    demod.set_backend(backend)
    panel.set_backend(backend)
    path = tmp_path / "rp_state.json"
    _write_state(path, 37.0)
    panel.rp_state_cb.setChecked(True)
    panel.rp_state_path.setText(str(path))
    monkeypatch.setattr("idus420_gui.gui.panel_acquire.AcquisitionWorker", _Worker)

    panel.start()
    _write_state(path, 50.0)

    assert panel.worker is not None
    assert panel.worker.started
    assert panel._pending_rp_start.expected_peak_hz == 37.0  # noqa: SLF001
    assert panel._pending_settings.f_expected == 37.0  # noqa: SLF001
