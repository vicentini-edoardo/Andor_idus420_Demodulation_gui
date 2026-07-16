from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QSettings

from idus420_gui.camera.base import AcquisitionTimings
from idus420_gui.camera.mock import MockBackend
from idus420_gui.gui.panel_demod import DemodPanel
from idus420_gui.gui.panel_live import LiveSpectrumPanel
from idus420_gui.io.rp_state import RPStateError


@pytest.fixture(autouse=True)
def clear_settings() -> None:
    for app in ("DemodPanel", "LiveSpectrumPanel"):
        QSettings("idus420_gui", app).clear()
    yield
    for app in ("DemodPanel", "LiveSpectrumPanel"):
        QSettings("idus420_gui", app).clear()


def _write_state(path: Path, **overrides) -> Path:
    data = {
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
        "frequency_shift_hz": -37.0,
        "expected_peak_hz": 37.0,
        "input_frequency_hz": 1_000_000.0,
        "output_frequency_hz": 999_963.0,
        "control": 1,
        "harmonic_n": 1,
        "trig_phase_step": 1125899915,
        "phase_step_offset": -83316594,
        "phase_step_base": 2251799833685,
        "phase_step": 2251716517091,
        "osc_half_period": 0,
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_sync_sets_trigger_and_fft_target_from_confirmed_state(qtbot, tmp_path: Path) -> None:
    panel = DemodPanel()
    qtbot.addWidget(panel)
    path = _write_state(tmp_path / "rp_state.json")
    panel.rp_state_path.setText(str(path))
    panel.rp_sync_cb.setChecked(True)
    emitted: list[float] = []
    panel.rp_trigger_synced.connect(emitted.append)

    state = panel.sync_rp_state(required=True)

    assert state is not None
    assert panel.trigger_spin.value() == pytest.approx(500.0)
    assert panel.expected.value() == pytest.approx(37.0)
    assert emitted[-1] == pytest.approx(500.0)
    assert "confirmed" in panel.rp_status_label.text().lower()


def test_sync_preserves_sub_hertz_register_precision(qtbot, tmp_path: Path) -> None:
    panel = DemodPanel()
    qtbot.addWidget(panel)
    path = _write_state(
        tmp_path / "rp_state.json",
        trigger_frequency_hz=500.123456,
        frequency_shift_hz=37.123456,
        expected_peak_hz=37.123456,
    )
    panel.rp_state_path.setText(str(path))
    panel.rp_sync_cb.setChecked(True)

    panel.sync_rp_state(required=True)

    assert panel.trigger_spin.value() == pytest.approx(500.123456, abs=5e-7)
    assert panel.expected.value() == pytest.approx(37.123456, abs=5e-7)


def test_sync_rejects_stale_state(qtbot, tmp_path: Path) -> None:
    panel = DemodPanel()
    qtbot.addWidget(panel)
    panel.rp_state_path.setText(
        str(_write_state(tmp_path / "rp_state.json", updated_at=time.time() - 10.0))
    )
    panel.rp_sync_cb.setChecked(True)

    with pytest.raises(RPStateError, match="stale"):
        panel.sync_rp_state(required=True)


def test_preflight_rejects_fft_search_above_nyquist(qtbot) -> None:
    panel = DemodPanel()
    qtbot.addWidget(panel)
    panel.trigger_spin.setValue(100.0)
    panel.expected.setValue(49.0)
    panel.search.setValue(2.0)

    with pytest.raises(ValueError, match="Nyquist"):
        panel.settings_for_run()


def test_preflight_rejects_trigger_above_camera_limit(qtbot, monkeypatch) -> None:
    backend = MockBackend()
    backend.connect()
    monkeypatch.setattr(
        backend,
        "query_timings",
        lambda: AcquisitionTimings(0.01, 0.01, 0.01, None),
    )
    panel = DemodPanel()
    qtbot.addWidget(panel)
    panel.set_backend(backend)
    panel.trigger_spin.setValue(101.0)
    panel.expected.setValue(10.0)
    panel.search.setValue(1.0)

    with pytest.raises(ValueError, match="camera maximum"):
        panel.settings_for_run()


def test_live_panel_accepts_synced_trigger(qtbot) -> None:
    panel = LiveSpectrumPanel()
    qtbot.addWidget(panel)

    panel.set_trigger_frequency(321.123456)

    assert panel.trigger_spin.value() == pytest.approx(321.123456, abs=5e-7)


def test_follow_poll_resumes_after_worker_finishes(qtbot, monkeypatch) -> None:
    panel = DemodPanel()
    qtbot.addWidget(panel)
    panel.rp_sync_cb.blockSignals(True)
    panel.rp_sync_cb.setChecked(True)
    panel.rp_sync_cb.blockSignals(False)
    panel.worker = SimpleNamespace(isRunning=lambda: False)  # type: ignore[assignment]
    calls: list[bool] = []
    monkeypatch.setattr(panel, "sync_rp_state", lambda **_kwargs: calls.append(True))

    panel._poll_rp_state()  # noqa: SLF001

    assert calls == [True]
