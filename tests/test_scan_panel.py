from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

import numpy as np
from PyQt6.QtCore import QSettings

from idus420_gui.camera.mock import MockBackend
from idus420_gui.gui.panel_demod import DemodPanel
from idus420_gui.motion.base import StagePoint
from idus420_gui.gui.panel_scan import (
    NEA_TOOLS_AVAILABLE,
    ScanPanel,
    _DEMOD_CHANNELS,
    _SNOM_CHANNELS,
)
from idus420_gui.workers.scan import PointResult


def _make_panels():
    """Create a DemodPanel + ScanPanel pair.

    Callers must pass qtbot so that a QApplication exists before widget
    construction (qtbot.addWidget() is skipped because it rejects PyQt6
    widgets when the test runner is PySide6-backed).
    """
    demod = DemodPanel()
    panel = ScanPanel(demod)
    return demod, panel


def test_panel_builds_without_backend(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, panel = _make_panels()
    assert panel is not None


def test_set_backend_accepted(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, panel = _make_panels()
    backend = MockBackend()
    backend.connect()
    panel.set_backend(backend)
    assert panel.backend is backend


def test_start_without_backend_logs_error(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, panel = _make_panels()
    messages: list[str] = []
    panel.log_message.connect(messages.append)
    panel.start()
    # With no camera (and, in CI, no SNOM stage) start() must refuse and report
    # a backend-related error rather than launching a worker.
    assert panel.worker is None
    assert any("backend" in m.lower() for m in messages)


def test_settings_round_trip(qtbot) -> None:  # type: ignore[no-untyped-def]
    QSettings("idus420_gui", "ScanPanel").clear()

    demod = DemodPanel()
    p1 = ScanPanel(demod)
    p1.nx.setValue(7)
    p1.ny.setValue(4)
    p1.x_length.setValue(3.0)
    p1.y_length.setValue(1.5)
    p1.snom_host.setText("custom-host")
    p1.stem.setText("my_scan")
    p1._save_settings()

    p2 = ScanPanel(demod)
    assert p2.nx.value() == 7
    assert p2.ny.value() == 4
    assert p2.x_length.value() == pytest.approx(3.0)
    assert p2.y_length.value() == pytest.approx(1.5)
    assert p2.snom_host.text() == "custom-host"
    assert p2.stem.text() == "my_scan"


def test_total_label_updates(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, panel = _make_panels()
    panel.nx.setValue(4)
    panel.ny.setValue(3)
    assert "12" in panel.total_label.text()


def test_stop_without_worker_does_not_crash(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, panel = _make_panels()
    panel.stop()


def test_scan_panel_plots_latest_point_spectrum(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, panel = _make_panels()
    panel._map_demod = {key: np.full((1, 1), np.nan) for key in _DEMOD_CHANNELS}
    panel._map_snom = {key: np.full((1, 1), np.nan) for key in _SNOM_CHANNELS}
    result = PointResult(
        point=StagePoint(ix=0, iy=0, x_nm=0.0, y_nm=0.0),
        actual_xyz_nm=(0.0, 0.0, 0.0),
        frames=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        roi_timeseries=np.array([6.0, 15.0]),
        demod_results=[],
        snom_samples=[],
    )

    panel._on_point_data(0, result)

    assert panel.spectrum_curve.getData()[1].tolist() == [4.0, 5.0, 6.0]


def test_scan_panel_plots_latest_point_roi_fft(qtbot) -> None:  # type: ignore[no-untyped-def]
    demod, panel = _make_panels()
    demod.trigger_spin.setValue(4.0)
    panel._map_demod = {key: np.full((1, 1), np.nan) for key in _DEMOD_CHANNELS}
    panel._map_snom = {key: np.full((1, 1), np.nan) for key in _SNOM_CHANNELS}
    result = PointResult(
        point=StagePoint(ix=0, iy=0, x_nm=0.0, y_nm=0.0),
        actual_xyz_nm=(0.0, 0.0, 0.0),
        frames=np.array([[1.0, 2.0, 3.0]]),
        roi_timeseries=np.array([0.0, 1.0, 0.0, -1.0]),
        demod_results=[],
        snom_samples=[],
    )

    panel._on_point_data(0, result)

    x, y = panel.roi_fft_curve.getData()
    assert x.tolist() == [0.0, 1.0, 2.0]
    assert y.tolist() == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.skipif(
    not NEA_TOOLS_AVAILABLE,
    reason="ScanPanel.start() requires the nea_tools SNOM stage backend.",
)
def test_running_changed_emitted_on_start(qtbot) -> None:  # type: ignore[no-untyped-def]
    demod, panel = _make_panels()
    backend = MockBackend()
    backend.connect()
    panel.set_backend(backend)

    running_states: list[bool] = []
    panel.running_changed.connect(running_states.append)
    panel.frames_per_point.setValue(8)
    panel.nx.setValue(1)
    panel.ny.setValue(1)

    panel.start()
    assert panel.worker is not None

    with qtbot.waitSignal(panel.worker.scan_finished, timeout=8000):
        pass

    panel.worker.wait(3000)
    assert True in running_states
