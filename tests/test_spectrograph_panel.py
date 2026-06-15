from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QSettings

pytest.importorskip("PyQt6")

from idus420_gui.gui.panel_spectrograph import SpectrographPanel


@pytest.fixture(autouse=True)
def clear_spectrograph_panel_settings():
    settings = QSettings("idus420_gui", "SpectrographPanel")
    settings.clear()
    settings.sync()
    yield
    settings.clear()
    settings.sync()


def test_panel_connect_populates_gratings_and_emits_calibration(qtbot) -> None:
    panel = SpectrographPanel()
    qtbot.addWidget(panel)
    panel.set_frame_width(1024)

    captured: list[object] = []
    panel.calibration_changed.connect(captured.append)

    panel._connect_backend()  # noqa: SLF001 - mirror Connect button click

    assert panel.backend is not None
    assert panel.grating_combo.count() == 3
    assert captured, "connecting should broadcast an initial calibration"
    cal = captured[-1]
    assert isinstance(cal, np.ndarray)
    assert cal.shape == (1024,)


def test_panel_apply_sets_grating_and_wavelength(qtbot) -> None:
    panel = SpectrographPanel()
    qtbot.addWidget(panel)
    panel.set_frame_width(1024)
    panel._connect_backend()  # noqa: SLF001

    idx = panel.grating_combo.findData(2)
    panel.grating_combo.setCurrentIndex(idx)
    panel.wavelength_spin.setValue(620.0)

    captured: list[object] = []
    panel.calibration_changed.connect(captured.append)
    panel._apply()  # noqa: SLF001 - mirror Apply button click

    assert panel.backend.current_grating() == 2
    assert panel.backend.get_wavelength() == 620.0
    assert isinstance(captured[-1], np.ndarray)
    assert "Grating: 2" in panel.status_label.text()


def test_panel_disconnect_clears_calibration(qtbot) -> None:
    panel = SpectrographPanel()
    qtbot.addWidget(panel)
    panel.set_frame_width(1024)
    panel._connect_backend()  # noqa: SLF001

    captured: list[object] = []
    panel.calibration_changed.connect(captured.append)
    panel._disconnect_backend()  # noqa: SLF001

    assert panel.backend is None
    assert captured[-1] is None


def test_frame_width_change_rebroadcasts_calibration(qtbot) -> None:
    panel = SpectrographPanel()
    qtbot.addWidget(panel)
    panel.set_frame_width(1024)
    panel._connect_backend()  # noqa: SLF001

    captured: list[object] = []
    panel.calibration_changed.connect(captured.append)
    panel.set_frame_width(512)

    cal = captured[-1]
    assert isinstance(cal, np.ndarray)
    assert cal.shape == (512,)
