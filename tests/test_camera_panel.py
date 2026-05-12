from __future__ import annotations

import pytest
from PyQt6.QtCore import QSettings

pytest.importorskip("PyQt6")

from idus420_gui.camera.base import AcquisitionTimings
from idus420_gui.camera.mock import MockBackend
from idus420_gui.gui.panel_camera import CameraPanel


@pytest.fixture(autouse=True)
def clear_camera_panel_settings() -> None:
    settings = QSettings("idus420_gui", "CameraPanel")
    settings.clear()
    settings.sync()
    yield
    settings.clear()
    settings.sync()


def test_camera_panel_max_frequency_uses_slowest_reported_period(qtbot, monkeypatch) -> None:
    backend = MockBackend()
    backend.connect()
    monkeypatch.setattr(
        backend,
        "query_timings",
        lambda: AcquisitionTimings(
            exposure_s=0.002,
            accumulate_s=0.010,
            kinetic_s=0.004,
            readout_s=0.020,
        ),
    )

    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.backend = backend
    panel._populate_camera_values()  # noqa: SLF001 - mirror post-connect setup

    panel._apply_config()  # noqa: SLF001 - deliberate GUI behavior test

    assert "max ext trigger 100 Hz" in panel.actual_label.text()
