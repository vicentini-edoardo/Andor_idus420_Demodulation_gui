from __future__ import annotations

import pytest
from PyQt6.QtCore import QSettings

pytest.importorskip("PyQt6")

from idus420_gui.camera.mock import MockBackend
from idus420_gui.gui.panel_live import LiveSpectrumPanel


@pytest.fixture(autouse=True)
def clear_live_panel_settings() -> None:
    settings = QSettings("idus420_gui", "LiveSpectrumPanel")
    settings.clear()
    settings.sync()
    yield
    settings.clear()
    settings.sync()


def test_live_panel_rejects_invalid_roi(qtbot) -> None:  # type: ignore[no-untyped-def]
    panel = LiveSpectrumPanel()
    qtbot.addWidget(panel)
    panel.roi_start.setValue(20)
    panel.roi_end.setValue(10)

    messages: list[str] = []
    panel.log_message.connect(messages.append)

    assert panel._validate_roi() is False  # noqa: SLF001 - deliberate GUI validation test
    assert messages == ["ROI1 start must be ≤ ROI1 end."]


def test_live_panel_clamps_roi_to_detector_width(qtbot) -> None:  # type: ignore[no-untyped-def]
    backend = MockBackend(detector=(64, 255))
    backend.connect()

    panel = LiveSpectrumPanel()
    qtbot.addWidget(panel)
    panel.roi_start.setValue(80)
    panel.roi_end.setValue(120)

    panel.set_backend(backend)

    assert panel.roi_start.maximum() == 63
    assert panel.roi_end.maximum() == 63
    assert panel.roi_start.value() == 63
    assert panel.roi_end.value() == 63


def test_live_panel_spinboxes_follow_roi_region(qtbot) -> None:  # type: ignore[no-untyped-def]
    backend = MockBackend(detector=(128, 255))
    backend.connect()

    panel = LiveSpectrumPanel()
    qtbot.addWidget(panel)
    panel.set_backend(backend)

    panel.roi_region.setRegion((11.2, 21.6))
    qtbot.waitUntil(
        lambda: panel.roi_start.value() == 11 and panel.roi_end.value() == 22,
        timeout=1000,
    )

    assert tuple(round(v) for v in panel.roi_region.getRegion()) == (11, 22)
