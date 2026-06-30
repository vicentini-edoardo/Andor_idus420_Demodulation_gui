from __future__ import annotations

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from idus420_gui.__main__ import _APP_ICON_PATH, _configure_window_icon
from idus420_gui.gui.main_window import MainWindow


def test_app_and_main_window_use_repo_icon(qtbot) -> None:  # type: ignore[no-untyped-def]
    app = QApplication.instance()
    assert app is not None
    assert _APP_ICON_PATH.is_file()

    previous_icon = app.windowIcon()
    window = MainWindow()
    qtbot.addWidget(window)

    try:
        _configure_window_icon(app, window)

        expected = QIcon(str(_APP_ICON_PATH))
        assert not expected.isNull()
        assert app.windowIcon().pixmap(32, 32).toImage() == expected.pixmap(32, 32).toImage()
        assert window.windowIcon().pixmap(32, 32).toImage() == expected.pixmap(32, 32).toImage()
    finally:
        app.setWindowIcon(previous_icon)
