"""Application entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QWidget

_APP_ICON_PATH = Path(__file__).resolve().parents[2] / "icon.ico"


def _configure_window_icon(app: QApplication, window: QWidget) -> None:
    icon = QIcon(str(_APP_ICON_PATH))
    if icon.isNull():
        return
    app.setWindowIcon(icon)
    window.setWindowIcon(icon)


def main() -> int:
    """Launch the Qt application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from idus420_gui.gui.main_window import MainWindow
    from idus420_gui.gui.theme import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    _configure_window_icon(app, window)
    window.resize(1320, 880)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
