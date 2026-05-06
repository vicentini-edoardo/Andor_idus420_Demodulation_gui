"""Application entry point."""

from __future__ import annotations

import logging
import sys


def main() -> int:
    """Launch the Qt application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from PyQt6.QtWidgets import QApplication

    from idus420_gui.gui.main_window import MainWindow
    from idus420_gui.gui.theme import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.resize(1320, 880)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
