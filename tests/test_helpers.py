from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from idus420_gui.gui._helpers import PanelSettings, roi_error_message


def test_roi_error_message_rejects_inverted_range() -> None:
    assert roi_error_message(20, 10, 1024) == "ROI start must be ≤ ROI end."


def test_roi_error_message_rejects_out_of_bounds_end() -> None:
    assert roi_error_message(0, 1024, 1024) == (
        "ROI end 1024 exceeds detector width 1024."
    )


def test_roi_error_message_accepts_valid_range() -> None:
    assert roi_error_message(10, 20, 1024) is None


def test_panel_settings_round_trip_and_bool_parsing() -> None:
    s = PanelSettings("HelpersTest", "prefix")
    s.set("flag", True)
    s.set("count", 7)
    s.set("rate", 3.5)
    s.set("name", "hann")
    assert s.get_bool("flag", False) is True
    assert s.get_int("count", 0) == 7
    assert s.get_float("rate", 0.0) == 3.5
    assert s.get_str("name", "") == "hann"
    assert s.get_bool("missing", True) is True


def test_panel_settings_parses_stringified_bool() -> None:
    s = PanelSettings("HelpersTest", "prefix")
    s.set("flag", "true")
    assert s.get_bool("flag", False) is True
    s.set("flag", "0")
    assert s.get_bool("flag", True) is False
