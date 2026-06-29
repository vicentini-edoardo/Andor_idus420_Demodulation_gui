"""Pure tests for the combo-label <-> enum maps shared by the Camera panel.

These run without Qt and guard against the GUI combo items and the
label->enum lookup drifting apart (which would raise KeyError at apply time).
"""

from __future__ import annotations

from idus420_gui.camera.base import (
    READ_MODE_LABELS,
    SHUTTER_MODE_LABELS,
    ReadMode,
    ShutterMode,
)


def test_shutter_labels_cover_user_facing_modes() -> None:
    assert set(SHUTTER_MODE_LABELS.values()) == {
        ShutterMode.OPEN,
        ShutterMode.AUTO,
        ShutterMode.CLOSED,
    }
    # Labels are unique and order is the combo order (Open first is the default).
    assert len(set(SHUTTER_MODE_LABELS)) == len(SHUTTER_MODE_LABELS)
    assert next(iter(SHUTTER_MODE_LABELS.values())) is ShutterMode.OPEN


def test_read_mode_labels_cover_supported_modes() -> None:
    assert set(READ_MODE_LABELS.values()) == {ReadMode.FVB, ReadMode.SINGLE_TRACK}
    assert next(iter(READ_MODE_LABELS.values())) is ReadMode.FVB


def test_label_lookup_round_trips() -> None:
    for label, mode in SHUTTER_MODE_LABELS.items():
        assert SHUTTER_MODE_LABELS[label] is mode
    for label, mode in READ_MODE_LABELS.items():
        assert READ_MODE_LABELS[label] is mode
