from __future__ import annotations

from idus420_gui.io.powermeter import read_power_metadata


def test_read_power_metadata_returns_none_without_hardware() -> None:
    """No PM100D / pyvisa installed on the test box: must degrade, never raise."""
    assert read_power_metadata() is None


def test_read_power_metadata_uses_prefix_and_never_raises() -> None:
    assert read_power_metadata(prefix="power_before_") is None
