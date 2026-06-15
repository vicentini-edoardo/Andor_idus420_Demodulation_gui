from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from idus420_gui.gui.panel_scan import _reduce_channel


def test_amplitude_channel_uses_arithmetic_mean() -> None:
    assert _reduce_channel("M1A", [1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_phase_channel_uses_circular_mean() -> None:
    # +179° and -179° straddle the ±pi wrap; the circular mean is 180°, not 0°.
    vals = [math.radians(179.0), math.radians(-179.0)]
    result = _reduce_channel("M1P", vals)
    assert abs(result) == pytest.approx(math.pi, abs=1e-6)


def test_phase_arithmetic_mean_would_be_wrong() -> None:
    vals = [math.radians(179.0), math.radians(-179.0)]
    # Arithmetic mean collapses to ~0 — the bug this guards against.
    assert np.mean(vals) == pytest.approx(0.0, abs=1e-6)
    assert abs(_reduce_channel("M1P", vals)) > 3.0


def test_all_nan_returns_nan() -> None:
    assert math.isnan(_reduce_channel("M1A", [float("nan"), float("nan")]))


def test_nan_values_are_ignored() -> None:
    assert _reduce_channel("M1A", [2.0, float("nan"), 4.0]) == pytest.approx(3.0)
