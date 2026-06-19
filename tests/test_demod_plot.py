from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from idus420_gui.gui.panel_demod import _MAX_PLOT_POINTS, _decimate


def test_small_array_is_not_decimated() -> None:
    y = np.arange(100.0)
    x, out = _decimate(y)
    assert x is None
    assert np.array_equal(out, y)


def test_large_array_is_capped() -> None:
    y = np.arange(_MAX_PLOT_POINTS * 5, dtype=np.float64)
    x, out = _decimate(y)
    assert x is None
    assert out.shape[0] <= _MAX_PLOT_POINTS
    assert out[0] == y[0]


def test_decimation_keeps_x_and_y_aligned() -> None:
    n = _MAX_PLOT_POINTS * 3
    y = np.arange(n, dtype=np.float64)
    xaxis = np.linspace(0.0, 10.0, n)
    x, out = _decimate(y, xaxis)
    assert x is not None
    assert x.shape == out.shape
    # x and y were sampled at the same indices, so y == index and x == xaxis[index].
    assert np.allclose(xaxis[out.astype(int)], x)
