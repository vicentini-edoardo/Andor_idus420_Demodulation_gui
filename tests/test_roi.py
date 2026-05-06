from __future__ import annotations

import numpy as np
import pytest

from idus420_gui.processing.roi import integrate_roi


def test_integrate_roi_sum_and_mean() -> None:
    frames = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.uint16)
    np.testing.assert_allclose(integrate_roi(frames, 1, 2, "sum"), [5, 50])
    np.testing.assert_allclose(integrate_roi(frames, 1, 2, "mean"), [2.5, 25])


def test_integrate_roi_requires_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        integrate_roi(np.array([1, 2, 3], dtype=np.uint16), 0, 1)


@pytest.mark.parametrize(
    ("start", "end", "method"),
    [(-1, 1, "sum"), (2, 1, "sum"), (0, 99, "sum"), (0, 1, "median")],
)
def test_integrate_roi_rejects_invalid_inputs(start: int, end: int, method: str) -> None:
    with pytest.raises(ValueError):
        integrate_roi(np.ones((2, 4)), start, end, method)  # type: ignore[arg-type]

