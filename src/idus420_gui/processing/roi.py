"""ROI integration helpers for FVB spectra."""

from __future__ import annotations

from typing import Literal

import numpy as np


def integrate_roi(
    frames: np.ndarray,
    pixel_start: int,
    pixel_end: int,
    method: Literal["sum", "mean"] = "sum",
) -> np.ndarray:
    """Integrate inclusive pixel ROI from frames shaped ``(N, xpix)``.

    Returns one scalar per frame as a float64 vector.  The caller must pass a
    2-D array; 1-D inputs are rejected to prevent silent caller bugs.
    """
    arr = np.asarray(frames)
    if arr.ndim != 2:
        raise ValueError("frames must be a 2-D array shaped (N, xpix).")
    if pixel_start < 0 or pixel_end < 0:
        raise ValueError("ROI pixel indices must be non-negative.")
    if pixel_start > pixel_end:
        raise ValueError("pixel_start must be <= pixel_end.")
    if pixel_end >= arr.shape[1]:
        raise ValueError(f"pixel_end {pixel_end} exceeds detector width {arr.shape[1]}.")
    roi = arr[:, pixel_start : pixel_end + 1].astype(np.float64, copy=False)
    if method == "sum":
        return np.sum(roi, axis=1)
    if method == "mean":
        return np.mean(roi, axis=1)
    raise ValueError("method must be 'sum' or 'mean'.")
