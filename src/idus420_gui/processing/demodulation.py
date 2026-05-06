"""FFT demodulation for externally-triggered ROI intensity blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class DemodResult:
    """Demodulated peak estimate and full FFT spectrum."""

    peak_frequency: float
    peak_amplitude: float
    f_axis: np.ndarray
    spectrum: np.ndarray
    snr: float


def demodulate(
    time_series: np.ndarray,
    sample_rate_hz: float,
    f_expected: float,
    f_search_halfwidth: float,
    window: Literal["hann", "blackman", "none"] = "hann",
) -> DemodResult:
    """Find the FFT peak near `f_expected` in a real-valued time series."""
    y = np.asarray(time_series, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("time_series must be one-dimensional.")
    if y.size < 4:
        raise ValueError("time_series must contain at least four samples.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if f_expected < 0:
        raise ValueError("f_expected must be non-negative.")
    if f_search_halfwidth < 0:
        raise ValueError("f_search_halfwidth must be non-negative.")

    centered = y - np.mean(y)
    win = _window(window, y.size)
    coherent_gain = float(np.mean(win)) if np.mean(win) else 1.0
    windowed = centered * win
    spectrum = np.abs(np.fft.rfft(windowed)) * 2.0 / (y.size * coherent_gain)
    f_axis = np.fft.rfftfreq(y.size, d=1.0 / sample_rate_hz)

    low = max(0.0, f_expected - f_search_halfwidth)
    high = min(sample_rate_hz / 2.0, f_expected + f_search_halfwidth)
    mask = (f_axis >= low) & (f_axis <= high)
    if not np.any(mask):
        raise ValueError("Search band contains no FFT bins.")

    band_indices = np.flatnonzero(mask)
    peak_idx = int(band_indices[np.argmax(spectrum[band_indices])])
    peak_frequency, peak_amplitude = _parabolic_peak(f_axis, spectrum, peak_idx)

    noise_mask = ~mask
    noise = spectrum[noise_mask]
    noise = noise[np.isfinite(noise)]
    floor = float(np.median(noise)) if noise.size else 0.0
    snr = float(peak_amplitude / floor) if floor > 0 else float("inf")
    return DemodResult(
        peak_frequency=float(peak_frequency),
        peak_amplitude=float(peak_amplitude),
        f_axis=f_axis,
        spectrum=spectrum,
        snr=snr,
    )


def _window(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name == "blackman":
        return np.blackman(n)
    if name == "none":
        return np.ones(n, dtype=np.float64)
    raise ValueError("window must be 'hann', 'blackman', or 'none'.")


def _parabolic_peak(f_axis: np.ndarray, spectrum: np.ndarray, idx: int) -> tuple[float, float]:
    if idx <= 0 or idx >= spectrum.size - 1:
        return float(f_axis[idx]), float(spectrum[idx])
    alpha = float(spectrum[idx - 1])
    beta = float(spectrum[idx])
    gamma = float(spectrum[idx + 1])
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-15:
        return float(f_axis[idx]), beta
    offset = 0.5 * (alpha - gamma) / denom
    offset = float(np.clip(offset, -1.0, 1.0))
    df = float(f_axis[1] - f_axis[0])
    amplitude = beta - 0.25 * (alpha - gamma) * offset
    return float(f_axis[idx] + offset * df), float(max(amplitude, 0.0))

