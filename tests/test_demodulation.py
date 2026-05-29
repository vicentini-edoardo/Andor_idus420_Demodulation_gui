from __future__ import annotations

import numpy as np

from idus420_gui.processing.demodulation import demodulate


def test_demodulate_recovers_synthetic_tone() -> None:
    sample_rate = 500.0
    n = 2048
    frequency = 152 * sample_rate / n
    amplitude = 12.0
    rng = np.random.default_rng(42)
    t = np.arange(n) / sample_rate
    y = 100.0 + amplitude * np.sin(2.0 * np.pi * frequency * t)
    y += rng.normal(0.0, 0.4, size=n)

    result = demodulate(y, sample_rate, f_expected=36.5, f_search_halfwidth=3.0)

    assert abs(result.peak_frequency - frequency) < sample_rate / n
    assert abs(result.peak_amplitude - amplitude) / amplitude < 0.05
    assert result.snr > 10


def test_demodulate_rejects_out_of_range_search_band() -> None:
    y = np.ones(32)
    try:
        demodulate(y, 100.0, f_expected=1000.0, f_search_halfwidth=1.0)
    except ValueError as exc:
        assert "Search band" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_demodulate_snaps_to_nearest_bin_for_narrow_band() -> None:
    # A search half-width smaller than one FFT bin is in range but contains no
    # bins; demodulate should snap to the nearest bin instead of failing.
    sample_rate = 500.0
    n = 256
    bin_width = sample_rate / n  # ~1.95 Hz
    t = np.arange(n) / sample_rate
    y = 100.0 + 5.0 * np.sin(2.0 * np.pi * 4 * bin_width * t)

    result = demodulate(
        y, sample_rate, f_expected=4 * bin_width, f_search_halfwidth=bin_width / 10.0
    )

    assert abs(result.peak_frequency - 4 * bin_width) <= bin_width
