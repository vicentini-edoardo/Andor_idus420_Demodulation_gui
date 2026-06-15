from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from idus420_gui.camera.mock import MockBackend, MockSpectrumConfig
from idus420_gui.processing.roi import integrate_roi
from idus420_gui.workers.acquisition import (
    AcquisitionWorker,
    DemodulationSettings,
    DemodulationWorker,
    LiveSpectrumWorker,
)


def _settings(n_block: int = 8) -> DemodulationSettings:
    return DemodulationSettings(
        exposure_s=0.001,
        trigger_frequency_hz=500.0,
        pixel_start=480,
        pixel_end=560,
        roi_method="sum",
        n_block=n_block,
        f_expected=37.0,
        f_search_halfwidth=5.0,
        window="hann",
    )


def test_demodulation_worker_one_block(qtbot) -> None:  # type: ignore[no-untyped-def]
    backend = MockBackend(
        spectrum=MockSpectrumConfig(sample_rate_hz=500.0, true_modulation_hz=37.0)
    )
    backend.connect()
    backend.cooler_on()
    settings = DemodulationSettings(
        exposure_s=0.001,
        trigger_frequency_hz=500.0,
        pixel_start=480,
        pixel_end=560,
        roi_method="sum",
        n_block=256,
        f_expected=37.0,
        f_search_halfwidth=5.0,
        window="hann",
    )
    worker = DemodulationWorker(backend, settings, continuous=False)
    with qtbot.waitSignal(worker.demod_result, timeout=3000) as blocker:
        worker.start()
    result = blocker.args[0]
    worker.wait(3000)
    assert abs(result.peak_frequency - 37.0) < 3.0


def test_demodulation_worker_continuous_rolling(qtbot) -> None:  # type: ignore[no-untyped-def]
    backend = MockBackend(
        spectrum=MockSpectrumConfig(sample_rate_hz=500.0, true_modulation_hz=37.0)
    )
    backend.connect()
    backend.cooler_on()
    settings = DemodulationSettings(
        exposure_s=0.001,
        trigger_frequency_hz=500.0,
        pixel_start=480,
        pixel_end=560,
        roi_method="sum",
        n_block=64,
        f_expected=37.0,
        f_search_halfwidth=5.0,
        window="hann",
    )
    worker = DemodulationWorker(backend, settings, continuous=True, hop_frames=8)

    windows: list[np.ndarray] = []
    results: list[object] = []
    worker.block_complete.connect(lambda ts: windows.append(np.asarray(ts).copy()))
    worker.demod_result.connect(lambda r: results.append(r))

    with qtbot.waitSignal(worker.worker_finished, timeout=5000):
        worker.start()
        qtbot.waitUntil(lambda: len(results) >= 4, timeout=5000)
        worker.stop()

    worker.wait(3000)

    # The rolling window keeps streaming demodulated results, not just one block.
    assert len(results) >= 4
    # The window fills up over time and then caps at n_block (rolls, never exceeds).
    assert all(w.size <= settings.n_block for w in windows)
    assert max(w.size for w in windows) == settings.n_block
    assert windows[0].size < windows[-1].size or windows[0].size == settings.n_block


def test_live_spectrum_worker_emits_roi_samples(qtbot) -> None:  # type: ignore[no-untyped-def]
    backend = MockBackend(
        spectrum=MockSpectrumConfig(sample_rate_hz=500.0, true_modulation_hz=37.0)
    )
    backend.connect()
    backend.cooler_on()

    worker = LiveSpectrumWorker(
        backend,
        exposure_s=0.001,
        trigger_frequency_hz=500.0,
        pixel_start=480,
        pixel_end=560,
        burst_frames=4,
    )

    captured: list[tuple[float, float]] = []
    frames: list[np.ndarray] = []
    worker.roi_sample.connect(
        lambda elapsed_s, roi_sum: captured.append((elapsed_s, roi_sum))
    )
    worker.frame_acquired.connect(lambda frame: frames.append(np.asarray(frame).copy()))

    with qtbot.waitSignal(worker.worker_finished, timeout=3000):
        worker.start()
        qtbot.waitUntil(lambda: len(captured) >= 4 and len(frames) >= 4, timeout=3000)
        worker.stop()

    worker.wait(3000)
    assert len(captured) >= 4
    assert len(frames) >= 4

    for idx, ((elapsed_s, roi_sum), frame) in enumerate(
        zip(captured[:4], frames[:4], strict=False)
    ):
        expected_sum = float(integrate_roi(frame.reshape(1, -1), 480, 560, "sum")[0])
        assert elapsed_s == pytest.approx(idx / 500.0)
        assert roi_sum == pytest.approx(expected_sum)


def test_acquisition_worker_roi_timeseries_matches_frames(qtbot) -> None:  # type: ignore[no-untyped-def]
    backend = MockBackend(
        spectrum=MockSpectrumConfig(sample_rate_hz=500.0, true_modulation_hz=37.0)
    )
    backend.connect()
    backend.cooler_on()

    worker = AcquisitionWorker(backend, _settings(n_block=8), total_frames=8)

    with qtbot.waitSignal(worker.run_finished, timeout=3000) as blocker:
        worker.start()
    worker.wait(3000)

    frames, processed = blocker.args
    roi_ts = processed["roi_timeseries"]
    assert frames.shape[0] == 8
    assert roi_ts.shape == (8,)
    expected = integrate_roi(frames, 480, 560, "sum")
    assert np.allclose(roi_ts, expected)
