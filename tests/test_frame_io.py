"""Tests for rearm/timeout recovery and atomic HDF5 save."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from idus420_gui.camera.base import AcquisitionStatus, AcquisitionTimings, TriggerMode
from idus420_gui.camera.mock import MockBackend
from idus420_gui.workers._frame_io import (
    _MAX_REARM_ATTEMPTS,
    _read_pending_frames,
    _read_ready_frames,
    _rearm_message,
    _timeout_failure_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TimeoutThenOkBackend(MockBackend):
    """Camera backend that times out for the first N wait_next_frame calls."""

    def __init__(self, n_timeouts: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout_count = 0
        self._n_timeouts = n_timeouts

    def wait_next_frame(self, timeout_ms: int) -> bool:
        if self._timeout_count < self._n_timeouts:
            self._timeout_count += 1
            return False
        return super().wait_next_frame(timeout_ms)

    def get_new_frames_batch(self) -> np.ndarray | None:
        return super().get_new_frames_batch()

    def acquisition_diagnostics(self) -> str:
        return "simulated_timeout"


# ---------------------------------------------------------------------------
# _frame_io unit tests
# ---------------------------------------------------------------------------

def test_max_rearm_attempts_is_three() -> None:
    assert _MAX_REARM_ATTEMPTS == 3


def test_rearm_message_includes_attempt_info() -> None:
    msg = _rearm_message(5000, 1, 3, "status=DRV_ACQUIRING")
    assert "1/3" in msg
    assert "5.0 s" in msg
    assert "status=DRV_ACQUIRING" in msg


def test_timeout_failure_message_includes_diagnostics() -> None:
    msg = _timeout_failure_message(10000, "new_images=0")
    assert "10.0 s" in msg
    assert "new_images=0" in msg


def test_read_ready_frames_returns_correct_shape() -> None:
    backend = MockBackend()
    backend.connect()
    backend.setup_kinetic(0.001, 4, TriggerMode.EXTERNAL)
    backend.start()
    backend.wait_next_frame(1000)
    frames = _read_ready_frames(backend, max_frames=4)
    assert frames.ndim == 2
    assert frames.shape[0] <= 4
    assert frames.shape[1] == backend.frame_width()
    assert frames.dtype == np.uint16


def test_read_pending_frames_returns_none_when_empty() -> None:
    backend = MockBackend()
    backend.connect()
    result = _read_pending_frames(backend)
    assert result is None


def test_read_pending_frames_returns_frames_when_available() -> None:
    backend = MockBackend()
    backend.connect()
    backend.setup_kinetic(0.001, 8, TriggerMode.EXTERNAL)
    backend.start()
    frames = _read_pending_frames(backend, max_frames=8)
    assert frames is not None
    assert frames.shape[0] <= 8


# ---------------------------------------------------------------------------
# Rearm recovery integration test
# ---------------------------------------------------------------------------

pytest.importorskip("PyQt6")

from idus420_gui.motion.base import ScanGrid
from idus420_gui.motion.mock import MockStageBackend
from idus420_gui.workers.acquisition import DemodulationSettings
from idus420_gui.workers.scan import PointResult, ScanWorker


def _make_settings(n_block: int = 4) -> DemodulationSettings:
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


def test_scan_worker_handles_initial_timeouts(qtbot) -> None:
    """Worker must complete a scan even when the first few wait_next_frame calls time out."""
    camera = _TimeoutThenOkBackend(n_timeouts=2)
    camera.connect()
    stage = MockStageBackend()
    grid = ScanGrid(x_start_nm=0.0, y_start_nm=0.0, x_step_nm=500.0, y_step_nm=500.0,
                    nx=1, ny=1)
    settings = _make_settings(n_block=4)

    results: list[PointResult] = []
    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    worker.point_data_ready.connect(lambda _idx, r: results.append(r))

    with qtbot.waitSignal(worker.scan_finished, timeout=15_000):
        worker.start()

    worker.wait(3000)
    assert len(results) == 1
    assert results[0].frames.shape[0] == 4


# ---------------------------------------------------------------------------
# Atomic HDF5 save tests
# ---------------------------------------------------------------------------

from idus420_gui.io.save import save_h5, save_scan_h5
from idus420_gui.workers.scan import ScanResult


def test_save_h5_produces_valid_file() -> None:
    frames = np.zeros((8, 1024), dtype=np.uint16)
    roi = np.arange(8, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "run.h5"
        save_h5(path, frames, roi, [], {"test": True})
        assert path.exists()
        tmp = path.parent / (path.name + ".tmp")
        assert not tmp.exists()
        with h5py.File(path, "r") as f:
            assert "frames" in f
            assert "roi_timeseries" in f
            assert f["frames"].shape == (8, 1024)


def test_save_h5_cleans_up_tmp_on_error() -> None:
    """If h5py write fails, the .tmp file must be removed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad" / "run.h5"
        # Parent doesn't exist, but save_h5 creates it.  Force an error by
        # passing data that cannot be written.
        path.parent.mkdir(parents=True)
        tmp_path = path.parent / (path.name + ".tmp")
        with pytest.raises(Exception):
            save_h5(path, "not_an_array", np.array([]), [], {})  # type: ignore[arg-type]
        assert not tmp_path.exists()


def test_save_scan_h5_no_partial_file_on_success(qtbot) -> None:
    camera = MockBackend()
    camera.connect()
    stage = MockStageBackend()
    grid = ScanGrid(x_start_nm=0.0, y_start_nm=0.0, x_step_nm=500.0, y_step_nm=500.0,
                    nx=2, ny=1)
    settings = _make_settings(n_block=4)

    scan_results: list[ScanResult] = []
    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    worker.scan_finished.connect(lambda r: scan_results.append(r))

    with qtbot.waitSignal(worker.scan_finished, timeout=10_000):
        worker.start()
    worker.wait(3000)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scan.h5"
        save_scan_h5(path, scan_results[0], {"test": True})
        assert path.exists()
        tmp = path.parent / (path.name + ".tmp")
        assert not tmp.exists()
