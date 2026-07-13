from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from idus420_gui.camera.mock import MockBackend
from idus420_gui.motion.base import ScanGrid
from idus420_gui.motion.mock import MockStageBackend
from idus420_gui.workers.acquisition import DemodulationSettings
from idus420_gui.workers.scan import PointResult, ScanResult, ScanWorker


def _make_settings(n_block: int = 16) -> DemodulationSettings:
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


def _make_grid(nx: int = 3, ny: int = 2, order: str = "raster_lr") -> ScanGrid:
    return ScanGrid(
        x_start_nm=0.0, y_start_nm=0.0,
        x_step_nm=1000.0, y_step_nm=1000.0,
        nx=nx, ny=ny,
        order=order,
    )


def test_scan_visits_all_points(qtbot) -> None:  # type: ignore[no-untyped-def]
    camera = MockBackend()
    camera.connect()
    stage = MockStageBackend()
    grid = _make_grid(nx=2, ny=2)
    settings = _make_settings(n_block=8)

    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    results: list[PointResult] = []
    worker.point_data_ready.connect(lambda idx, r: results.append(r))

    with qtbot.waitSignal(worker.scan_finished, timeout=10_000):
        worker.start()

    worker.wait(3000)
    assert len(results) == 4
    visited = {(r.point.ix, r.point.iy) for r in results}
    assert visited == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_scan_raster_lr_order(qtbot) -> None:  # type: ignore[no-untyped-def]
    camera = MockBackend()
    camera.connect()
    stage = MockStageBackend()
    grid = _make_grid(nx=3, ny=2, order="raster_lr")
    settings = _make_settings(n_block=8)

    points_in_order: list[tuple[int, int]] = []
    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    worker.point_started.connect(lambda p: points_in_order.append((p.ix, p.iy)))

    with qtbot.waitSignal(worker.scan_finished, timeout=15_000):
        worker.start()

    worker.wait(3000)
    assert points_in_order == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_scan_snake_order(qtbot) -> None:  # type: ignore[no-untyped-def]
    camera = MockBackend()
    camera.connect()
    stage = MockStageBackend()
    grid = _make_grid(nx=3, ny=2, order="snake")
    settings = _make_settings(n_block=8)

    points_in_order: list[tuple[int, int]] = []
    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    worker.point_started.connect(lambda p: points_in_order.append((p.ix, p.iy)))

    with qtbot.waitSignal(worker.scan_finished, timeout=15_000):
        worker.start()

    worker.wait(3000)
    assert points_in_order == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]


def test_scan_frames_shape(qtbot) -> None:  # type: ignore[no-untyped-def]
    camera = MockBackend()
    camera.connect()
    stage = MockStageBackend()
    grid = _make_grid(nx=2, ny=1)
    settings = _make_settings(n_block=16)

    scan_results: list[ScanResult] = []
    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    worker.scan_finished.connect(lambda r: scan_results.append(r))

    with qtbot.waitSignal(worker.scan_finished, timeout=10_000):
        worker.start()

    worker.wait(3000)
    assert len(scan_results) == 1
    sr = scan_results[0]
    for pr in sr.point_results:
        assert pr.frames.shape == (16, 1024)
        assert pr.frames.dtype == np.uint16
        assert pr.roi_timeseries.shape == (16,)
        assert len(pr.snom_samples) == 16


def test_scan_snom_samples_have_correct_shapes(qtbot) -> None:  # type: ignore[no-untyped-def]
    camera = MockBackend()
    camera.connect()
    stage = MockStageBackend()
    grid = _make_grid(nx=2, ny=1)
    settings = _make_settings(n_block=8)

    scan_results: list[ScanResult] = []
    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    worker.scan_finished.connect(lambda r: scan_results.append(r))

    with qtbot.waitSignal(worker.scan_finished, timeout=10_000):
        worker.start()

    worker.wait(3000)
    sr = scan_results[0]
    for pr in sr.point_results:
        for s in pr.snom_samples:
            assert s.o_amp.shape == (6,)
            assert s.m_amp.shape == (6,)
            assert s.o_phase.shape == (6,)
            assert s.m_phase.shape == (6,)


def test_scan_save_h5_groups(qtbot) -> None:  # type: ignore[no-untyped-def]
    from idus420_gui.io.save import save_scan_h5

    camera = MockBackend()
    camera.connect()
    stage = MockStageBackend()
    grid = ScanGrid(
        x_start_nm=0.0, y_start_nm=0.0,
        x_step_nm=1000.0, y_step_nm=1000.0,
        nx=2, ny=2,
        angle_deg=12.5,
    )
    settings = _make_settings(n_block=8)

    scan_results: list[ScanResult] = []
    worker = ScanWorker(camera, stage, grid, settings, metadata={"snom_host": "mock"})
    worker.scan_finished.connect(lambda r: scan_results.append(r))

    with qtbot.waitSignal(worker.scan_finished, timeout=10_000):
        worker.start()

    worker.wait(3000)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_scan.h5"
        save_scan_h5(path, scan_results[0], {"test": True})
        assert path.exists()
        with h5py.File(path, "r") as f:
            metadata = json.loads(f.attrs["metadata"])
            assert metadata["grid"]["angle_deg"] == pytest.approx(12.5)
            assert "scan" in f
            assert "points" in f
            assert f["scan/coords_xy_nm"].shape == (4, 2)
            assert f["scan/coords_xyz_nm"].shape == (4, 3)
            assert f["scan/point_index_grid"].shape == (2, 2)
            for i in range(4):
                grp = f[f"points/point_{i:06d}"]
                assert "frames" in grp
                assert "roi_timeseries" in grp
                assert "snom_o_amp" in grp
