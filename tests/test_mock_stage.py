from __future__ import annotations

import math

import numpy as np
import pytest

from idus420_gui.motion.base import ScanGrid, StageError
from idus420_gui.motion.mock import MockStageBackend


def test_connect_disconnect() -> None:
    stage = MockStageBackend()
    assert not stage.is_connected()
    stage.connect()
    assert stage.is_connected()
    stage.disconnect()
    assert not stage.is_connected()


def test_goto_and_readback() -> None:
    stage = MockStageBackend()
    stage.connect()
    stage.goto_xy_nm(1000.0, 2500.0)
    x, y, z = stage.read_xyz_nm()
    assert x == pytest.approx(1000.0)
    assert y == pytest.approx(2500.0)
    assert z == pytest.approx(0.0)


def test_read_sample_shapes() -> None:
    stage = MockStageBackend()
    stage.connect()
    stage.goto_xy_nm(500.0, 500.0)
    sample = stage.read_sample(t_s=0.1)
    assert sample.t_s == pytest.approx(0.1)
    assert len(sample.xyz_nm) == 3
    for arr in (sample.o_amp, sample.o_phase, sample.m_amp, sample.m_phase):
        assert arr.shape == (6,)
        assert arr.dtype == np.float64


def test_read_sample_decay_with_distance() -> None:
    stage = MockStageBackend(seed=0)
    stage.connect()
    stage.goto_xy_nm(0.0, 0.0)
    s_origin = stage.read_sample(t_s=0.0)

    stage.goto_xy_nm(50_000.0, 0.0)
    s_far = stage.read_sample(t_s=0.1)

    assert s_origin.o_amp[0] > s_far.o_amp[0]


def test_error_when_not_connected() -> None:
    stage = MockStageBackend()
    with pytest.raises(StageError):
        stage.goto_xy_nm(0, 0)
    with pytest.raises(StageError):
        stage.read_xyz_nm()
    with pytest.raises(StageError):
        stage.read_sample(t_s=0.0)


def test_scan_grid_points_raster_lr() -> None:
    grid = ScanGrid(
        x_start_nm=0.0, y_start_nm=0.0,
        x_step_nm=100.0, y_step_nm=100.0,
        nx=3, ny=2,
        order="raster_lr",
    )
    pts = list(grid.points())
    assert len(pts) == 6
    # All rows left-to-right.
    ixs = [p.ix for p in pts]
    assert ixs == [0, 1, 2, 0, 1, 2]


def test_scan_grid_points_snake() -> None:
    grid = ScanGrid(
        x_start_nm=0.0, y_start_nm=0.0,
        x_step_nm=100.0, y_step_nm=100.0,
        nx=3, ny=2,
        order="snake",
    )
    pts = list(grid.points())
    # Row 0 left-to-right, row 1 right-to-left.
    row0 = [p.ix for p in pts if p.iy == 0]
    row1 = [p.ix for p in pts if p.iy == 1]
    assert row0 == [0, 1, 2]
    assert row1 == [2, 1, 0]


def test_scan_grid_coordinates() -> None:
    grid = ScanGrid(
        x_start_nm=1000.0, y_start_nm=2000.0,
        x_step_nm=500.0, y_step_nm=300.0,
        nx=2, ny=2,
        order="raster_lr",
    )
    pts = list(grid.points())
    assert pts[0].x_nm == pytest.approx(1000.0)
    assert pts[0].y_nm == pytest.approx(2000.0)
    assert pts[1].x_nm == pytest.approx(1500.0)
    assert pts[1].y_nm == pytest.approx(2000.0)
    assert pts[2].y_nm == pytest.approx(2300.0)
