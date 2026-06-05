"""Save captured frames and demodulation products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from idus420_gui.processing.demodulation import DemodResult

if TYPE_CHECKING:
    from idus420_gui.workers.scan import ScanResult


def _frames_sha256(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames)).hexdigest()


def save_run(
    path: str | Path,
    frames: np.ndarray,
    roi_timeseries: np.ndarray,
    demod_results: list[DemodResult],
    metadata: dict[str, Any],
) -> None:
    """Save by extension, supporting `.npz` and `.h5`."""
    target = Path(path)
    if target.suffix == ".npz":
        save_npz(target, frames, roi_timeseries, demod_results, metadata)
        return
    if target.suffix in {".h5", ".hdf5"}:
        save_h5(target, frames, roi_timeseries, demod_results, metadata)
        return
    raise ValueError("Output path must end with .npz, .h5, or .hdf5.")


def save_npz(
    path: str | Path,
    frames: np.ndarray,
    roi_timeseries: np.ndarray,
    demod_results: list[DemodResult],
    metadata: dict[str, Any],
) -> None:
    """Save run products in compressed NumPy format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_u16 = np.asarray(frames, dtype=np.uint16)
    meta = dict(metadata)
    meta["frames_sha256"] = _frames_sha256(frames_u16)
    np.savez_compressed(
        path,
        frames=frames_u16,
        roi_timeseries=np.asarray(roi_timeseries, dtype=np.float64),
        demod_results=_demod_structured(demod_results),
        metadata=json.dumps(meta, default=str),
    )


def save_h5(
    path: str | Path,
    frames: np.ndarray,
    roi_timeseries: np.ndarray,
    demod_results: list[DemodResult],
    metadata: dict[str, Any],
) -> None:
    """Save run products in HDF5 format (atomic: writes to .tmp then renames)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    frames_u16 = np.asarray(frames, dtype=np.uint16)
    meta = dict(metadata)
    meta["frames_sha256"] = _frames_sha256(frames_u16)
    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("frames", data=frames_u16, compression="gzip")
            h5.create_dataset("roi_timeseries", data=np.asarray(roi_timeseries, dtype=np.float64))
            h5.create_dataset("demod_results", data=_demod_structured(demod_results))
            h5.attrs["metadata"] = json.dumps(meta, default=str)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_txt(
    path: str | Path,
    frames: np.ndarray,
    roi_timeseries: np.ndarray,
    demod_results: list[DemodResult],
    metadata: dict[str, Any],
) -> None:
    """Tab-separated text export with ``#``-prefixed metadata header.

    Sections:
    1. ``# key: value`` metadata lines (one per entry, values JSON-encoded).
    2. ``# roi_timeseries: frame_index<TAB>roi_value`` + data rows.
    3. ``# demod_results: peak_frequency<TAB>peak_amplitude<TAB>snr`` + data rows.

    Raw frames are not embedded — use .npz/.h5 for full frame data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_u16 = np.asarray(frames, dtype=np.uint16)
    meta = dict(metadata)
    meta["frames_sha256"] = _frames_sha256(frames_u16)

    roi_arr = np.asarray(roi_timeseries, dtype=np.float64)
    n_roi = roi_arr.shape[0]
    indices = np.arange(n_roi, dtype=np.int64)

    with path.open("w", encoding="utf-8") as f:
        f.write("# idus420_gui export\n")
        for k, v in meta.items():
            f.write(f"# {k}: {json.dumps(v, default=str)}\n")
        f.write("# roi_timeseries: frame_index\troi_value\n")
        np.savetxt(f, np.column_stack([indices, roi_arr]), delimiter="\t", fmt=["%d", "%.10g"])
        f.write("# demod_results: peak_frequency\tpeak_amplitude\tsnr\n")
        if demod_results:
            demod_arr = np.array(
                [(r.peak_frequency, r.peak_amplitude, r.snr) for r in demod_results],
                dtype=np.float64,
            )
            np.savetxt(f, demod_arr, delimiter="\t", fmt="%.10g")


def save_scan_h5(
    path: str | Path,
    scan: "ScanResult",
    metadata: dict[str, Any],
) -> None:
    """Save a full 2-D raster scan to a single HDF5 file.

    Layout::

        /                            attrs: metadata JSON, grid params
          scan/
            coords_xy_nm    (N,2)   planned XY in scan order
            coords_xyz_nm   (N,3)   actual motor readback
            point_index_grid (ny,nx) linear scan index at each grid cell
          points/
            point_000000/
              frames          (n_frames, frame_width) uint16 gzip
              roi_timeseries  (n_frames,) float64
              demod           structured dtype
              snom_t_s        (n_samples,) float64
              snom_xyz_nm     (n_samples,3)
              snom_o_amp      (n_samples,6)
              snom_o_phase    (n_samples,6)
              snom_m_amp      (n_samples,6)
              snom_m_phase    (n_samples,6)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = scan.grid
    n_points = len(scan.point_results)

    # Build planned coord arrays.
    coords_xy = np.array(
        [(pr.point.x_nm, pr.point.y_nm) for pr in scan.point_results], dtype=np.float64
    )
    coords_xyz = np.array(
        [list(pr.actual_xyz_nm) for pr in scan.point_results], dtype=np.float64
    )

    # point_index_grid: ny × nx, filled with the scan-order index.
    index_grid = np.full((grid.ny, grid.nx), -1, dtype=np.int64)
    for scan_idx, pr in enumerate(scan.point_results):
        index_grid[pr.point.iy, pr.point.ix] = scan_idx

    meta = dict(metadata)
    meta["grid"] = {
        "x_start_nm": grid.x_start_nm,
        "y_start_nm": grid.y_start_nm,
        "x_step_nm": grid.x_step_nm,
        "y_step_nm": grid.y_step_nm,
        "nx": grid.nx,
        "ny": grid.ny,
        "order": grid.order,
    }

    tmp_path = path.parent / (path.name + ".tmp")
    try:
        with h5py.File(tmp_path, "w") as h5:
            h5.attrs["metadata"] = json.dumps(meta, default=str)

            sg = h5.create_group("scan")
            sg.create_dataset("coords_xy_nm", data=coords_xy)
            sg.create_dataset("coords_xyz_nm", data=coords_xyz)
            sg.create_dataset("point_index_grid", data=index_grid)

            pg = h5.create_group("points")
            for scan_idx, pr in enumerate(scan.point_results):
                grp = pg.create_group(f"point_{scan_idx:06d}")
                grp.attrs["ix"] = pr.point.ix
                grp.attrs["iy"] = pr.point.iy
                grp.attrs["x_nm"] = pr.point.x_nm
                grp.attrs["y_nm"] = pr.point.y_nm
                grp.attrs["actual_x_nm"] = pr.actual_xyz_nm[0]
                grp.attrs["actual_y_nm"] = pr.actual_xyz_nm[1]
                grp.attrs["actual_z_nm"] = pr.actual_xyz_nm[2]

                frames_u16 = np.asarray(pr.frames, dtype=np.uint16)
                grp.create_dataset("frames", data=frames_u16, compression="gzip")
                grp.attrs["frames_sha256"] = _frames_sha256(frames_u16)
                grp.create_dataset(
                    "roi_timeseries",
                    data=np.asarray(pr.roi_timeseries, dtype=np.float64),
                )
                grp.create_dataset("demod", data=_demod_structured(pr.demod_results))

                if pr.snom_samples:
                    t_arr = np.array([s.t_s for s in pr.snom_samples], dtype=np.float64)
                    xyz_arr = np.array([list(s.xyz_nm) for s in pr.snom_samples], dtype=np.float64)
                    o_amp_arr = np.stack([s.o_amp for s in pr.snom_samples], axis=0)
                    o_ph_arr = np.stack([s.o_phase for s in pr.snom_samples], axis=0)
                    m_amp_arr = np.stack([s.m_amp for s in pr.snom_samples], axis=0)
                    m_ph_arr = np.stack([s.m_phase for s in pr.snom_samples], axis=0)
                    grp.create_dataset("snom_t_s", data=t_arr)
                    grp.create_dataset("snom_xyz_nm", data=xyz_arr)
                    grp.create_dataset("snom_o_amp", data=o_amp_arr)
                    grp.create_dataset("snom_o_phase", data=o_ph_arr)
                    grp.create_dataset("snom_m_amp", data=m_amp_arr)
                    grp.create_dataset("snom_m_phase", data=m_ph_arr)
                else:
                    for dname in ("snom_t_s", "snom_xyz_nm", "snom_o_amp", "snom_o_phase",
                                   "snom_m_amp", "snom_m_phase"):
                        grp.create_dataset(dname, data=np.array([]))
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _demod_structured(results: list[DemodResult]) -> np.ndarray:
    dtype = np.dtype(
        [
            ("peak_frequency", "f8"),
            ("peak_amplitude", "f8"),
            ("snr", "f8"),
        ]
    )
    out = np.zeros(len(results), dtype=dtype)
    for idx, result in enumerate(results):
        out[idx] = (result.peak_frequency, result.peak_amplitude, result.snr)
    return out
