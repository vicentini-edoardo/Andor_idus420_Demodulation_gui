"""Save captured frames and demodulation products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from idus420_gui.processing.demodulation import DemodResult


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
    """Save run products in HDF5 format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_u16 = np.asarray(frames, dtype=np.uint16)
    meta = dict(metadata)
    meta["frames_sha256"] = _frames_sha256(frames_u16)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frames", data=frames_u16, compression="gzip")
        h5.create_dataset("roi_timeseries", data=np.asarray(roi_timeseries, dtype=np.float64))
        h5.create_dataset("demod_results", data=_demod_structured(demod_results))
        h5.attrs["metadata"] = json.dumps(meta, default=str)


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
