from __future__ import annotations

import numpy as np
import pytest

from idus420_gui.io.save import save_run


def _sample_inputs():
    frames = np.arange(8, dtype=np.uint16).reshape(2, 4)
    roi = np.array([1.0, 2.0], dtype=np.float64)
    metadata = {"software": "idus420_gui"}
    return frames, roi, [], metadata


def test_save_run_dispatches_npz(tmp_path) -> None:
    frames, roi, results, metadata = _sample_inputs()
    path = tmp_path / "run.npz"
    save_run(path, frames, roi, results, metadata)
    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data["frames"], frames)
        assert np.array_equal(data["roi_timeseries"], roi)


def test_save_run_dispatches_txt(tmp_path) -> None:
    frames, roi, results, metadata = _sample_inputs()
    path = tmp_path / "run.txt"
    save_run(path, frames, roi, results, metadata)
    text = path.read_text(encoding="utf-8")
    assert "# roi_timeseries:" in text
    assert "frames_sha256" in text


def test_save_run_rejects_unknown_extension(tmp_path) -> None:
    frames, roi, results, metadata = _sample_inputs()
    with pytest.raises(ValueError, match=r"\.npz, \.h5, \.hdf5, or \.txt"):
        save_run(tmp_path / "run.csv", frames, roi, results, metadata)
