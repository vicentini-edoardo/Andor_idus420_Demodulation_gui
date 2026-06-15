from __future__ import annotations

import json

import numpy as np
import pytest

from idus420_gui.io.save import save_h5, save_npz, save_txt
from idus420_gui.processing.demodulation import DemodResult


def _demods() -> list[DemodResult]:
    f_axis = np.array([0.0, 1.0, 2.0])
    return [
        DemodResult(
            peak_frequency=37.0,
            peak_amplitude=12.5,
            f_axis=f_axis,
            spectrum=np.array([0.1, 0.2, 0.3]),
            snr=4.0,
        )
    ]


def test_npz_round_trip_with_frame_times(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames = np.arange(12, dtype=np.uint16).reshape(3, 4)
    roi = np.array([1.0, 2.0, 3.0])
    times = np.array([0.0, 0.5, 1.0])
    path = tmp_path / "run.npz"
    save_npz(path, frames, roi, _demods(), {"k": "v"}, times)

    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data["frames"], frames)
        assert np.allclose(data["roi_timeseries"], roi)
        assert np.allclose(data["frame_times_s"], times)
        meta = json.loads(str(data["metadata"]))
        assert meta["k"] == "v"
        assert "frames_sha256" in meta


def test_npz_without_frame_times_writes_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames = np.zeros((2, 3), dtype=np.uint16)
    path = tmp_path / "run.npz"
    save_npz(path, frames, np.array([0.0, 0.0]), _demods(), {})
    with np.load(path) as data:
        assert data["frame_times_s"].shape == (0,)


def test_npz_save_is_atomic_on_failure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # demod_results that cannot be serialised into the structured dtype forces
    # an error mid-save; the target must not be left as a partial file and no
    # .tmp artifact should remain.
    path = tmp_path / "run.npz"
    bad = ["not-a-demod-result"]
    with pytest.raises((AttributeError, TypeError)):
        save_npz(path, np.zeros((1, 2), dtype=np.uint16), np.array([0.0]), bad, {})  # type: ignore[arg-type]
    assert not path.exists()
    assert not (tmp_path / "run.npz.tmp.npz").exists()


def test_h5_round_trip_with_frame_times(tmp_path) -> None:  # type: ignore[no-untyped-def]
    h5py = pytest.importorskip("h5py")
    frames = np.arange(8, dtype=np.uint16).reshape(2, 4)
    roi = np.array([5.0, 6.0])
    times = np.array([0.0, 0.25])
    path = tmp_path / "run.h5"
    save_h5(path, frames, roi, _demods(), {}, times)
    with h5py.File(path, "r") as h5:
        assert np.array_equal(h5["frames"][...], frames)
        assert np.allclose(h5["frame_times_s"][...], times)


def test_txt_includes_time_column_when_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames = np.zeros((3, 2), dtype=np.uint16)
    roi = np.array([1.0, 2.0, 3.0])
    times = np.array([0.0, 0.5, 1.0])
    path = tmp_path / "run.txt"
    save_txt(path, frames, roi, _demods(), {}, times)
    text = path.read_text()
    assert "frame_index\ttime_s\troi_value" in text


def test_txt_omits_time_column_when_absent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames = np.zeros((2, 2), dtype=np.uint16)
    roi = np.array([1.0, 2.0])
    path = tmp_path / "run.txt"
    save_txt(path, frames, roi, _demods(), {})
    text = path.read_text()
    assert "frame_index\troi_value" in text
    assert "time_s" not in text
