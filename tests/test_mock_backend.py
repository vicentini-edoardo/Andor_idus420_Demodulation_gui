from __future__ import annotations

import numpy as np

from idus420_gui.camera.base import CameraConfig, TempStatus, TriggerMode
from idus420_gui.camera.mock import MockBackend


def test_mock_backend_kinetic_series_end_to_end() -> None:
    backend = MockBackend()
    backend.connect()
    assert backend.is_connected()
    assert backend.serial_number() == 420000
    assert backend.detector_size() == (1024, 255)
    backend.set_target_temperature(-60)
    backend.cooler_on()
    temp, status = backend.get_temperature()
    assert temp == -60
    assert status is TempStatus.STABILIZED
    backend.configure(CameraConfig(exposure_s=0.002))
    timings = backend.setup_kinetic(0.002, 16, TriggerMode.EXTERNAL)
    assert timings.exposure_s == 0.002
    backend.start()
    frames = []
    for _ in range(16):
        assert backend.wait_next_frame(100)
        frames.append(backend.get_oldest_frame())
    arr = np.stack(frames)
    assert arr.shape == (16, 1024)
    assert arr.dtype == np.uint16
    backend.setup_kinetic(0.002, 8, TriggerMode.EXTERNAL)
    backend.start()
    batch = backend.get_new_frames_batch()
    assert batch is not None
    assert batch.shape == (8, 1024)
    assert backend.get_all_frames(8).shape == (8, 1024)

