from __future__ import annotations

import numpy as np
import pytest

from idus420_gui.spectrograph.andor import AndorShamrockBackend
from idus420_gui.spectrograph.base import GratingInfo, SpectrographError
from idus420_gui.spectrograph.mock import MockSpectrograph


def test_mock_spectrograph_grating_and_wavelength_round_trip() -> None:
    spec = MockSpectrograph()
    spec.connect()
    assert spec.is_connected()
    gratings = spec.list_gratings()
    assert [g.index for g in gratings] == [1, 2, 3]
    assert spec.current_grating() == 1

    spec.set_grating(2)
    assert spec.current_grating() == 2
    spec.set_wavelength(650.0)
    assert spec.get_wavelength() == 650.0


def test_mock_spectrograph_rejects_out_of_range_wavelength() -> None:
    spec = MockSpectrograph()
    spec.connect()
    spec.set_grating(3)
    lo, hi = spec.wavelength_limits(3)
    with pytest.raises(SpectrographError):
        spec.set_wavelength(hi + 100.0)


def test_mock_spectrograph_clamps_wavelength_when_switching_grating() -> None:
    spec = MockSpectrograph()
    spec.connect()
    spec.set_wavelength(1300.0)  # valid on grating 1 (up to 1400)
    spec.set_grating(3)  # grating 3 only reaches 900
    lo, hi = spec.wavelength_limits(3)
    assert spec.get_wavelength() <= hi


def test_mock_calibration_centres_on_wavelength_and_scales_with_grating() -> None:
    spec = MockSpectrograph()
    spec.connect()
    spec.set_grating(1)
    spec.set_wavelength(500.0)
    cal = spec.get_calibration(1024)
    assert cal.shape == (1024,)
    # Centre wavelength sits at the array midpoint.
    assert cal[511] == pytest.approx(500.0, abs=0.3)
    span_150 = cal[-1] - cal[0]

    # A denser grating disperses less, so its span across the detector shrinks.
    spec.set_grating(2)
    spec.set_wavelength(500.0)
    span_600 = spec.get_calibration(1024)[-1] - spec.get_calibration(1024)[0]
    assert abs(span_600) < abs(span_150)


def test_mock_calibration_tracks_pixel_count() -> None:
    spec = MockSpectrograph()
    spec.connect()
    spec.set_pixel_geometry(26.0, 512)
    assert spec.get_calibration(512).shape == (512,)


def test_mock_requires_connection() -> None:
    spec = MockSpectrograph()
    with pytest.raises(SpectrographError):
        spec.list_gratings()


def test_andor_parse_grating_info_extracts_fields() -> None:
    lines, blaze, home, offset = AndorShamrockBackend._parse_grating_info(
        (20202, 150.0, "800nm", 12345, 6789)
    )
    assert lines == 150.0
    assert blaze == "800nm"
    assert home == 12345
    assert offset == 6789


def test_andor_get_calibration_parses_array_payload() -> None:
    backend = AndorShamrockBackend.__new__(AndorShamrockBackend)
    backend._success = 20202
    backend._device = 0

    class _Sdk:
        def GetCalibration(self, _device, n):  # noqa: N802 - SDK naming
            return 20202, np.linspace(400.0, 500.0, n)

    backend._sdk = _Sdk()
    cal = backend.get_calibration(8)
    assert cal.shape == (8,)
    assert cal[0] == pytest.approx(400.0)


def test_andor_check_raises_on_error_code() -> None:
    backend = AndorShamrockBackend.__new__(AndorShamrockBackend)
    backend._success = 20202
    with pytest.raises(SpectrographError):
        backend._check(20201, "SetGrating")


def test_grating_info_label() -> None:
    assert GratingInfo(index=1, lines_per_mm=150.0, blaze="800nm").label() == (
        "1: 150 l/mm, blaze 800nm"
    )
    assert GratingInfo(index=2, lines_per_mm=600.0).label() == "2: 600 l/mm"
