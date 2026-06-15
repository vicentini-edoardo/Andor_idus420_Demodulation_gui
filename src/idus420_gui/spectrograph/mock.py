"""Synthetic spectrograph backend for development and tests without hardware."""

from __future__ import annotations

import numpy as np

from idus420_gui.spectrograph.base import (
    GratingInfo,
    SpectrographBackend,
    SpectrographError,
)

# A plausible turret of three gratings, mirroring a common Shamrock/Kymera setup.
_MOCK_GRATINGS: tuple[GratingInfo, ...] = (
    GratingInfo(index=1, lines_per_mm=150.0, blaze="800nm"),
    GratingInfo(index=2, lines_per_mm=600.0, blaze="500nm"),
    GratingInfo(index=3, lines_per_mm=1200.0, blaze="500nm"),
)

# Per-grating (min, max) centre-wavelength limits in nm.
_MOCK_LIMITS: dict[int, tuple[float, float]] = {
    1: (0.0, 1400.0),
    2: (0.0, 1100.0),
    3: (0.0, 900.0),
}


class MockSpectrograph(SpectrographBackend):
    """Spectrograph backend that emulates grating selection and calibration.

    The synthetic dispersion scales inversely with the groove density so that
    switching gratings produces a visibly different wavelength axis, which is
    handy for exercising the GUI without hardware.
    """

    def __init__(self) -> None:
        self._connected = False
        self._grating = 1
        self._wavelength = 500.0
        self._pixel_width_um = 26.0
        self._number_pixels = 1024

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def serial_number(self) -> str:
        self._require_connected()
        return "MOCK-SR303i"

    def list_gratings(self) -> list[GratingInfo]:
        self._require_connected()
        return list(_MOCK_GRATINGS)

    def current_grating(self) -> int:
        self._require_connected()
        return self._grating

    def set_grating(self, index: int) -> None:
        self._require_connected()
        if index not in _MOCK_LIMITS:
            raise SpectrographError(f"Grating {index} is not installed.")
        self._grating = int(index)
        # Clamp the standing wavelength into the new grating's range.
        lo, hi = self.wavelength_limits(self._grating)
        self._wavelength = float(min(max(self._wavelength, lo), hi))

    def get_wavelength(self) -> float:
        self._require_connected()
        return self._wavelength

    def set_wavelength(self, nm: float) -> None:
        self._require_connected()
        lo, hi = self.wavelength_limits(self._grating)
        if not lo <= nm <= hi:
            raise SpectrographError(
                f"Wavelength {nm:g} nm is outside {lo:g}..{hi:g} nm for grating {self._grating}."
            )
        self._wavelength = float(nm)

    def wavelength_limits(self, grating_index: int) -> tuple[float, float]:
        self._require_connected()
        try:
            return _MOCK_LIMITS[int(grating_index)]
        except KeyError as exc:
            raise SpectrographError(f"Grating {grating_index} is not installed.") from exc

    def set_pixel_geometry(self, pixel_width_um: float, number_pixels: int) -> None:
        self._require_connected()
        if pixel_width_um <= 0:
            raise SpectrographError("Pixel width must be positive.")
        if number_pixels < 1:
            raise SpectrographError("Number of pixels must be >= 1.")
        self._pixel_width_um = float(pixel_width_um)
        self._number_pixels = int(number_pixels)

    def get_calibration(self, number_pixels: int) -> np.ndarray:
        self._require_connected()
        n = int(number_pixels)
        if n < 1:
            raise SpectrographError("Number of pixels must be >= 1.")
        nm_per_px = self._dispersion_nm_per_px()
        offsets = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * nm_per_px
        return self._wavelength + offsets

    def sdk_version(self) -> str:
        return "mock-spectrograph-0.1"

    def _dispersion_nm_per_px(self) -> float:
        # Reference dispersion of ~0.2 nm/px at 150 l/mm with a 26 µm pixel,
        # scaled inversely with groove density and linearly with pixel width.
        grating = next(g for g in _MOCK_GRATINGS if g.index == self._grating)
        base = 0.2 * (150.0 / grating.lines_per_mm)
        return base * (self._pixel_width_um / 26.0)

    def _require_connected(self) -> None:
        if not self._connected:
            raise SpectrographError("Spectrograph backend is not connected.")
