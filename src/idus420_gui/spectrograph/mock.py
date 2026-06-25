"""Mock spectrograph backend for testing and demo without hardware."""

from __future__ import annotations

import numpy as np

from .base import GratingInfo, SpectroBackend, SpectroError

_GRATINGS = [
    GratingInfo(1, 150.0, "VIS"),
    GratingInfo(2, 600.0, "500"),
    GratingInfo(3, 1200.0, "750"),
]

# Approximate nm/pixel dispersion for each grating at the iDus 420 focal plane.
_DISPERSION: dict[int, float] = {1: 0.50, 2: 0.13, 3: 0.065}
_LIMITS: dict[int, tuple[float, float]] = {
    1: (200.0, 1100.0),
    2: (200.0, 1100.0),
    3: (400.0, 1000.0),
}


class MockSpectroBackend(SpectroBackend):
    def __init__(self) -> None:
        self._connected = False
        self._grating = 1
        self._wavelength_nm = 600.0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def list_gratings(self) -> list[GratingInfo]:
        return list(_GRATINGS)

    def get_grating(self) -> int:
        return self._grating

    def set_grating(self, index: int) -> None:
        if index < 1 or index > len(_GRATINGS):
            raise SpectroError(f"Invalid grating index {index}")
        self._grating = index
        lo, hi = _LIMITS[index]
        self._wavelength_nm = float(np.clip(self._wavelength_nm, lo, hi))

    def get_wavelength(self) -> float:
        return self._wavelength_nm

    def set_wavelength(self, nm: float) -> None:
        lo, hi = self.get_wavelength_limits()
        if not (lo <= nm <= hi):
            raise SpectroError(f"Wavelength {nm:.1f} nm out of range [{lo:.0f}, {hi:.0f}] nm")
        self._wavelength_nm = float(nm)

    def get_wavelength_limits(self) -> tuple[float, float]:
        return _LIMITS.get(self._grating, (200.0, 1100.0))

    def get_calibration(self, n_pixels: int, pixel_width_um: float = 26.0) -> np.ndarray:
        if not self._connected:
            raise SpectroError("Not connected.")
        d = _DISPERSION.get(self._grating, 0.2)
        half = d * n_pixels / 2.0
        return np.linspace(self._wavelength_nm - half, self._wavelength_nm + half, n_pixels)
