"""Andor Shamrock spectrograph backend via pyAndorSpectrograph.

All direct pyAndorSpectrograph usage is isolated here so the rest of the
codebase remains testable with MockSpectroBackend.
"""

from __future__ import annotations

import contextlib

import numpy as np

from .base import GratingInfo, SpectroBackend, SpectroError

_SUCCESS = 20202  # SHAMROCK_SUCCESS error code


def _check(ret: int, name: str) -> None:
    if ret != _SUCCESS:
        raise SpectroError(f"Shamrock {name} failed (code {ret})")


class AndorShamrockBackend(SpectroBackend):
    """Controls an Andor Shamrock spectrograph via pyAndorSpectrograph."""

    def __init__(self) -> None:
        self._spc: object | None = None
        self._connected = False
        self._n_gratings = 0

    def connect(self) -> None:
        try:
            from pyAndorSpectrograph.spectrograph import (
                ATSpectrograph,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            raise SpectroError(
                "pyAndorSpectrograph not importable. "
                "Install it from the Andor SDK Python folder."
            ) from exc
        spc = ATSpectrograph()
        ret = spc.Initialize("")
        # Initialize returns a tuple (ret,) or just ret depending on wrapper version
        ret_code = ret[0] if isinstance(ret, (tuple, list)) else int(ret)
        _check(ret_code, "Initialize")
        self._spc = spc
        ret, n = spc.GetNumberGratings(0, 64)
        _check(ret, "GetNumberGratings")
        self._n_gratings = int(n)
        self._connected = True

    def disconnect(self) -> None:
        if self._spc is not None:
            with contextlib.suppress(Exception):
                self._spc.Close(0)  # type: ignore[attr-defined]
        self._spc = None
        self._connected = False
        self._n_gratings = 0

    def is_connected(self) -> bool:
        return self._connected

    def list_gratings(self) -> list[GratingInfo]:
        if not self._spc:
            raise SpectroError("Not connected.")
        gratings: list[GratingInfo] = []
        for i in range(1, self._n_gratings + 1):
            ret, lines, blaze, home, offset = self._spc.GetGratingInfo(0, i, 64)  # type: ignore[attr-defined]
            _check(ret, f"GetGratingInfo({i})")
            gratings.append(GratingInfo(index=i, lines_per_mm=float(lines), blaze=str(blaze)))
        return gratings

    def get_grating(self) -> int:
        if not self._spc:
            raise SpectroError("Not connected.")
        ret, grating = self._spc.GetGrating(0)  # type: ignore[attr-defined]
        _check(ret, "GetGrating")
        return int(grating)

    def set_grating(self, index: int) -> None:
        if not self._spc:
            raise SpectroError("Not connected.")
        ret = self._spc.SetGrating(0, index)  # type: ignore[attr-defined]
        _check(ret, f"SetGrating({index})")

    def get_wavelength(self) -> float:
        if not self._spc:
            raise SpectroError("Not connected.")
        ret, wl = self._spc.GetWavelength(0)  # type: ignore[attr-defined]
        _check(ret, "GetWavelength")
        return float(wl)

    def set_wavelength(self, nm: float) -> None:
        if not self._spc:
            raise SpectroError("Not connected.")
        ret = self._spc.SetWavelength(0, float(nm))  # type: ignore[attr-defined]
        _check(ret, f"SetWavelength({nm:.2f})")

    def get_wavelength_limits(self) -> tuple[float, float]:
        if not self._spc:
            raise SpectroError("Not connected.")
        grating = self.get_grating()
        ret, minwl, maxwl = self._spc.GetWavelengthLimits(0, grating)  # type: ignore[attr-defined]
        _check(ret, "GetWavelengthLimits")
        return float(minwl), float(maxwl)

    def get_calibration(self, n_pixels: int, pixel_width_um: float = 26.0) -> np.ndarray:
        if not self._spc:
            raise SpectroError("Not connected.")
        ret = self._spc.SetNumberPixels(0, int(n_pixels))  # type: ignore[attr-defined]
        _check(ret, "SetNumberPixels")
        ret = self._spc.SetPixelWidth(0, float(pixel_width_um))  # type: ignore[attr-defined]
        _check(ret, "SetPixelWidth")
        ret, cal = self._spc.GetCalibration(0, int(n_pixels))  # type: ignore[attr-defined]
        _check(ret, "GetCalibration")
        return np.asarray(cal, dtype=np.float64)
