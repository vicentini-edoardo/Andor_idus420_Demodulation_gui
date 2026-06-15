"""Andor Shamrock/Kymera backend via the ``pyAndorSpectrograph`` wrapper.

All direct ``pyAndorSpectrograph`` usage is contained in this module so the GUI
remains testable with the mock backend.  The ``ATSpectrograph`` wrapper returns
``(return_code, *values)`` tuples; helpers here parse them defensively because
exact signatures differ slightly between SDK releases.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from idus420_gui.spectrograph.base import (
    GratingInfo,
    SpectrographBackend,
    SpectrographError,
)

LOG = logging.getLogger(__name__)

# ATSPECTROGRAPH_SUCCESS from the Shamrock SDK headers.
_SUCCESS = 20202


class AndorShamrockBackend(SpectrographBackend):
    """Spectrograph backend implemented through the vendor ``ATSpectrograph`` API."""

    def __init__(self, device: int = 0) -> None:
        try:
            from pyAndorSpectrograph.spectrograph import (  # type: ignore[import-not-found]
                ATSpectrograph,
            )
        except ImportError as exc:
            raise SpectrographError(
                "pyAndorSpectrograph is not importable. Install it from the Andor "
                "Spectrograph SDK Python folder."
            ) from exc

        self._sdk = ATSpectrograph()
        self._device = int(device)
        self._connected = False
        self._success = int(getattr(self._sdk, "ATSPECTROGRAPH_SUCCESS", _SUCCESS))

    def connect(self) -> None:
        self._check(self._sdk.Initialize(""), "Initialize")
        self._connected = True
        _, count = self._checked_tuple(self._sdk.GetNumberDevices(), "GetNumberDevices")
        if int(count) <= self._device:
            self.disconnect()
            raise SpectrographError(
                f"No spectrograph at device index {self._device} (found {int(count)})."
            )
        LOG.info("Shamrock SDK version: %s", self.sdk_version())

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            self._sdk.Close()
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def serial_number(self) -> str:
        result = self._sdk.GetSerialNumber(self._device)
        if isinstance(result, tuple):
            self._check(int(result[0]), "GetSerialNumber")
            return str(result[-1]).strip()
        return str(result).strip()

    def list_gratings(self) -> list[GratingInfo]:
        _, number = self._checked_tuple(
            self._sdk.GetNumberGratings(self._device), "GetNumberGratings"
        )
        gratings: list[GratingInfo] = []
        for index in range(1, int(number) + 1):
            result = self._checked_tuple(
                self._sdk.GetGratingInfo(self._device, index), "GetGratingInfo"
            )
            lines, blaze, home, offset = self._parse_grating_info(result)
            gratings.append(
                GratingInfo(
                    index=index,
                    lines_per_mm=lines,
                    blaze=blaze,
                    home=home,
                    offset=offset,
                )
            )
        return gratings

    def current_grating(self) -> int:
        _, grating = self._checked_tuple(
            self._sdk.GetGrating(self._device), "GetGrating"
        )
        return int(grating)

    def set_grating(self, index: int) -> None:
        self._check(self._sdk.SetGrating(self._device, int(index)), "SetGrating")

    def get_wavelength(self) -> float:
        _, wavelength = self._checked_tuple(
            self._sdk.GetWavelength(self._device), "GetWavelength"
        )
        return float(wavelength)

    def set_wavelength(self, nm: float) -> None:
        self._check(
            self._sdk.SetWavelength(self._device, float(nm)), "SetWavelength"
        )

    def wavelength_limits(self, grating_index: int) -> tuple[float, float]:
        result = self._checked_tuple(
            self._sdk.GetWavelengthLimits(self._device, int(grating_index)),
            "GetWavelengthLimits",
        )
        lo, hi = float(result[1]), float(result[2])
        return lo, hi

    def set_pixel_geometry(self, pixel_width_um: float, number_pixels: int) -> None:
        self._check(
            self._sdk.SetPixelWidth(self._device, float(pixel_width_um)),
            "SetPixelWidth",
        )
        self._check(
            self._sdk.SetNumberPixels(self._device, int(number_pixels)),
            "SetNumberPixels",
        )

    def get_calibration(self, number_pixels: int) -> np.ndarray:
        result = self._sdk.GetCalibration(self._device, int(number_pixels))
        if isinstance(result, tuple):
            self._check(int(result[0]), "GetCalibration")
            for item in result[1:]:
                if isinstance(item, (np.ndarray, list, tuple)):
                    return np.asarray(item, dtype=np.float64)
            raise SpectrographError("GetCalibration returned no calibration array.")
        return np.asarray(result, dtype=np.float64)

    def sdk_version(self) -> str:
        if hasattr(self._sdk, "GetSoftwareVersion"):
            try:
                result = self._sdk.GetSoftwareVersion()
                if isinstance(result, tuple):
                    self._check(int(result[0]), "GetSoftwareVersion")
                    return ".".join(str(v) for v in result[1:])
            except Exception:  # noqa: BLE001 - SDK wrappers differ here.
                LOG.debug("Could not read Shamrock SDK version", exc_info=True)
        return "Andor Shamrock SDK"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check(self, ret: int, name: str) -> None:
        if int(ret) != self._success:
            raise SpectrographError(f"{name} failed: code {int(ret)}")

    def _checked_tuple(self, result: tuple[Any, ...], name: str) -> tuple[Any, ...]:
        if not isinstance(result, tuple):
            raise SpectrographError(f"{name} returned {result!r}, expected a tuple.")
        self._check(int(result[0]), name)
        return result

    @staticmethod
    def _parse_grating_info(result: tuple[Any, ...]) -> tuple[float, str, int, int]:
        """Extract (lines/mm, blaze, home, offset) from a GetGratingInfo tuple.

        The wrapper returns ``(ret, lines, blaze, home, offset)`` but field
        order and presence vary between releases, so parse positionally with
        sensible fallbacks.
        """
        values = list(result[1:])
        lines = float(values[0]) if values else 0.0
        blaze = ""
        home = 0
        offset = 0
        for item in values[1:]:
            if isinstance(item, str) and not blaze:
                blaze = item.strip()
            elif isinstance(item, (int, float)):
                if home == 0:
                    home = int(item)
                else:
                    offset = int(item)
        return lines, blaze, home, offset
