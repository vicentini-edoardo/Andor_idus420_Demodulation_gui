"""Backend contract isolating GUI code from the spectrograph SDK.

The spectrograph (an Andor Shamrock/Kymera) is a separate device from the
iDus camera.  This module mirrors the structure of :mod:`idus420_gui.camera`
so the GUI can talk to either a mock or the real ``pyAndorSpectrograph``
backend through one abstract interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class SpectrographError(RuntimeError):
    """Raised when the spectrograph backend or SDK reports an error."""


@dataclass(frozen=True)
class GratingInfo:
    """Description of a single installed grating.

    ``index`` is 1-based to match the Andor SDK convention.
    """

    index: int
    lines_per_mm: float
    blaze: str = ""
    home: int = 0
    offset: int = 0

    def label(self) -> str:
        """Human-readable combo-box label."""
        blaze = f", blaze {self.blaze}" if self.blaze else ""
        return f"{self.index}: {self.lines_per_mm:g} l/mm{blaze}"


class SpectrographBackend(ABC):
    """Abstract interface used by all GUI code to drive the spectrograph."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def serial_number(self) -> str: ...

    @abstractmethod
    def list_gratings(self) -> list[GratingInfo]: ...

    @abstractmethod
    def current_grating(self) -> int: ...

    @abstractmethod
    def set_grating(self, index: int) -> None: ...

    @abstractmethod
    def get_wavelength(self) -> float: ...

    @abstractmethod
    def set_wavelength(self, nm: float) -> None: ...

    @abstractmethod
    def wavelength_limits(self, grating_index: int) -> tuple[float, float]: ...

    @abstractmethod
    def set_pixel_geometry(self, pixel_width_um: float, number_pixels: int) -> None: ...

    @abstractmethod
    def get_calibration(self, number_pixels: int) -> np.ndarray: ...

    def sdk_version(self) -> str:
        """Return a backend SDK version string when available."""
        return "unavailable"
