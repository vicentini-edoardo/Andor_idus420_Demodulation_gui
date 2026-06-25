"""Backend contract for Shamrock-style spectrograph control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class SpectroError(RuntimeError):
    """Raised when the spectrograph backend reports an error."""


@dataclass(frozen=True)
class GratingInfo:
    """Properties of a single grating installed in the spectrograph."""

    index: int           # 1-based slot index
    lines_per_mm: float
    blaze: str           # "500" (nm), "UV", "VIS", or "0" for mirror


class SpectroBackend(ABC):
    """Abstract interface for spectrograph grating and wavelength control."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def list_gratings(self) -> list[GratingInfo]: ...

    @abstractmethod
    def get_grating(self) -> int:
        """Return the active grating slot (1-based)."""
        ...

    @abstractmethod
    def set_grating(self, index: int) -> None:
        """Set the active grating; blocks until the move completes."""
        ...

    @abstractmethod
    def get_wavelength(self) -> float:
        """Return the current centre wavelength in nm."""
        ...

    @abstractmethod
    def set_wavelength(self, nm: float) -> None:
        """Move to the requested centre wavelength in nm."""
        ...

    @abstractmethod
    def get_wavelength_limits(self) -> tuple[float, float]:
        """Return (min_nm, max_nm) accessible with the current grating."""
        ...

    @abstractmethod
    def get_calibration(self, n_pixels: int, pixel_width_um: float = 26.0) -> np.ndarray:
        """Return a wavelength-per-pixel calibration array of length n_pixels.

        pixel_width_um: physical pixel pitch of the detector in µm (iDus 420 = 26 µm).
        """
        ...
