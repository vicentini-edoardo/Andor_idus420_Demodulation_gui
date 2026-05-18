"""Backend contract and data types for SNOM stage control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np


class StageError(RuntimeError):
    """Raised when a stage backend reports an unrecoverable error."""


@dataclass(frozen=True)
class StagePoint:
    """A single point in the raster scan grid."""

    ix: int
    iy: int
    x_nm: float
    y_nm: float


@dataclass(frozen=True)
class ScanGrid:
    """Defines a 2-D XY raster grid in nm."""

    x_start_nm: float
    y_start_nm: float
    x_step_nm: float
    y_step_nm: float
    nx: int
    ny: int
    order: Literal["raster_lr", "snake"] = "snake"
    angle_deg: float = 0.0

    def total_points(self) -> int:
        return self.nx * self.ny

    def points(self) -> Iterator[StagePoint]:
        """Yield StagePoints in the configured scan order."""
        cos_a = np.cos(np.radians(self.angle_deg))
        sin_a = np.sin(np.radians(self.angle_deg))
        for iy in range(self.ny):
            if self.order == "snake" and iy % 2 == 1:
                x_range = range(self.nx - 1, -1, -1)
            else:
                x_range = range(self.nx)
            for ix in x_range:
                dx = ix * self.x_step_nm
                dy = iy * self.y_step_nm
                yield StagePoint(
                    ix=ix,
                    iy=iy,
                    x_nm=self.x_start_nm + dx * cos_a - dy * sin_a,
                    y_nm=self.y_start_nm + dx * sin_a + dy * cos_a,
                )


@dataclass
class SnomSample:
    """One snapshot of all SNOM signals sampled during Andor acquisition."""

    t_s: float
    xyz_nm: tuple[float, float, float]
    o_amp: np.ndarray    # shape (6,) optical amplitude harmonics 0-5
    o_phase: np.ndarray  # shape (6,) optical phase harmonics 0-5
    m_amp: np.ndarray    # shape (6,) mechanical amplitude harmonics 0-5
    m_phase: np.ndarray  # shape (6,) mechanical phase harmonics 0-5


class StageBackend(ABC):
    """Abstract interface for SNOM stage motion and signal readout."""

    @abstractmethod
    def connect(self, host: str) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def goto_xy_nm(self, x_nm: float, y_nm: float) -> None:
        """Move to (x_nm, y_nm) and block until the move completes."""
        ...

    @abstractmethod
    def read_xyz_nm(self) -> tuple[float, float, float]:
        """Return the current motor position (x, y, z) in nm."""
        ...

    @abstractmethod
    def read_sample(self, t_s: float) -> SnomSample:
        """Read one synchronous snapshot of all SNOM signals."""
        ...
