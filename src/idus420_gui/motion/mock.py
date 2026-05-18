"""Synthetic SNOM stage backend for development and tests without hardware."""

from __future__ import annotations

import math

import numpy as np

from idus420_gui.motion.base import SnomSample, StageBackend, StageError

_N_HARMONICS = 6


class MockStageBackend(StageBackend):
    """Simulates motor moves and returns synthetic harmonic signal values.

    Signal amplitudes decay with distance from the origin so that scan maps
    look non-trivial in tests.
    """

    def __init__(self, seed: int = 42) -> None:
        self._connected = False
        self._x_nm: float = 0.0
        self._y_nm: float = 0.0
        self._z_nm: float = 0.0
        self._rng = np.random.default_rng(seed)

    def connect(self, host: str = "mock") -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def goto_xy_nm(self, x_nm: float, y_nm: float) -> None:
        self._require_connected()
        self._x_nm = float(x_nm)
        self._y_nm = float(y_nm)

    def read_xyz_nm(self) -> tuple[float, float, float]:
        self._require_connected()
        return self._x_nm, self._y_nm, self._z_nm

    def read_sample(self, t_s: float) -> SnomSample:
        self._require_connected()
        r = math.sqrt(self._x_nm ** 2 + self._y_nm ** 2)
        decay = math.exp(-r / 5000.0)

        o_amp = np.array(
            [decay * (1.0 / (h + 1)) + self._rng.normal(0, 0.01) for h in range(_N_HARMONICS)]
        )
        o_phase = np.array(
            [math.pi * h / _N_HARMONICS + self._rng.normal(0, 0.05) for h in range(_N_HARMONICS)]
        )
        m_amp = np.array(
            [decay * 0.5 / (h + 1) + self._rng.normal(0, 0.005) for h in range(_N_HARMONICS)]
        )
        m_phase = np.array(
            [math.pi * h / (_N_HARMONICS + 1) + self._rng.normal(0, 0.05) for h in range(_N_HARMONICS)]
        )
        return SnomSample(
            t_s=t_s,
            xyz_nm=(self._x_nm, self._y_nm, self._z_nm),
            o_amp=o_amp,
            o_phase=o_phase,
            m_amp=m_amp,
            m_phase=m_phase,
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise StageError("Mock stage backend is not connected.")
