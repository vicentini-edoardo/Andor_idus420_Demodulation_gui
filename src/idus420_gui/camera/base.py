"""Backend contract isolating GUI code from camera SDK details."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CameraError(RuntimeError):
    """Raised when the camera backend or SDK reports an unrecoverable error."""


class TempStatus(str, Enum):
    """Temperature state reported by the camera controller."""

    OFF = "off"
    NOT_REACHED = "not_reached"
    NOT_STABILIZED = "not_stabilized"
    STABILIZED = "stabilized"
    DRIFT = "drift"
    UNKNOWN = "unknown"


class TriggerMode(str, Enum):
    """Supported acquisition trigger modes."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    EXTERNAL_START = "external_start"
    EXTERNAL_EXPOSURE = "external_exposure"


class AcquisitionStatus(str, Enum):
    """High-level acquisition state."""

    IDLE = "idle"
    ACQUIRING = "acquiring"
    ERROR = "error"


class ShutterMode(str, Enum):
    """User-facing shutter modes."""

    AUTO = "auto"
    OPEN = "open"
    CLOSED = "closed"
    OPEN_FVB_SERIES = "open_fvb_series"
    OPEN_ANY_SERIES = "open_any_series"


class ReadMode(str, Enum):
    """Supported pixel read modes."""

    FVB = "fvb"
    SINGLE_TRACK = "single_track"


@dataclass(frozen=True)
class SingleTrackConfig:
    """Parameters for Single-Track read mode."""

    center_row: int = 128
    height: int = 10
    horizontal_bin: int = 1


@dataclass(frozen=True)
class CropConfig:
    """Parameters for isolated crop mode (SetIsolatedCropMode).

    Reduces the effective CCD area for higher throughput.  On the iDus only
    FVB read mode is supported.  hbin is taken from the parent CameraConfig's
    fvb_horizontal_bin field so that a single control governs both modes.
    """

    active: bool = False
    crop_height: int = 50
    crop_width: int = 1024
    vbin: int = 1


@dataclass(frozen=True)
class CameraConfig:
    """Static camera settings selected in the Camera Settings panel."""

    hs_speed_index: int = 0
    vs_speed_index: int = 0
    preamp_gain_index: int = 0
    shutter_mode: ShutterMode = ShutterMode.OPEN
    exposure_s: float = 0.001
    ad_channel: int = 0
    output_amplifier: int = 0
    read_mode: ReadMode = ReadMode.FVB
    fvb_horizontal_bin: int = 1
    single_track: SingleTrackConfig = SingleTrackConfig()
    crop: CropConfig = CropConfig()


@dataclass(frozen=True)
class AcquisitionTimings:
    """Actual timing values returned by the camera driver, in seconds."""

    exposure_s: float
    accumulate_s: float
    kinetic_s: float
    readout_s: float | None = None


class CameraBackend(ABC):
    """Abstract interface used by all GUI and worker code."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def serial_number(self) -> int: ...

    @abstractmethod
    def detector_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def frame_width(self) -> int: ...

    @abstractmethod
    def temperature_range(self) -> tuple[int, int]: ...

    @abstractmethod
    def set_target_temperature(self, t_celsius: int) -> None: ...

    @abstractmethod
    def cooler_on(self) -> None: ...

    @abstractmethod
    def cooler_off(self) -> None: ...

    @abstractmethod
    def get_temperature(self) -> tuple[float, TempStatus]: ...

    @abstractmethod
    def list_hs_speeds(self) -> list[float]: ...

    @abstractmethod
    def list_vs_speeds(self) -> list[float]: ...

    @abstractmethod
    def list_preamp_gains(self) -> list[float]: ...

    @abstractmethod
    def configure(self, cfg: CameraConfig) -> None: ...

    @abstractmethod
    def setup_kinetic(
        self,
        exposure_s: float,
        n_kinetics: int,
        trigger: TriggerMode,
        n_accumulations: int = 1,
    ) -> AcquisitionTimings: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def abort(self) -> None: ...

    @abstractmethod
    def status(self) -> AcquisitionStatus: ...

    def wait_until_idle(self, timeout_s: float = 2.0, poll_s: float = 0.01) -> bool:
        """Poll the backend until it reports idle.

        Andor SDK calls are sensitive to reconfiguration while an abort is still
        settling. Backends can override this, but the generic status poll keeps
        worker code from immediately preparing a half-aborted camera.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            if self.status() is AcquisitionStatus.IDLE:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.001, float(poll_s)))

    def acquisition_diagnostics(self) -> str:
        """Return lightweight acquisition state for timeout/recovery logs."""
        try:
            return f"status={self.status().value}"
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break recovery.
            return f"status_error={exc}"

    @abstractmethod
    def wait_next_frame(self, timeout_ms: int) -> bool: ...

    @abstractmethod
    def get_oldest_frame(self) -> np.ndarray: ...

    @abstractmethod
    def get_new_frames_batch(self) -> np.ndarray | None: ...

    @abstractmethod
    def query_timings(self) -> AcquisitionTimings: ...

    @abstractmethod
    def get_all_frames(self, n: int) -> np.ndarray: ...

    def sdk_version(self) -> str:
        """Return a backend SDK version string when available."""
        return "unavailable"

    def save_as_sif(self, path: str) -> None:
        """Optional native Andor save hook."""
        raise CameraError("Native SIF saving is not supported by this backend.")
