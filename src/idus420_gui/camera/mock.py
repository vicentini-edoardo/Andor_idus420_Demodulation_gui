"""Synthetic camera backend for development and tests without Andor hardware."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idus420_gui.camera.base import (
    AcquisitionStatus,
    AcquisitionTimings,
    CameraBackend,
    CameraConfig,
    CameraError,
    ReadMode,
    TempStatus,
    TriggerMode,
)


@dataclass
class MockSpectrumConfig:
    """Synthetic spectrum parameters."""

    baseline: float = 600.0
    noise_sigma: float = 8.0
    true_modulation_hz: float = 37.0
    sample_rate_hz: float = 500.0
    modulation_depth: float = 0.35


class MockBackend(CameraBackend):
    """Camera backend that generates plausible FVB spectra with a modulated line.

    Phase is computed from a deterministic integer frame counter so that test
    results are reproducible regardless of wall-clock timing or system load.
    """

    def __init__(
        self,
        detector: tuple[int, int] = (1024, 255),
        spectrum: MockSpectrumConfig | None = None,
        seed: int = 1234,
    ) -> None:
        self._detector = detector
        self._spectrum_cfg = spectrum or MockSpectrumConfig()
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._cooler_on = False
        self._target_temp = -60
        self._temperature = 20.0
        self._status = AcquisitionStatus.IDLE
        self._camera_cfg = CameraConfig()
        self._n_kinetics = 0
        # Global frame counter: monotonically increasing across all start() calls
        # so phase is continuous and deterministic without wall-clock dependence.
        self._frame_counter = 0
        self._run_frames: list[np.ndarray] = []
        self._read_index = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self.abort()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def serial_number(self) -> int:
        self._require_connected()
        return 420000

    def detector_size(self) -> tuple[int, int]:
        self._require_connected()
        return self._detector

    def frame_width(self) -> int:
        self._require_connected()
        xpix, _ = self._detector
        crop = self._camera_cfg.crop
        if crop.active:
            return crop.crop_width // self._camera_cfg.fvb_horizontal_bin
        if self._camera_cfg.read_mode is ReadMode.SINGLE_TRACK:
            return xpix // self._camera_cfg.single_track.horizontal_bin
        return xpix // self._camera_cfg.fvb_horizontal_bin

    def temperature_range(self) -> tuple[int, int]:
        self._require_connected()
        return (-95, 20)

    def set_target_temperature(self, t_celsius: int) -> None:
        self._require_connected()
        t_min, t_max = self.temperature_range()
        if not t_min <= t_celsius <= t_max:
            raise CameraError(f"Target temperature {t_celsius} C is outside {t_min}..{t_max} C")
        self._target_temp = int(t_celsius)

    def cooler_on(self) -> None:
        self._require_connected()
        self._cooler_on = True
        self._temperature = float(self._target_temp)

    def cooler_off(self) -> None:
        self._require_connected()
        self._cooler_on = False
        self._temperature = 20.0

    def get_temperature(self) -> tuple[float, TempStatus]:
        self._require_connected()
        if not self._cooler_on:
            return self._temperature, TempStatus.OFF
        return self._temperature, TempStatus.STABILIZED

    def list_hs_speeds(self) -> list[float]:
        self._require_connected()
        return [5.0, 3.0, 1.0, 0.05]

    def list_vs_speeds(self) -> list[float]:
        self._require_connected()
        return [4.7, 8.0, 16.0, 32.0]

    def list_preamp_gains(self) -> list[float]:
        self._require_connected()
        return [1.0, 2.4, 4.9]

    def configure(self, cfg: CameraConfig) -> None:
        self._require_connected()
        xpix, ypix = self._detector
        crop = cfg.crop
        if crop.active:
            hbin = cfg.fvb_horizontal_bin
            if crop.crop_width < 1 or crop.crop_width > xpix:
                raise CameraError(f"Crop width {crop.crop_width} must be 1..{xpix}.")
            if crop.crop_height < 1 or crop.crop_height > ypix:
                raise CameraError(f"Crop height {crop.crop_height} must be 1..{ypix}.")
            if crop.vbin < 1:
                raise CameraError("Crop vbin must be >= 1.")
            self._validate_horizontal_bin(hbin, crop.crop_width)
        else:
            hbin = (
                cfg.single_track.horizontal_bin
                if cfg.read_mode is ReadMode.SINGLE_TRACK
                else cfg.fvb_horizontal_bin
            )
            self._validate_horizontal_bin(hbin, xpix)
        if cfg.read_mode is ReadMode.SINGLE_TRACK:
            st = cfg.single_track
            if st.height < 1:
                raise CameraError("Single-Track height must be >= 1.")
            half = st.height // 2
            if st.center_row - half < 1 or st.center_row + half > ypix:
                raise CameraError(
                    f"Single-Track track (center={st.center_row}, height={st.height}) "
                    f"extends outside the detector ({ypix} rows)."
                )
        self._validate_index(cfg.hs_speed_index, self.list_hs_speeds(), "HS speed")
        self._validate_index(cfg.vs_speed_index, self.list_vs_speeds(), "VS speed")
        self._validate_index(cfg.preamp_gain_index, self.list_preamp_gains(), "pre-amp gain")
        if cfg.exposure_s <= 0:
            raise CameraError("Exposure must be positive.")
        self._camera_cfg = cfg

    def setup_kinetic(
        self,
        exposure_s: float,
        n_kinetics: int,
        trigger: TriggerMode,
        n_accumulations: int = 1,
    ) -> AcquisitionTimings:
        self._require_connected()
        if exposure_s <= 0:
            raise CameraError("Exposure must be positive.")
        if n_kinetics <= 0:
            raise CameraError("Number of kinetics must be positive.")
        if n_accumulations <= 0:
            raise CameraError("Number of accumulations must be positive.")
        self._camera_cfg = CameraConfig(
            hs_speed_index=self._camera_cfg.hs_speed_index,
            vs_speed_index=self._camera_cfg.vs_speed_index,
            preamp_gain_index=self._camera_cfg.preamp_gain_index,
            shutter_mode=self._camera_cfg.shutter_mode,
            exposure_s=exposure_s,
            ad_channel=self._camera_cfg.ad_channel,
            output_amplifier=self._camera_cfg.output_amplifier,
            read_mode=self._camera_cfg.read_mode,
            fvb_horizontal_bin=self._camera_cfg.fvb_horizontal_bin,
            single_track=self._camera_cfg.single_track,
            crop=self._camera_cfg.crop,
        )
        self._n_kinetics = int(n_kinetics)
        if trigger is TriggerMode.INTERNAL:
            self._spectrum_cfg.sample_rate_hz = 1.0 / max(exposure_s, 1e-6)
        return AcquisitionTimings(
            exposure_s=exposure_s,
            accumulate_s=exposure_s * n_accumulations,
            kinetic_s=1.0 / self._spectrum_cfg.sample_rate_hz,
            readout_s=0.0005,
        )

    def start(self) -> None:
        self._require_connected()
        if self._n_kinetics <= 0:
            raise CameraError("setup_kinetic must be called before start.")
        # Frames are generated lazily on read so that a large kinetic series
        # does not allocate the entire run up front.
        self._status = AcquisitionStatus.ACQUIRING
        self._run_frames = []
        self._read_index = 0
        self._status = AcquisitionStatus.IDLE

    def abort(self) -> None:
        self._status = AcquisitionStatus.IDLE

    def status(self) -> AcquisitionStatus:
        return self._status

    def wait_next_frame(self, timeout_ms: int) -> bool:
        self._require_connected()
        return self._read_index < self._n_kinetics

    def _generate_through(self, count: int) -> None:
        """Ensure synthetic frames exist for indices ``0 .. count - 1``."""
        while len(self._run_frames) < count:
            self._run_frames.append(self._make_frame())

    def get_oldest_frame(self) -> np.ndarray:
        self._require_connected()
        if self._read_index >= self._n_kinetics:
            raise CameraError("No unread mock frames are available.")
        self._generate_through(self._read_index + 1)
        frame = self._run_frames[self._read_index]
        self._read_index += 1
        return frame.copy()

    def get_new_frames_batch(self) -> np.ndarray | None:
        self._require_connected()
        if self._read_index >= self._n_kinetics:
            return None
        self._generate_through(self._n_kinetics)
        frames = np.stack(self._run_frames[self._read_index:], axis=0)
        self._read_index = self._n_kinetics
        return frames.copy()

    def get_all_frames(self, n: int) -> np.ndarray:
        self._require_connected()
        if n < 0:
            raise CameraError("Requested frame count cannot be negative.")
        if n > self._n_kinetics:
            raise CameraError(
                f"Only {self._n_kinetics} frames are available, requested {n}."
            )
        self._generate_through(n)
        return np.stack(self._run_frames[:n], axis=0).copy()

    def query_timings(self) -> AcquisitionTimings:
        self._require_connected()
        cfg = self._camera_cfg
        return AcquisitionTimings(
            exposure_s=cfg.exposure_s,
            accumulate_s=cfg.exposure_s,
            kinetic_s=1.0 / self._spectrum_cfg.sample_rate_hz,
            readout_s=0.0005,
        )

    def sdk_version(self) -> str:
        return "mock-0.1"

    def _make_frame(self) -> np.ndarray:
        xpix, _ = self._detector
        crop = self._camera_cfg.crop
        width = crop.crop_width if crop.active else xpix
        x = np.arange(width, dtype=np.float64)
        t = self._frame_counter / self._spectrum_cfg.sample_rate_hz
        self._frame_counter += 1

        spectrum = np.full(width, self._spectrum_cfg.baseline, dtype=np.float64)
        spectrum += 900.0 * np.exp(-0.5 * ((x - 220.0) / 9.0) ** 2)
        modulation = 1.0 + self._spectrum_cfg.modulation_depth * np.sin(
            2.0 * np.pi * self._spectrum_cfg.true_modulation_hz * t
        )
        spectrum += 1400.0 * modulation * np.exp(-0.5 * ((x - 520.0) / 14.0) ** 2)
        spectrum += 700.0 * np.exp(-0.5 * ((x - 790.0) / 20.0) ** 2)
        spectrum += self._rng.normal(0.0, self._spectrum_cfg.noise_sigma, size=width)
        hbin = (
            self._camera_cfg.single_track.horizontal_bin
            if (not crop.active and self._camera_cfg.read_mode is ReadMode.SINGLE_TRACK)
            else self._camera_cfg.fvb_horizontal_bin
        )
        if hbin > 1:
            spectrum = spectrum.reshape(width // hbin, hbin).sum(axis=1)
        return np.clip(spectrum, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def _require_connected(self) -> None:
        if not self._connected:
            raise CameraError("Camera backend is not connected.")

    @staticmethod
    def _validate_index(index: int, values: list[float], name: str) -> None:
        if not 0 <= index < len(values):
            raise CameraError(f"{name} index {index} is outside 0..{len(values) - 1}.")

    @staticmethod
    def _validate_horizontal_bin(hbin: int, xpix: int) -> None:
        if hbin < 1:
            raise CameraError("Horizontal bin must be >= 1.")
        if xpix % hbin != 0:
            raise CameraError(f"Horizontal bin {hbin} must divide detector width {xpix}.")
