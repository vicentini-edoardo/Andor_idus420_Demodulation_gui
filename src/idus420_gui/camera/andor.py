"""Andor SDK v2 backend for the iDus 420.

All direct `pyAndorSDK2` usage is contained in this module so GUI and worker code remain
testable with the mock backend.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from idus420_gui.camera.base import (
    AcquisitionStatus,
    AcquisitionTimings,
    CameraBackend,
    CameraConfig,
    CameraError,
    CropConfig,
    ReadMode,
    ShutterMode,
    TempStatus,
    TriggerMode,
)

LOG = logging.getLogger(__name__)


class AndorIDusBackend(CameraBackend):
    """Andor iDus backend implemented through the vendor `pyAndorSDK2` wrapper."""

    def __init__(self) -> None:
        try:
            from pyAndorSDK2 import atmcd, atmcd_errors  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CameraError(
                "pyAndorSDK2 is not importable. Install it from the Andor SDK v2 Python folder."
            ) from exc

        self._err = atmcd_errors.Error_Codes
        self._sdk = atmcd()
        self._connected = False
        self._xpix = 0
        self._ypix = 0
        self._frame_width = 0
        self._n_kinetics = 0
        self._code_names = self._build_code_name_map()

    def connect(self) -> None:
        self._check(self._sdk.Initialize(""), "Initialize")
        self._connected = True
        _, self._xpix, self._ypix = self._checked_tuple(
            self._sdk.GetDetector(),
            "GetDetector",
        )
        self._frame_width = self._xpix
        self._check(self._sdk.SetReadMode(0), "SetReadMode(FVB)")
        self._check(self._sdk.SetFVBHBin(1), "SetFVBHBin(1)")
        LOG.info("Andor SDK version: %s", self.sdk_version())

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            self.abort()
            self._sdk.CoolerOFF()
            self._check(self._sdk.ShutDown(), "ShutDown")
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def serial_number(self) -> int:
        _, serial = self._checked_tuple(
            self._sdk.GetCameraSerialNumber(),
            "GetCameraSerialNumber",
        )
        return int(serial)

    def detector_size(self) -> tuple[int, int]:
        if self._xpix and self._ypix:
            return self._xpix, self._ypix
        _, xpix, ypix = self._checked_tuple(self._sdk.GetDetector(), "GetDetector")
        self._xpix, self._ypix = int(xpix), int(ypix)
        return self._xpix, self._ypix

    def frame_width(self) -> int:
        if getattr(self, "_frame_width", 0) > 0:
            return self._frame_width
        xpix, _ = self.detector_size()
        return xpix

    def temperature_range(self) -> tuple[int, int]:
        _, t_min, t_max = self._checked_tuple(
            self._sdk.GetTemperatureRange(),
            "GetTemperatureRange",
        )
        return int(t_min), int(t_max)

    def set_target_temperature(self, t_celsius: int) -> None:
        self._check(self._sdk.SetTemperature(int(t_celsius)), "SetTemperature")

    def cooler_on(self) -> None:
        self._check(self._sdk.CoolerON(), "CoolerON")

    def cooler_off(self) -> None:
        self._check(self._sdk.CoolerOFF(), "CoolerOFF")

    def get_temperature(self) -> tuple[float, TempStatus]:
        ret, temp = self._sdk.GetTemperature()
        status = self._temperature_status(ret)
        if status is TempStatus.UNKNOWN:
            self._check(ret, "GetTemperature")
        return float(temp), status

    def list_hs_speeds(self) -> list[float]:
        _, n_hs = self._checked_tuple(self._sdk.GetNumberHSSpeeds(0, 0), "GetNumberHSSpeeds")
        return [
            float(self._checked_tuple(self._sdk.GetHSSpeed(0, 0, idx), "GetHSSpeed")[1])
            for idx in range(int(n_hs))
        ]

    def list_vs_speeds(self) -> list[float]:
        _, n_vs = self._checked_tuple(self._sdk.GetNumberVSSpeeds(), "GetNumberVSSpeeds")
        return [
            float(self._checked_tuple(self._sdk.GetVSSpeed(idx), "GetVSSpeed")[1])
            for idx in range(int(n_vs))
        ]

    def list_preamp_gains(self) -> list[float]:
        _, n_pre = self._checked_tuple(self._sdk.GetNumberPreAmpGains(), "GetNumberPreAmpGains")
        return [
            float(self._checked_tuple(self._sdk.GetPreAmpGain(idx), "GetPreAmpGain")[1])
            for idx in range(int(n_pre))
        ]

    def configure(self, cfg: CameraConfig) -> None:
        xpix, _ = self.detector_size()
        crop = cfg.crop
        if crop.active:
            if cfg.read_mode is not ReadMode.FVB:
                raise CameraError(
                    "Isolated crop mode on the iDus is only supported in FVB read mode."
                )
            hbin = int(cfg.fvb_horizontal_bin)
            if hbin < 1:
                raise CameraError("FVB horizontal bin must be >= 1.")
            if crop.crop_width % hbin != 0:
                raise CameraError(
                    f"Crop width {crop.crop_width} must be divisible by hbin {hbin}."
                )
            self._check(self._sdk.SetReadMode(0), "SetReadMode(FVB)")
            self._check(
                self._sdk.SetIsolatedCropMode(
                    1, crop.crop_height, crop.crop_width, crop.vbin, hbin
                ),
                "SetIsolatedCropMode",
            )
            self._frame_width = crop.crop_width // hbin
        else:
            # Disable crop mode, then configure read mode normally.
            self._check(
                self._sdk.SetIsolatedCropMode(0, 1, xpix, 1, 1),
                "SetIsolatedCropMode(disable)",
            )
            if cfg.read_mode is ReadMode.SINGLE_TRACK:
                hbin = int(cfg.single_track.horizontal_bin)
                self._validate_horizontal_bin(hbin, xpix, "Single-Track")
                self._check(self._sdk.SetReadMode(3), "SetReadMode(SingleTrack)")
                self._check(
                    self._sdk.SetSingleTrack(
                        cfg.single_track.center_row,
                        cfg.single_track.height,
                    ),
                    "SetSingleTrack",
                )
                self._check(
                    self._sdk.SetSingleTrackHBin(hbin),
                    f"SetSingleTrackHBin({hbin})",
                )
            else:
                hbin = int(cfg.fvb_horizontal_bin)
                self._validate_horizontal_bin(hbin, xpix, "FVB")
                self._check(self._sdk.SetReadMode(0), "SetReadMode(FVB)")
                self._check(self._sdk.SetFVBHBin(hbin), f"SetFVBHBin({hbin})")
            self._frame_width = xpix // hbin
        self._check(
            self._sdk.SetShutter(1, self._shutter_code(cfg.shutter_mode), 0, 0),
            "SetShutter",
        )
        self._check(self._sdk.SetADChannel(cfg.ad_channel), "SetADChannel")
        self._check(self._sdk.SetOutputAmplifier(cfg.output_amplifier), "SetOutputAmplifier")
        self._check(self._sdk.SetHSSpeed(cfg.output_amplifier, cfg.hs_speed_index), "SetHSSpeed")
        self._check(self._sdk.SetVSSpeed(cfg.vs_speed_index), "SetVSSpeed")
        self._check(self._sdk.SetPreAmpGain(cfg.preamp_gain_index), "SetPreAmpGain")
        self._check(self._sdk.SetExposureTime(float(cfg.exposure_s)), "SetExposureTime")
        self._checked_tuple(self._sdk.GetAcquisitionTimings(), "GetAcquisitionTimings")

    def setup_kinetic(
        self,
        exposure_s: float,
        n_kinetics: int,
        trigger: TriggerMode,
        n_accumulations: int = 1,
    ) -> AcquisitionTimings:
        self.abort()  # ensure camera is idle before reconfiguring
        if not self.wait_until_idle(timeout_s=2.0):
            raise CameraError("Camera did not become idle before kinetic setup.")
        self._n_kinetics = int(n_kinetics)
        self._check(self._sdk.SetAcquisitionMode(3), "SetAcquisitionMode(Kinetic)")
        self._check(self._sdk.SetTriggerMode(self._trigger_code(trigger)), "SetTriggerMode")
        self._check(self._sdk.SetExposureTime(float(exposure_s)), "SetExposureTime")
        self._check(
            self._sdk.SetNumberAccumulations(int(n_accumulations)),
            "SetNumberAccumulations",
        )
        self._check(self._sdk.SetNumberKinetics(int(n_kinetics)), "SetNumberKinetics")
        self._check(self._sdk.SetKineticCycleTime(0.0), "SetKineticCycleTime")
        _, exp, acc, kin = self._checked_tuple(
            self._sdk.GetAcquisitionTimings(),
            "GetAcquisitionTimings",
        )
        readout: float | None = None
        try:
            _, readout = self._checked_tuple(
                self._sdk.GetReadOutTime(),
                "GetReadOutTime",
            )
        except CameraError:
            LOG.debug("GetReadOutTime not available", exc_info=True)
        self._check(self._sdk.PrepareAcquisition(), "PrepareAcquisition")
        return AcquisitionTimings(
            float(exp),
            float(acc),
            float(kin),
            None if readout is None else float(readout),
        )

    def query_timings(self) -> AcquisitionTimings:
        """Return current acquisition timings without calling PrepareAcquisition."""
        _, exp, acc, kin = self._checked_tuple(
            self._sdk.GetAcquisitionTimings(),
            "GetAcquisitionTimings",
        )
        readout: float | None = None
        try:
            _, readout = self._checked_tuple(
                self._sdk.GetReadOutTime(),
                "GetReadOutTime",
            )
        except CameraError:
            LOG.debug("GetReadOutTime not available", exc_info=True)
        return AcquisitionTimings(
            float(exp),
            float(acc),
            float(kin),
            None if readout is None else float(readout),
        )

    def start(self) -> None:
        self._check(self._sdk.StartAcquisition(), "StartAcquisition")

    def abort(self) -> None:
        if hasattr(self._sdk, "CancelWait"):
            self._sdk.CancelWait()
        ret = self._sdk.AbortAcquisition()
        name = self._error_name(ret)
        if ret != self._success_code and name not in {"DRV_IDLE", "DRV_NOT_INITIALIZED"}:
            self._check(ret, "AbortAcquisition")
        self.wait_until_idle(timeout_s=2.0)

    def status(self) -> AcquisitionStatus:
        _, status = self._checked_tuple(self._sdk.GetStatus(), "GetStatus")
        name = self._error_name(status)
        if name == "DRV_IDLE":
            return AcquisitionStatus.IDLE
        if name == "DRV_ACQUIRING":
            return AcquisitionStatus.ACQUIRING
        return AcquisitionStatus.ERROR

    def wait_next_frame(self, timeout_ms: int) -> bool:
        ret = self._sdk.WaitForAcquisitionTimeOut(int(timeout_ms))
        if ret == self._success_code:
            return True
        name = self._error_name(ret)
        if name in {"DRV_NO_NEW_DATA", "DRV_TIMEOUT"}:
            return False
        raise CameraError(f"WaitForAcquisitionTimeOut: {name} ({ret})")

    def wait_until_idle(self, timeout_s: float = 2.0, poll_s: float = 0.01) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            try:
                if self.status() is AcquisitionStatus.IDLE:
                    return True
            except CameraError:
                LOG.debug("GetStatus failed while waiting for idle", exc_info=True)
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.001, float(poll_s)))

    def get_oldest_frame(self) -> np.ndarray:
        frame_width = self.frame_width()
        buf = np.empty(frame_width, dtype=np.uint16)
        try:
            data = self._read_frame_method("GetOldestImage16", frame_width, buf)
            if data is not None:
                return data
        except TypeError:
            if not hasattr(self._sdk, "GetMostRecentImage16"):
                raise
        if hasattr(self._sdk, "GetMostRecentImage16"):
            data = self._read_frame_method("GetMostRecentImage16", frame_width, buf)
            if data is not None:
                return data
        return buf

    def get_new_frames_batch(self) -> np.ndarray | None:
        ret, first, last = self._sdk.GetNumberNewImages()
        if ret != self._success_code:
            if self._error_name(ret) == "DRV_NO_NEW_DATA":
                return None
            self._check(ret, "GetNumberNewImages")
        if last < first:
            return None
        frame_width = self.frame_width()
        n = int(last - first + 1)
        buf = np.empty(n * frame_width, dtype=np.uint16)
        ret, valid_first, valid_last = self._call_with_optional_length(
            "GetImages16",
            first,
            last,
            buf,
        )
        self._check(ret, "GetImages16")
        valid_n = int(valid_last - valid_first + 1)
        return buf.reshape(n, frame_width)[:valid_n].copy()

    def get_all_frames(self, n: int) -> np.ndarray:
        frame_width = self.frame_width()
        buf = np.empty(int(n) * frame_width, dtype=np.uint16)
        data = self._read_frame_method("GetAcquiredData16", int(n) * frame_width, buf)
        if data is not None:
            return data.reshape(int(n), frame_width).copy()
        return buf.reshape(int(n), frame_width)

    def sdk_version(self) -> str:
        if hasattr(self._sdk, "GetVersionInfo"):
            try:
                ret = self._sdk.GetVersionInfo(0)
                if isinstance(ret, tuple):
                    self._check(ret[0], "GetVersionInfo")
                    return str(ret[-1])
            except Exception:  # noqa: BLE001 - SDK wrappers differ here.
                LOG.debug("Could not read SDK version", exc_info=True)
        return "Andor SDK v2"

    def save_as_sif(self, path: str) -> None:
        self._check(self._sdk.SaveAsSif(path), "SaveAsSif")

    @property
    def _success_code(self) -> int:
        return int(self._err.DRV_SUCCESS)

    def _check(self, ret: int, name: str) -> None:
        if int(ret) != self._success_code:
            raise CameraError(f"{name} failed: {self._error_name(ret)} ({ret})")

    def _checked_tuple(self, result: tuple[Any, ...], name: str) -> tuple[Any, ...]:
        self._check(int(result[0]), name)
        return result

    def _error_name(self, ret: int) -> str:
        try:
            return self._err(int(ret)).name
        except Exception:  # noqa: BLE001 - SDK enum may not contain every code.
            return self._code_names.get(int(ret), f"SDK_CODE_{ret}")

    def _temperature_status(self, ret: int) -> TempStatus:
        name = self._error_name(ret)
        normalized = name.replace("DRV_TEMPERATURE_", "DRV_TEMP_")
        return {
            "DRV_TEMP_OFF": TempStatus.OFF,
            "DRV_TEMP_NOT_REACHED": TempStatus.NOT_REACHED,
            "DRV_TEMP_NOT_STABILIZED": TempStatus.NOT_STABILIZED,
            "DRV_TEMP_STABILIZED": TempStatus.STABILIZED,
            "DRV_TEMP_DRIFT": TempStatus.DRIFT,
        }.get(normalized, TempStatus.UNKNOWN)

    def _build_code_name_map(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for attr in dir(self._err):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(self._err, attr)
            except Exception:  # noqa: BLE001 - defensive against unusual SDK wrappers.
                continue
            try:
                mapping[int(value)] = attr
            except Exception:  # noqa: BLE001 - skip non-numeric attributes.
                continue
        return mapping

    def _call_with_optional_length(self, method_name: str, *args: Any) -> Any:
        method = getattr(self._sdk, method_name)
        last_arg = args[-1]
        size = getattr(last_arg, "size", None)
        if size is None:
            size = len(last_arg)
        try:
            return method(*args)
        except TypeError as exc:
            message = str(exc)
            if (
                "required positional argument" not in message
                and "positional arguments" not in message
            ):
                raise
        return method(*args, size)

    def _read_frame_method(
        self,
        method_name: str,
        size: int,
        buf: np.ndarray,
    ) -> np.ndarray | None:
        method = getattr(self._sdk, method_name)
        attempts: tuple[tuple[Any, ...], ...] = (
            (buf,),
            (buf, size),
            (size,),
        )
        last_type_error: TypeError | None = None
        for args in attempts:
            try:
                result = method(*args)
            except TypeError as exc:
                last_type_error = exc
                continue
            parsed = self._parse_frame_result(method_name, result, size)
            if parsed is not None:
                return parsed
            return None
        if last_type_error is not None:
            raise last_type_error
        return None

    def _parse_frame_result(
        self,
        method_name: str,
        result: Any,
        size: int,
    ) -> np.ndarray | None:
        if isinstance(result, tuple):
            self._check(int(result[0]), method_name)
            for item in result[1:]:
                if isinstance(item, np.ndarray):
                    return np.asarray(item, dtype=np.uint16).reshape(size).copy()
                if isinstance(item, (list, tuple)) and len(item) == size:
                    return np.asarray(item, dtype=np.uint16).reshape(size).copy()
            return None
        if isinstance(result, np.ndarray):
            return np.asarray(result, dtype=np.uint16).reshape(size).copy()
        if isinstance(result, (list, tuple)) and len(result) == size:
            return np.asarray(result, dtype=np.uint16).reshape(size).copy()
        self._check(int(result), method_name)
        return None

    @staticmethod
    def _validate_horizontal_bin(hbin: int, xpix: int, read_mode_name: str) -> None:
        if hbin < 1:
            raise CameraError(f"{read_mode_name} horizontal bin must be >= 1.")
        if hbin != 1:
            raise CameraError(
                f"{read_mode_name} horizontal bin {hbin} is not supported on this iDus backend. "
                "Use horizontal bin = 1."
            )
        if xpix % hbin != 0:
            raise CameraError(
                f"{read_mode_name} horizontal bin {hbin} must divide detector width {xpix}."
            )

    @staticmethod
    def _trigger_code(trigger: TriggerMode) -> int:
        return {
            TriggerMode.INTERNAL: 0,
            TriggerMode.EXTERNAL: 1,
            TriggerMode.EXTERNAL_START: 6,
            TriggerMode.EXTERNAL_EXPOSURE: 7,
        }[trigger]

    @staticmethod
    def _shutter_code(shutter: ShutterMode) -> int:
        return {
            ShutterMode.AUTO: 0,
            ShutterMode.OPEN: 1,
            ShutterMode.CLOSED: 2,
            ShutterMode.OPEN_FVB_SERIES: 4,
            ShutterMode.OPEN_ANY_SERIES: 5,
        }[shutter]
