from __future__ import annotations

from idus420_gui.camera.andor import (
    _DEFAULT_SHUTTER_TRANSFER_MS,
    _SHUTTER_TTL_TYPE,
    AndorIDusBackend,
)
from idus420_gui.camera.base import (
    CameraConfig,
    ReadMode,
    ShutterMode,
    TempStatus,
    TriggerMode,
)


class _FakeErrorCodes:
    DRV_SUCCESS = 20002
    DRV_IDLE = 20073
    DRV_NOT_INITIALIZED = 20075
    DRV_TEMPERATURE_OFF = 20034
    DRV_TEMPERATURE_STABILIZED = 20036


def test_error_name_falls_back_to_numeric_attributes() -> None:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._err = _FakeErrorCodes
    backend._code_names = backend._build_code_name_map()

    assert backend._error_name(20034) == "DRV_TEMPERATURE_OFF"


def test_temperature_status_accepts_temperature_aliases() -> None:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._err = _FakeErrorCodes
    backend._code_names = backend._build_code_name_map()

    assert backend._temperature_status(20034) is TempStatus.OFF
    assert backend._temperature_status(20036) is TempStatus.STABILIZED


class _SdkWithoutLength:
    def GetOldestImage16(self, _buf):  # noqa: N802 - SDK naming
        return 20002


class _SdkWithLength:
    def GetOldestImage16(self, _buf, _size):  # noqa: N802 - SDK naming
        return 20002


class _SdkWithLengthOnlyForBatch:
    def GetImages16(self, _first, _last, _buf, _size):  # noqa: N802 - SDK naming
        return 20002, 0, 0


class _SdkOldestBrokenMostRecentWorks:
    def GetOldestImage16(self, _buf):  # noqa: N802 - SDK naming
        raise TypeError("'numpy.ndarray' object is not callable")

    def GetMostRecentImage16(self, _buf):  # noqa: N802 - SDK naming
        return 20002


class _SdkReturnsArrayForOldest:
    def GetOldestImage16(self, _size):  # noqa: N802 - SDK naming
        return 20002, [1, 2, 3, 4]


class _SdkWithShutterEx:
    def __init__(self) -> None:
        self.shutter_ex_calls: list[tuple[int, ...]] = []
        self.shutter_calls: list[tuple[int, ...]] = []

    def GetShutterMinTimes(self):  # noqa: N802 - SDK naming
        return 20002, 13, 27

    def SetShutterEx(self, typ, mode, closing, opening, extmode):  # noqa: N802
        self.shutter_ex_calls.append((typ, mode, closing, opening, extmode))
        return 20002

    def SetShutter(self, typ, mode, closing, opening):  # noqa: N802
        self.shutter_calls.append((typ, mode, closing, opening))
        return 20002


class _SdkWithoutShutterEx:
    def __init__(self) -> None:
        self.shutter_calls: list[tuple[int, ...]] = []

    def SetShutter(self, typ, mode, closing, opening):  # noqa: N802
        self.shutter_calls.append((typ, mode, closing, opening))
        return 20002


def _shutter_backend(sdk: object) -> AndorIDusBackend:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._sdk = sdk
    backend._err = _FakeErrorCodes
    backend._code_names = backend._build_code_name_map()
    return backend


def test_apply_shutter_uses_shutter_ex_with_internal_mode_and_min_times() -> None:
    sdk = _SdkWithShutterEx()
    backend = _shutter_backend(sdk)

    backend._apply_shutter(1)  # 1 == "permanently open"

    # The internal shutter (extmode) must be driven, not just the external TTL,
    # and the transfer times come from GetShutterMinTimes (never 0).
    assert sdk.shutter_ex_calls == [(_SHUTTER_TTL_TYPE, 1, 13, 27, 1)]
    assert sdk.shutter_calls == []


def test_apply_shutter_falls_back_to_set_shutter_with_nonzero_times() -> None:
    sdk = _SdkWithoutShutterEx()
    backend = _shutter_backend(sdk)

    backend._apply_shutter(2)  # 2 == "permanently closed"

    typ, mode, closing, opening = sdk.shutter_calls[0]
    assert (typ, mode) == (_SHUTTER_TTL_TYPE, 2)
    assert closing == _DEFAULT_SHUTTER_TRANSFER_MS
    assert opening == _DEFAULT_SHUTTER_TRANSFER_MS
    assert closing > 0 and opening > 0


class _RecordingSdk:
    """Fake Andor SDK that records the order of every method call.

    Tuple-returning queries are stubbed explicitly; every other ``SetXxx`` call
    is captured via ``__getattr__`` and reports success.  This lets tests assert
    the *sequence* of SDK calls a high-level method issues — the kind of check
    that would have caught the shutter being driven through the wrong call.
    """

    DRV_SUCCESS = 20002

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def names(self) -> list[str]:
        return [name for name, _args in self.calls]

    def args_for(self, name: str) -> tuple:
        for call_name, args in self.calls:
            if call_name == name:
                return args
        raise AssertionError(f"{name} was never called")

    # --- tuple-returning queries -------------------------------------------
    def GetDetector(self):  # noqa: N802 - SDK naming
        self.calls.append(("GetDetector", ()))
        return 20002, 1024, 256

    def GetAcquisitionTimings(self):  # noqa: N802
        self.calls.append(("GetAcquisitionTimings", ()))
        return 20002, 0.001, 0.002, 0.003

    def GetReadOutTime(self):  # noqa: N802
        self.calls.append(("GetReadOutTime", ()))
        return 20002, 0.0005

    def GetStatus(self):  # noqa: N802
        self.calls.append(("GetStatus", ()))
        return 20002, 20073  # DRV_IDLE

    def GetShutterMinTimes(self):  # noqa: N802
        self.calls.append(("GetShutterMinTimes", ()))
        return 20002, 11, 23

    def __getattr__(self, name: str):
        def _record(*args):
            self.calls.append((name, args))
            return 20002
        return _record


def _recording_backend() -> tuple[AndorIDusBackend, _RecordingSdk]:
    sdk = _RecordingSdk()
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._sdk = sdk
    backend._err = _FakeErrorCodes
    backend._code_names = backend._build_code_name_map()
    backend._xpix, backend._ypix = 1024, 256
    backend._frame_width = 1024
    backend._n_kinetics = 0
    backend._crop_active = False
    return backend, sdk


def test_configure_drives_shutter_before_timings_query() -> None:
    backend, sdk = _recording_backend()

    backend.configure(
        CameraConfig(
            read_mode=ReadMode.FVB,
            fvb_horizontal_bin=1,
            shutter_mode=ShutterMode.OPEN,
        )
    )

    names = sdk.names()
    # FVB read mode configured and the internal shutter driven via SetShutterEx.
    assert "SetReadMode" in names
    assert "SetShutterEx" in names
    assert "SetShutter" not in names  # the Ex variant must be preferred
    # Shutter is applied before the timings are read back.
    assert names.index("SetShutterEx") < names.index("GetAcquisitionTimings")
    # The core analog/exposure chain is all issued.
    for required in (
        "SetADChannel",
        "SetOutputAmplifier",
        "SetHSSpeed",
        "SetVSSpeed",
        "SetPreAmpGain",
        "SetExposureTime",
    ):
        assert required in names
    # SetShutterEx(typ, mode, closing, opening, extmode) with non-zero min times.
    assert sdk.args_for("SetShutterEx") == (_SHUTTER_TTL_TYPE, 1, 11, 23, 1)


def test_setup_kinetic_sets_mode_then_prepares() -> None:
    backend, sdk = _recording_backend()

    timings = backend.setup_kinetic(0.01, 16, TriggerMode.EXTERNAL)

    names = sdk.names()
    for required in (
        "SetAcquisitionMode",
        "SetTriggerMode",
        "SetNumberAccumulations",
        "SetNumberKinetics",
        "PrepareAcquisition",
    ):
        assert required in names
    # Kinetic mode (3) and external trigger (1) configured before preparing.
    assert sdk.args_for("SetAcquisitionMode") == (3,)
    assert sdk.args_for("SetNumberKinetics") == (16,)
    assert names.index("SetAcquisitionMode") < names.index("PrepareAcquisition")
    # Timings come straight from the SDK stub.
    assert timings.exposure_s == 0.001
    assert timings.kinetic_s == 0.003


def test_optional_length_helper_handles_wrapper_without_size_parameter() -> None:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._sdk = _SdkWithoutLength()

    ret = backend._call_with_optional_length("GetOldestImage16", [0, 1, 2])

    assert ret == 20002


def test_optional_length_helper_handles_wrapper_with_size_parameter() -> None:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._sdk = _SdkWithLength()

    class _Buf:
        size = 3

    ret = backend._call_with_optional_length("GetOldestImage16", _Buf())

    assert ret == 20002


def test_optional_length_helper_falls_back_to_size_when_wrapper_requires_it() -> None:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._sdk = _SdkWithLengthOnlyForBatch()

    class _Buf:
        size = 8

    ret = backend._call_with_optional_length("GetImages16", 0, 0, _Buf())

    assert ret == (20002, 0, 0)


def test_get_oldest_frame_falls_back_to_most_recent_on_wrapper_type_error() -> None:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._sdk = _SdkOldestBrokenMostRecentWorks()
    backend._err = _FakeErrorCodes
    backend._code_names = backend._build_code_name_map()
    backend._xpix = 4
    backend._ypix = 1

    frame = backend.get_oldest_frame()

    assert frame.shape == (4,)


def test_get_oldest_frame_accepts_returned_array_payload() -> None:
    backend = AndorIDusBackend.__new__(AndorIDusBackend)
    backend._sdk = _SdkReturnsArrayForOldest()
    backend._err = _FakeErrorCodes
    backend._code_names = backend._build_code_name_map()
    backend._xpix = 4
    backend._ypix = 1

    frame = backend.get_oldest_frame()

    assert frame.tolist() == [1, 2, 3, 4]
