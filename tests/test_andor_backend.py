from __future__ import annotations

from idus420_gui.camera.andor import (
    _DEFAULT_SHUTTER_TRANSFER_MS,
    _SHUTTER_TTL_TYPE,
    AndorIDusBackend,
)
from idus420_gui.camera.base import TempStatus


class _FakeErrorCodes:
    DRV_SUCCESS = 20002
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
