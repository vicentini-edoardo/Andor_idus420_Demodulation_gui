from __future__ import annotations

from idus420_gui.camera.andor import AndorIDusBackend
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
