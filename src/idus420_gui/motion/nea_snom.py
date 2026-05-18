"""Real SNOM stage backend using nea_tools and neaspec SDK."""

from __future__ import annotations

try:
    import asyncio

    import nest_asyncio
    import nea_tools
    from nea_tools.microscope import motors as _nea_motors

    NEA_TOOLS_AVAILABLE = True
except ImportError:
    NEA_TOOLS_AVAILABLE = False

import numpy as np

from idus420_gui.motion.base import SnomSample, StageBackend, StageError

_N_HARMONICS = 6


class NeaSnomBackend(StageBackend):
    """Stage backend that wraps nea_tools for the SNOM Sample motor."""

    def __init__(self) -> None:
        if not NEA_TOOLS_AVAILABLE:
            raise StageError(
                "nea_tools / nest_asyncio are not installed. "
                "Install with: pip install nea_tools nest_asyncio"
            )
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None   # neaspec stream.Stream context-manager object
        self._context = None  # neaspec.context module reference

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, host: str = "nea-server") -> None:
        self._loop = asyncio.get_event_loop()
        nest_asyncio.apply(self._loop)
        self._loop.run_until_complete(
            nea_tools.connect(host, fingerprint=None, path_to_dll="")
        )
        # Import neaspec only after connect() because the module is injected
        # by nea_tools at runtime.
        import neaspec  # noqa: PLC0415 — runtime import after connect
        import neaspec.stream as stream_module  # noqa: PLC0415

        self._context = neaspec.context
        self._stream_module = stream_module
        self._stream_ctx = stream_module.Stream()
        self._stream = self._stream_ctx.__enter__()
        self._connected = True

    def disconnect(self) -> None:
        if self._stream_ctx is not None:
            try:
                self._stream_ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
            self._stream_ctx = None
        if self._connected:
            try:
                nea_tools.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def goto_xy_nm(self, x_nm: float, y_nm: float) -> None:
        self._require_connected()
        # Use the Sample motor via context-manager to ensure cleanup.
        # ActiveMotorGotoXyzAsync expects a System.Point3D equivalent;
        # nea_tools exposes it as a Python tuple/named structure.
        loop = self._loop
        with _nea_motors.Sample() as sample:
            sample.activate()
            result = loop.run_until_complete(
                sample.ActiveMotorGotoXyzAsync((x_nm, 0.0, 0.0))
            )
        if not result:
            raise StageError(f"Motor failed to reach target ({x_nm}, {0.0})")
        # Move Y separately — or use combined move if supported.
        with _nea_motors.Sample() as sample:
            sample.activate()
            result = loop.run_until_complete(
                sample.ActiveMotorGotoXyzAsync((x_nm, y_nm, 0.0))
            )
        if not result:
            raise StageError(f"Motor failed to reach target ({x_nm}, {y_nm})")

    def read_xyz_nm(self) -> tuple[float, float, float]:
        self._require_connected()
        loop = self._loop
        pos = loop.run_until_complete(
            self._context.Microscope.GetActiveMotorDistanceToReferenceXyz()
        )
        return float(pos.X), float(pos.Y), float(pos.Z)

    # ------------------------------------------------------------------
    # Signal readout
    # ------------------------------------------------------------------

    def read_sample(self, t_s: float) -> SnomSample:
        self._require_connected()
        mic = self._context.Microscope.Py
        s = self._stream
        xyz = self.read_xyz_nm()

        o_amp = np.empty(_N_HARMONICS, dtype=np.float64)
        o_phase = np.empty(_N_HARMONICS, dtype=np.float64)
        m_amp = np.empty(_N_HARMONICS, dtype=np.float64)
        m_phase = np.empty(_N_HARMONICS, dtype=np.float64)

        for h in range(_N_HARMONICS):
            o_amp[h] = float(mic.OpticalAmplitude(h))
            m_amp[h] = float(mic.MechanicalAmplitude(h))
            try:
                o_phase[h] = float(s.data[f"O{h}P"][-1])
            except Exception:  # noqa: BLE001
                o_phase[h] = np.nan
            try:
                m_phase[h] = float(s.data[f"M{h}P"][-1])
            except Exception:  # noqa: BLE001
                m_phase[h] = np.nan

        return SnomSample(
            t_s=t_s,
            xyz_nm=xyz,
            o_amp=o_amp,
            o_phase=o_phase,
            m_amp=m_amp,
            m_phase=m_phase,
        )

    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise StageError("SNOM stage backend is not connected.")
