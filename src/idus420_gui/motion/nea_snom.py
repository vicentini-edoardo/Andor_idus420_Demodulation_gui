"""Real SNOM stage backend using nea_tools and neaspec SDK."""

from __future__ import annotations

try:
    import asyncio
    from time import sleep

    import nest_asyncio
    import nea_tools

    NEA_TOOLS_AVAILABLE = True
except ImportError:
    NEA_TOOLS_AVAILABLE = False

import numpy as np

from idus420_gui.motion.base import SnomSample, StageBackend, StageError

_N_HARMONICS = 6
_DEFAULT_SPEED_UM_S = 0.2   # µm/s tip speed
_MOVE_POLL_S = 0.1          # seconds between do_wait polls


class NeaSnomBackend(StageBackend):
    """Stage backend that wraps nea_tools / neaspec for SNOM tip motion."""

    def __init__(self) -> None:
        if not NEA_TOOLS_AVAILABLE:
            raise StageError(
                "nea_tools / nest_asyncio are not installed. "
                "Install with: pip install nea_tools nest_asyncio"
            )
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._context = None   # neaspec.context, injected by nea_tools after connect
        self._nea = None       # Nea.Client.SharedDefinitions module
        self._stream = None    # neaspec.stream.Stream object (optional)
        self._stream_ctx = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, host: str = "nea-server") -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        nest_asyncio.apply(self._loop)
        self._loop.run_until_complete(
            nea_tools.connect(host, fingerprint=None, path_to_dll="")
        )
        # neaspec is injected at runtime by nea_tools after connect()
        import neaspec  # noqa: PLC0415
        import Nea.Client.SharedDefinitions as nea  # noqa: PLC0415

        self._context = neaspec.context
        self._nea = nea

        # neaspec.stream is optional (not present in all SDK versions)
        try:
            import neaspec.stream as stream_module  # noqa: PLC0415
            self._stream_ctx = stream_module.Stream()
            self._stream = self._stream_ctx.__enter__()
        except (ImportError, AttributeError):
            self._stream_ctx = None
            self._stream = None

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
    # Motion  (uses context.Logic.MoveTipPosition per developer API)
    # ------------------------------------------------------------------

    def goto_xy_nm(self, x_nm: float, y_nm: float, speed_um_s: float = _DEFAULT_SPEED_UM_S) -> None:
        """Move tip to (x_nm, y_nm) and block until the move completes."""
        self._require_connected()
        nea = self._nea
        ctx = self._context

        # Coordinates are in µm for MoveTipPositionArgs
        x_um = x_nm / 1000.0
        y_um = y_nm / 1000.0

        do_wait: list[bool] = [False]  # mutable container so closure can write it

        def on_moved(sender, args):  # noqa: ANN001
            do_wait[0] = False

        def on_moving(sender, args):  # noqa: ANN001
            do_wait[0] = True

        ctx.Logic.TipPositionMoved += on_moved
        ctx.Logic.TipPositionMoving += on_moving
        try:
            args = nea.MoveTipPositionArgs(
                nea.Geometry.Point2D(x_um, y_um),
                speed_um_s / 1000.0,   # SDK expects µm/ms
            )
            ctx.Logic.MoveTipPosition.Execute(args)
            sleep(_MOVE_POLL_S)
            while do_wait[0]:
                sleep(_MOVE_POLL_S)
            sleep(0.2)   # settling time recommended by developer
        finally:
            ctx.Logic.TipPositionMoved -= on_moved
            ctx.Logic.TipPositionMoving -= on_moving

    def read_xyz_nm(self) -> tuple[float, float, float]:
        self._require_connected()
        pos = self._loop.run_until_complete(
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
                o_phase[h] = float(s.data[f"O{h}P"][-1]) if s is not None else np.nan
            except Exception:  # noqa: BLE001
                o_phase[h] = np.nan
            try:
                m_phase[h] = float(s.data[f"M{h}P"][-1]) if s is not None else np.nan
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
