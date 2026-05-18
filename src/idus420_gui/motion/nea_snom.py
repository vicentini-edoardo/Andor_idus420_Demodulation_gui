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
_DEFAULT_SPEED_UM_S = 0.2
_MOVE_POLL_S = 0.1


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
        self._context = None
        self._nea = None
        self._stream_module = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, host: str = "nea-server") -> None:
        self._loop = asyncio.get_event_loop()
        nest_asyncio.apply(self._loop)
        self._loop.run_until_complete(
            nea_tools.connect(host, fingerprint=None, path_to_dll="")
        )
        import neaspec  # noqa: PLC0415
        import neaspec.stream as stream  # noqa: PLC0415
        import Nea.Client.SharedDefinitions as nea  # noqa: PLC0415

        self._context = neaspec.context
        self._nea = nea
        self._stream_module = stream
        self._connected = True

    def disconnect(self) -> None:
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

    def goto_xy_nm(
        self, x_nm: float, y_nm: float, speed_um_s: float = _DEFAULT_SPEED_UM_S
    ) -> None:
        """Move tip to (x_nm, y_nm) and block until the move completes."""
        self._require_connected()
        nea = self._nea
        ctx = self._context

        x_um = x_nm / 1000.0
        y_um = y_nm / 1000.0

        do_wait = [False]

        def on_tip_position_moved(sender, args):  # noqa: ANN001
            do_wait[0] = False

        def on_tip_position_moving(sender, args):  # noqa: ANN001
            do_wait[0] = True

        ctx.Logic.TipPositionMoved += on_tip_position_moved
        ctx.Logic.TipPositionMoving += on_tip_position_moving
        try:
            move_args = nea.MoveTipPositionArgs(
                nea.Geometry.Point2D(x_um, y_um),
                speed_um_s / 1000.0,
            )
            ctx.Logic.MoveTipPosition.Execute(move_args)
            sleep(_MOVE_POLL_S)
            while do_wait[0]:
                sleep(_MOVE_POLL_S)
            sleep(0.2)
        finally:
            ctx.Logic.TipPositionMoved -= on_tip_position_moved
            ctx.Logic.TipPositionMoving -= on_tip_position_moving

    def read_xyz_nm(self) -> tuple[float, float, float]:
        raise NotImplementedError("Ask Vincent for the position readback API.")

    # ------------------------------------------------------------------
    # Signal readout
    # ------------------------------------------------------------------

    def read_optical_amplitude(self, harmonic: int) -> float:
        self._require_connected()
        return float(self._context.Microscope.Py.OpticalAmplitude(harmonic))

    def read_mechanical_amplitude(self, harmonic: int) -> float:
        self._require_connected()
        return float(self._context.Microscope.Py.MechanicalAmplitude(harmonic))

    def read_sample(self, t_s: float) -> SnomSample:
        self._require_connected()
        o_amp = np.empty(_N_HARMONICS, dtype=np.float64)
        o_phase = np.empty(_N_HARMONICS, dtype=np.float64)
        m_amp = np.empty(_N_HARMONICS, dtype=np.float64)
        m_phase = np.empty(_N_HARMONICS, dtype=np.float64)

        with self._stream_module.Stream() as s:
            for h in range(_N_HARMONICS):
                o_amp[h] = self.read_optical_amplitude(h)
                m_amp[h] = self.read_mechanical_amplitude(h)
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
            xyz_nm=(np.nan, np.nan, np.nan),
            o_amp=o_amp,
            o_phase=o_phase,
            m_amp=m_amp,
            m_phase=m_phase,
        )

    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise StageError("SNOM stage backend is not connected.")
