"""Real SNOM stage backend using nea_tools and neaspec SDK."""

from __future__ import annotations

import queue
import threading

try:
    import asyncio
    import time
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
_MOVE_TIMEOUT_S = 300.0


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
        # Always create a fresh event loop to avoid reusing a closed or
        # dirty loop from a previous scan session.
        try:
            old_loop = asyncio.get_event_loop()
            if not old_loop.is_closed():
                old_loop.close()
        except Exception:  # noqa: BLE001
            pass
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        nest_asyncio.apply(self._loop)
        self._loop.run_until_complete(
            nea_tools.connect(host, fingerprint=None, path_to_dll="")
        )
        import neaspec  # noqa: PLC0415
        import Nea.Client.SharedDefinitions as nea  # noqa: PLC0415
        from nea_tools.microscope import stream  # noqa: PLC0415

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
        self,
        x_nm: float,
        y_nm: float,
        speed_um_s: float = _DEFAULT_SPEED_UM_S,
        timeout_s: float = _MOVE_TIMEOUT_S,
    ) -> None:
        """Move tip to (x_nm, y_nm) and block until the move completes."""
        self._require_connected()
        nea = self._nea
        ctx = self._context

        x_um = x_nm / 1000.0
        y_um = y_nm / 1000.0

        # Start True so that if Moved fires before the polling loop starts we
        # don't stall.  Only the Moved callback clears the flag; removing the
        # Moving callback eliminates the race where Moved fires before Moving.
        do_wait = [True]

        def on_tip_position_moved(sender, args):  # noqa: ANN001
            do_wait[0] = False

        ctx.Logic.TipPositionMoved += on_tip_position_moved
        try:
            move_args = nea.MoveTipPositionArgs(
                nea.Geometry.Point2D(x_um, y_um),
                speed_um_s / 1000.0,
            )
            ctx.Logic.MoveTipPosition.Execute(move_args)
            deadline = time.monotonic() + timeout_s
            sleep(_MOVE_POLL_S)
            while do_wait[0]:
                if time.monotonic() >= deadline:
                    raise StageError(
                        f"goto_xy_nm timed out after {timeout_s:.0f} s"
                        f" (target: x={x_nm:.0f} nm, y={y_nm:.0f} nm)"
                    )
                sleep(_MOVE_POLL_S)
            sleep(0.2)
        finally:
            ctx.Logic.TipPositionMoved -= on_tip_position_moved

    def read_xyz_nm(self) -> tuple[float, float, float]:
        self._require_connected()
        with self._stream_module.Stream() as s:
            sleep(0.025)  # wait for the first ~20 ms batch to arrive
            x_nm = float(s.data["AveragedX"][-1]) * 1000.0
            y_nm = float(s.data["AveragedY"][-1]) * 1000.0
            z_nm = float(s.data["AveragedZ"][-1]) * 1000.0
        return (x_nm, y_nm, z_nm)

    # ------------------------------------------------------------------
    # Signal readout
    # ------------------------------------------------------------------

    def read_optical_amplitude(self, harmonic: int) -> float:
        self._require_connected()
        return float(self._context.Microscope.Py.OpticalAmplitude[harmonic])

    def read_mechanical_amplitude(self, harmonic: int) -> float:
        self._require_connected()
        return float(self._context.Microscope.Py.MechanicalAmplitude[harmonic])

    def read_sample(
        self, t_s: float, t_integ_s: float = 0.05, n_avg: int = 1
    ) -> SnomSample:
        self._require_connected()

        keys = (
            [f"O{h}A" for h in range(_N_HARMONICS)]
            + [f"O{h}P" for h in range(_N_HARMONICS)]
            + [f"M{h}A" for h in range(_N_HARMONICS)]
            + [f"M{h}P" for h in range(_N_HARMONICS)]
            + ["AveragedX", "AveragedY", "AveragedZ"]
        )
        totals: dict[str, float] = {k: 0.0 for k in keys}
        counts: dict[str, int] = {k: 0 for k in keys}

        for _ in range(n_avg):
            win_sum: dict[str, float] = {k: 0.0 for k in keys}
            win_cnt: dict[str, int] = {k: 0 for k in keys}

            with self._stream_module.Stream() as s:
                t_end = time.time() + t_integ_s
                while time.time() < t_end:
                    for k in keys:
                        try:
                            temp = float(s.data[k][-1])
                            win_sum[k] += temp
                            win_cnt[k] += 1
                        except Exception:  # noqa: BLE001
                            pass
                    time.sleep(0.02)

            for k in keys:
                if win_cnt[k]:
                    totals[k] += win_sum[k] / win_cnt[k]
                    counts[k] += 1

        def _get(k: str) -> float:
            return totals[k] / counts[k] if counts[k] else float("nan")

        x_nm = _get("AveragedX") * 1000.0
        y_nm = _get("AveragedY") * 1000.0
        z_nm = _get("AveragedZ") * 1000.0

        n = _N_HARMONICS
        return SnomSample(
            t_s=t_s,
            xyz_nm=(x_nm, y_nm, z_nm),
            o_amp=np.array([_get(f"O{h}A") for h in range(n)]),
            o_phase=np.array([_get(f"O{h}P") for h in range(n)]),
            m_amp=np.array([_get(f"M{h}A") for h in range(n)]),
            m_phase=np.array([_get(f"M{h}P") for h in range(n)]),
        )

    def stream_continuous(
        self,
        stop_event: threading.Event,
        frame_event: threading.Semaphore,
        out_queue: queue.Queue,
        t0_scan: float,
    ) -> None:
        """Run in a background thread: one Stream for the whole point acquisition.

        Polls every 20 ms. Each time frame_event is released (one camera frame
        arrived), flush accumulated SNOM polls into a SnomSample and put it in
        out_queue.  Using a Semaphore instead of an Event counts each frame
        individually so rapid back-to-back frames are not collapsed into one.
        Stops when stop_event is set.
        """

        keys = (
            [f"O{h}A" for h in range(_N_HARMONICS)]
            + [f"O{h}P" for h in range(_N_HARMONICS)]
            + [f"M{h}A" for h in range(_N_HARMONICS)]
            + [f"M{h}P" for h in range(_N_HARMONICS)]
            + ["AveragedX", "AveragedY", "AveragedZ"]
        )

        def _flush(win_sum: dict, win_cnt: dict, t_s: float) -> SnomSample:
            def _get(k: str) -> float:
                return win_sum[k] / win_cnt[k] if win_cnt[k] else float("nan")
            n = _N_HARMONICS
            return SnomSample(
                t_s=t_s,
                xyz_nm=(
                    _get("AveragedX") * 1000.0,
                    _get("AveragedY") * 1000.0,
                    _get("AveragedZ") * 1000.0,
                ),
                o_amp=np.array([_get(f"O{h}A") for h in range(n)]),
                o_phase=np.array([_get(f"O{h}P") for h in range(n)]),
                m_amp=np.array([_get(f"M{h}A") for h in range(n)]),
                m_phase=np.array([_get(f"M{h}P") for h in range(n)]),
            )

        win_sum: dict[str, float] = {k: 0.0 for k in keys}
        win_cnt: dict[str, int] = {k: 0 for k in keys}

        with self._stream_module.Stream() as s:
            while not stop_event.is_set():
                # Poll one 20 ms tick
                for k in keys:
                    try:
                        temp = float(s.data[k][-1])
                        win_sum[k] += temp
                        win_cnt[k] += 1
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(0.02)

                # Drain all frame notifications that arrived during this poll tick.
                while frame_event.acquire(blocking=False):
                    t_s = time.monotonic() - t0_scan
                    out_queue.put(_flush(win_sum, win_cnt, t_s))
                    win_sum = {k: 0.0 for k in keys}
                    win_cnt = {k: 0 for k in keys}

    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise StageError("SNOM stage backend is not connected.")
