"""Read a one-shot power reading from a Thorlabs PM100D over VISA/USB.

Mirrors io.rp_state.load_rp_metadata: best-effort, flat prefixed scalar dict,
never raises — a missing/flaky powermeter must not block a scan.
"""

from __future__ import annotations

from typing import Any

_THORLABS_USB_VID = "0x1313"


def read_power_metadata(prefix: str = "power_") -> dict[str, Any] | None:
    """Auto-detect a PM100D over USB and return one power + wavelength reading.

    Returns ``{f"{prefix}w": <power_w>, f"{prefix}wavelength_nm": <wavelength_nm>}``,
    or ``None`` if pyvisa/ThorlabsPM100 aren't installed, no PM100D is found, or
    the read fails for any reason.
    """
    try:
        import pyvisa  # noqa: PLC0415
        from ThorlabsPM100 import ThorlabsPM100  # noqa: PLC0415

        rm = pyvisa.ResourceManager()
        resource_name = next(
            (r for r in rm.list_resources("USB?*") if _THORLABS_USB_VID in r.upper()),
            None,
        )
        if resource_name is None:
            return None

        inst = rm.open_resource(resource_name)
        try:
            power_meter = ThorlabsPM100(inst=inst)
            power_w = float(power_meter.read)
            wavelength_nm = float(power_meter.sense.correction.wavelength)
        finally:
            inst.close()

        return {
            f"{prefix}w": power_w,
            f"{prefix}wavelength_nm": wavelength_nm,
        }
    except Exception:  # noqa: BLE001 — best-effort read, never block the scan
        return None
