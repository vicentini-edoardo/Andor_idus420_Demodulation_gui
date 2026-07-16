"""Read the FPGA-confirmed Red Pitaya cross-application state."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RPStateError(ValueError):
    """The Red Pitaya state file is unavailable, invalid, or unsafe to use."""


@dataclass(frozen=True)
class RedPitayaState:
    raw: dict[str, Any]
    schema_version: int
    connected: bool
    hardware_confirmed: bool
    sequence: int
    updated_at: float
    mode: str
    output_mode: str
    period_stable: bool
    trigger_frequency_hz: float
    frequency_shift_hz: float
    expected_peak_hz: float
    control: int
    harmonic_n: int
    trig_phase_step: int
    phase_step_offset: int
    osc_half_period: int

    @property
    def configuration_signature(self) -> tuple[Any, ...]:
        """Register-backed configuration, excluding heartbeat fields."""
        return (
            self.mode,
            self.output_mode,
            self.control,
            self.harmonic_n,
            self.raw.get("width_cycles"),
            self.osc_half_period,
            self.raw.get("osc_phase_preload"),
            self.trig_phase_step,
            self.phase_step_offset,
        )

    def metadata(self, prefix: str = "rp_") -> dict[str, Any]:
        return {
            f"{prefix}{key}": value
            for key, value in self.raw.items()
            if value is None or isinstance(value, (bool, int, float, str))
        }

    def require_ready(self, max_age_s: float, now: float | None = None) -> None:
        if not self.connected:
            raise RPStateError("Red Pitaya is disconnected.")
        if not self.hardware_confirmed:
            raise RPStateError("Red Pitaya state is not confirmed by hardware.")
        if self.output_mode != "modulated":
            raise RPStateError(f"Red Pitaya output is not modulated ({self.output_mode}).")
        if not self.period_stable:
            raise RPStateError("Red Pitaya input period is not stable.")
        if self.trigger_frequency_hz <= 0:
            raise RPStateError("Red Pitaya DIO2 trigger is off.")
        age_s = (time.time() if now is None else float(now)) - self.updated_at
        if age_s > max_age_s:
            raise RPStateError(
                f"Red Pitaya state is stale ({age_s:.1f} s old; limit {max_age_s:g} s)."
            )


_REQUIRED = (
    "schema_version",
    "connected",
    "hardware_confirmed",
    "sequence",
    "updated_at",
    "mode",
    "output_mode",
    "period_stable",
    "trigger_frequency_hz",
    "frequency_shift_hz",
    "expected_peak_hz",
    "control",
    "harmonic_n",
    "trig_phase_step",
    "phase_step_offset",
    "osc_half_period",
)


def default_rp_state_path() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "RedPitayaTTLFrequencyDivider" / "rp_state.json"


def load_rp_state(
    path: str | Path | None = None,
    *,
    max_age_s: float | None = None,
    now: float | None = None,
) -> RedPitayaState:
    state_path = Path(path) if path is not None else default_rp_state_path()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RPStateError(f"Could not read Red Pitaya state from {state_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RPStateError("Red Pitaya state must be a JSON object.")
    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        raise RPStateError(f"Red Pitaya state is missing {', '.join(missing)}.")
    if data["schema_version"] != 1:
        raise RPStateError(f"Unsupported Red Pitaya state schema {data['schema_version']!r}.")
    if data["mode"] not in {"pulse", "harmonic", "osc"}:
        raise RPStateError(f"Invalid Red Pitaya mode {data['mode']!r}.")
    if data["output_mode"] not in {"off", "modulated", "on"}:
        raise RPStateError(f"Invalid Red Pitaya output mode {data['output_mode']!r}.")
    for key in ("connected", "hardware_confirmed", "period_stable"):
        if not isinstance(data[key], bool):
            raise RPStateError(f"Red Pitaya field {key} must be boolean.")
    for key in (
        "updated_at",
        "trigger_frequency_hz",
        "frequency_shift_hz",
        "expected_peak_hz",
    ):
        if isinstance(data[key], bool) or not isinstance(data[key], (int, float)):
            raise RPStateError(f"Red Pitaya field {key} must be numeric.")
        if not math.isfinite(float(data[key])):
            raise RPStateError(f"Red Pitaya field {key} must be finite.")
    for key in (
        "sequence",
        "control",
        "harmonic_n",
        "trig_phase_step",
        "phase_step_offset",
        "osc_half_period",
    ):
        if isinstance(data[key], bool) or not isinstance(data[key], int):
            raise RPStateError(f"Red Pitaya field {key} must be an integer.")

    state = RedPitayaState(
        raw=dict(data),
        schema_version=1,
        connected=data["connected"],
        hardware_confirmed=data["hardware_confirmed"],
        sequence=data["sequence"],
        updated_at=float(data["updated_at"]),
        mode=data["mode"],
        output_mode=data["output_mode"],
        period_stable=data["period_stable"],
        trigger_frequency_hz=float(data["trigger_frequency_hz"]),
        frequency_shift_hz=float(data["frequency_shift_hz"]),
        expected_peak_hz=float(data["expected_peak_hz"]),
        control=data["control"],
        harmonic_n=data["harmonic_n"],
        trig_phase_step=data["trig_phase_step"],
        phase_step_offset=data["phase_step_offset"],
        osc_half_period=data["osc_half_period"],
    )
    if max_age_s is not None:
        state.require_ready(float(max_age_s), now)
    return state


def rp_state_changed(start: RedPitayaState, end: RedPitayaState) -> bool:
    return start.configuration_signature != end.configuration_signature


def rp_run_metadata(
    start: RedPitayaState, end: RedPitayaState | None
) -> dict[str, Any]:
    metadata = start.metadata()
    if end is not None:
        metadata.update(end.metadata("rp_end_"))
    metadata["rp_state_changed_during_run"] = end is None or rp_state_changed(start, end)
    return metadata


def load_rp_metadata(path: str | Path) -> dict[str, Any] | None:
    """Compatibility wrapper returning prefixed scalar metadata."""
    try:
        return load_rp_state(path).metadata()
    except RPStateError:
        return None
