from __future__ import annotations

import json
from pathlib import Path

import pytest

from idus420_gui.io.rp_state import (
    RPStateError,
    default_rp_state_path,
    load_rp_metadata,
    load_rp_state,
    rp_state_changed,
)


def _state(**overrides):
    data = {
        "schema_version": 1,
        "source": "redpitaya_ttl_frequency_divider",
        "connected": True,
        "hardware_confirmed": True,
        "sequence": 12,
        "updated_at": 100.0,
        "mode": "pulse",
        "output_mode": "modulated",
        "period_stable": True,
        "trigger_frequency_hz": 500.0,
        "frequency_shift_hz": -37.0,
        "expected_peak_hz": 37.0,
        "input_frequency_hz": 1_000_000.0,
        "output_frequency_hz": 999_963.0,
        "control": 1,
        "harmonic_n": 1,
        "trig_phase_step": 1125899915,
        "phase_step_offset": -83316594,
        "phase_step_base": 2251799833685,
        "phase_step": 2251716517091,
        "osc_half_period": 0,
    }
    data.update(overrides)
    return data


def _write(path: Path, **overrides) -> Path:
    path.write_text(json.dumps(_state(**overrides)), encoding="utf-8")
    return path


def test_default_state_path_uses_redpitaya_application_state_directory() -> None:
    path = default_rp_state_path()

    assert path.name == "rp_state.json"
    assert path.parent.name == "RedPitayaTTLFrequencyDivider"


def test_load_ready_state_parses_confirmed_values(tmp_path: Path) -> None:
    state = load_rp_state(_write(tmp_path / "rp_state.json"), max_age_s=3.0, now=102.0)

    assert state.trigger_frequency_hz == 500.0
    assert state.frequency_shift_hz == -37.0
    assert state.expected_peak_hz == 37.0
    assert state.configuration_signature[-2:] == (1125899915, -83316594)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema"),
        ({"connected": False}, "disconnected"),
        ({"hardware_confirmed": False}, "not confirmed"),
        ({"output_mode": "off"}, "not modulated"),
        ({"period_stable": False}, "not stable"),
        ({"trigger_frequency_hz": 0.0}, "trigger is off"),
        ({"updated_at": 90.0}, "stale"),
    ],
)
def test_load_ready_state_rejects_unsafe_state(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    path = _write(tmp_path / "rp_state.json", **overrides)

    with pytest.raises(RPStateError, match=message):
        load_rp_state(path, max_age_s=3.0, now=102.0)


def test_load_state_rejects_missing_required_field(tmp_path: Path) -> None:
    data = _state()
    del data["trig_phase_step"]
    path = tmp_path / "rp_state.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RPStateError, match="trig_phase_step"):
        load_rp_state(path)


def test_configuration_change_ignores_heartbeat_fields(tmp_path: Path) -> None:
    start = load_rp_state(_write(tmp_path / "start.json"))
    same = load_rp_state(
        _write(tmp_path / "same.json", sequence=99, updated_at=200.0)
    )
    changed = load_rp_state(
        _write(tmp_path / "changed.json", trig_phase_step=1125899916)
    )

    assert not rp_state_changed(start, same)
    assert rp_state_changed(start, changed)


def test_metadata_uses_confirmed_contract_fields(tmp_path: Path) -> None:
    path = _write(tmp_path / "rp_state.json")

    metadata = load_rp_metadata(path)

    assert metadata is not None
    assert metadata["rp_schema_version"] == 1
    assert metadata["rp_trigger_frequency_hz"] == 500.0
    assert metadata["rp_expected_peak_hz"] == 37.0
