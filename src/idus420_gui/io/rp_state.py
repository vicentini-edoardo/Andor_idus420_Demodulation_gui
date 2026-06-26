"""Load Red Pitaya state file and return prefixed metadata dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PREFIX = "rp_"
_FIELDS = (
    "mode",
    "output_mode",
    "pulse_freq_shift_hz",
    "harmonic_freq_shift_hz",
    "duty_cycle_pct",
    "harmonic_n",
    "updated_at",
)


def load_rp_metadata(path: str | Path) -> dict[str, Any] | None:
    """Read rp_state.json and return flat dict with rp_ prefix.

    Returns None if the file is missing or malformed.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {f"{_PREFIX}{k}": data[k] for k in _FIELDS if k in data}
