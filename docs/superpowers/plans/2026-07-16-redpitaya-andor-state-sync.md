# Red Pitaya–Andor State Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Andor application derive its external-trigger sample rate and FFT target from fresh, FPGA-confirmed Red Pitaya state and preserve start/end state with every acquisition.

**Architecture:** Keep the existing atomic JSON file as the only process boundary. The Red Pitaya GUI remains the hardware authority and publishes register-derived state after acknowledgements and polls; the Andor GUI reads, validates, synchronizes, and snapshots that state without gaining SSH ownership.

**Tech Stack:** Python 3.10+, PySide6, PyQt6, standard-library JSON/path/time, pytest/unittest.

---

### Task 1: Publish confirmed Red Pitaya state

**Files:**
- Modify: `/Users/edoardovicentini/.config/superpowers/worktrees/Redpitaya_TTL_frequency_divider/rp-state-contract/redpitaya_combined_gui_qt.py`
- Test: `/Users/edoardovicentini/.config/superpowers/worktrees/Redpitaya_TTL_frequency_divider/rp-state-contract/tests/test_gui_layout.py`

- [ ] **Step 1: Write failing state-contract tests**

Add tests that pass a representative FPGA status dictionary to `_confirmed_state` and assert register-derived `trigger_frequency_hz`, `frequency_shift_hz`, `expected_peak_hz`, mode, output mode, and confirmation flags. Add an oscillating-mode case asserting that the expected peak comes from `osc_half_period`, plus a disconnected-state case.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
QT_QPA_PLATFORM=offscreen /Users/edoardovicentini/Documents/GitHub/Redpitaya_TTL_frequency_divider/.venv/bin/python -m unittest tests.test_gui_layout -v
```

Expected: failure because `_confirmed_state` does not exist.

- [ ] **Step 3: Implement the minimal producer contract**

Add `_confirmed_state(status, connected, sequence, now)` which derives values using existing `phase_to_hz`, publishes schema version 1, and reports disconnected/unconfirmed state without a status. Connect `SshBackend.sig_status` to a MainWindow slot that increments the sequence and atomically writes confirmed state. Publish disconnected state on startup and disconnect. Stop rewriting authoritative state from pending GUI edits.

- [ ] **Step 4: Verify producer tests pass**

Run the command from Step 2 and expect all GUI tests to pass.

### Task 2: Parse and validate the contract in Andor

**Files:**
- Modify: `src/idus420_gui/io/rp_state.py`
- Create: `tests/test_rp_state.py`

- [ ] **Step 1: Write failing parser tests**

Cover the OS-default state path, valid parsing, malformed/missing fields, disconnected/unconfirmed/stale states, configuration signatures, and prefixed metadata.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src PYTEST_QT_API=pyqt6 python -m pytest tests/test_rp_state.py -q
```

Expected: import failures for the new parser API.

- [ ] **Step 3: Implement the minimal consumer API**

Add `RedPitayaState`, `RPStateError`, `default_rp_state_path`, `load_rp_state`, readiness validation, metadata flattening, and configuration comparison. Preserve `load_rp_metadata` as a compatibility wrapper.

- [ ] **Step 4: Verify parser tests pass**

Run the command from Step 2 and expect all parser tests to pass.

### Task 3: Synchronize Andor runtime settings

**Files:**
- Modify: `src/idus420_gui/gui/panel_demod.py`
- Modify: `src/idus420_gui/gui/panel_live.py`
- Modify: `src/idus420_gui/gui/main_window.py`
- Test: `tests/test_demodulation.py`
- Test: `tests/test_live_panel.py`

- [ ] **Step 1: Write failing synchronization and preflight tests**

Assert that a fresh confirmed state sets the demodulation trigger and expected peak, emits the trigger for the Live panel, rejects stale/unconfirmed state, rejects a search band above Nyquist, and rejects trigger rates above camera timing limits.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src PYTEST_QT_API=pyqt6 python -m pytest tests/test_demodulation.py tests/test_live_panel.py -q
```

Expected: failures because synchronization controls and preflight methods do not exist.

- [ ] **Step 3: Implement synchronization**

Add a compact Red Pitaya sync section to the demodulation panel with enable, path override, browse, sync-now, and status controls. Poll the local JSON only while idle, copy confirmed trigger/peak values into existing controls, run preflight before acquisition, and propagate the trigger to Live through MainWindow.

- [ ] **Step 4: Verify synchronization tests pass**

Run the command from Step 2 and expect all selected tests to pass.

### Task 4: Snapshot start/end state in acquisition and scan metadata

**Files:**
- Modify: `src/idus420_gui/gui/panel_acquire.py`
- Modify: `src/idus420_gui/gui/panel_scan.py`
- Test: `tests/test_save.py`
- Test: `tests/test_scan_panel.py`

- [ ] **Step 1: Write failing snapshot tests**

Assert that state is captured before worker start, saved settings are the frozen start settings, and final metadata reports whether the confirmed RP configuration changed during a run.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src PYTEST_QT_API=pyqt6 python -m pytest tests/test_save.py tests/test_scan_panel.py -q
```

Expected: failures because current code reads RP state only after acquisition.

- [ ] **Step 3: Implement start/end snapshots**

Default existing RP path fields to the live application-state location. Capture and validate state at start, store prefixed start metadata and frozen demodulation settings, read end state only for comparison, and add `rp_state_changed_during_run` plus end-state fields.

- [ ] **Step 4: Verify snapshot tests pass**

Run the command from Step 2 and expect all selected tests to pass.

### Task 5: Full verification and self-review

**Files:**
- Modify if needed: both repositories' README files

- [ ] **Step 1: Run complete relevant suites**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src PYTEST_QT_API=pyqt6 python -m pytest -q
QT_QPA_PLATFORM=offscreen /Users/edoardovicentini/Documents/GitHub/Redpitaya_TTL_frequency_divider/.venv/bin/python -m unittest tests.test_gui_layout tests.test_rp_math -q
```

Expected: Andor suite passes with its existing hardware skip; Red Pitaya GUI/math suites pass.

- [ ] **Step 2: Run lint and compile checks**

```bash
ruff check src tests
/Users/edoardovicentini/Documents/GitHub/Redpitaya_TTL_frequency_divider/.venv/bin/python -m py_compile redpitaya_combined_gui_qt.py
```

Expected: both commands exit zero.

- [ ] **Step 3: Review diffs against Option 1**

Confirm no SSH ownership was added to Andor, no dependency was added, state comes from FPGA registers, acquisition snapshots occur at start, and stale/Nyquist/camera-rate checks are enforced.
