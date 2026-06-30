# Camera Shutter Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore working iDus 420 camera shutter open/close commands.

**Architecture:** Keep shutter control inside `AndorIDusBackend.configure()`, where all camera settings already converge. Replace the unsupported `SetShutterEx` path with the exact `SetShutter` call used before the regression and retain one focused SDK-call test.

**Tech Stack:** Python 3.10+, pytest, Andor SDK2 wrapper

---

### Task 1: Restore the camera shutter SDK call

**Files:**
- Modify: `tests/test_andor_backend.py`
- Modify: `src/idus420_gui/camera/andor.py`

- [ ] **Step 1: Replace the incorrect shutter test with a failing regression test**

Remove imports of `_DEFAULT_SHUTTER_TRANSFER_MS` and `_SHUTTER_TTL_TYPE`, remove
`_SdkWithoutShutterEx`, and replace the two `_apply_shutter` tests with:

```python
def test_apply_shutter_uses_known_working_camera_command() -> None:
    sdk = _SdkWithShutterEx()
    backend = _shutter_backend(sdk)

    backend._apply_shutter(2)  # 2 == "permanently closed"

    assert sdk.shutter_calls == [(1, 2, 0, 0)]
    assert sdk.shutter_ex_calls == []
```

- [ ] **Step 2: Run the regression test and verify it fails**

Run:

```bash
pytest tests/test_andor_backend.py::test_apply_shutter_uses_known_working_camera_command -v
```

Expected: FAIL because the current implementation records `SetShutterEx` and
does not record `SetShutter`.

- [ ] **Step 3: Restore the minimal production implementation**

Delete `_SHUTTER_TTL_TYPE`, `_DEFAULT_SHUTTER_TRANSFER_MS`, and
`_shutter_transfer_times()`. Replace `_apply_shutter()` with:

```python
def _apply_shutter(self, mode_code: int) -> None:
    self._check(
        self._sdk.SetShutter(1, mode_code, 0, 0),
        "SetShutter",
    )
```

- [ ] **Step 4: Update the configure call-sequence assertions**

In `test_configure_drives_shutter_before_timings_query`, require `SetShutter`,
reject `SetShutterEx`, and assert:

```python
assert names.index("SetShutter") < names.index("GetAcquisitionTimings")
assert sdk.args_for("SetShutter") == (1, 1, 0, 0)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_andor_backend.py -v
```

Expected: all backend tests PASS.

- [ ] **Step 6: Run project verification**

Run:

```bash
pytest
ruff check .
mypy
```

Expected: all tests PASS and both static checks exit successfully.

- [ ] **Step 7: Commit the fix**

```bash
git add src/idus420_gui/camera/andor.py tests/test_andor_backend.py
git commit -m "fix: restore iDus camera shutter control"
```
