# Camera Shutter Regression Fix

## Problem

Camera shutter open/close worked before commit `da7b3f3`, which replaced the
iDus 420 call `SetShutter(1, mode, 0, 0)` with `SetShutterEx(...)`. The current
hardware still uses camera shutter control, not Shamrock shutter control.

## Design

Restore the last known-working SDK call in `AndorIDusBackend`: call
`SetShutter(1, mode, 0, 0)` when applying camera configuration. Remove the
unused `SetShutterEx` transfer-time helper and constants introduced by the
regression.

Update the focused backend regression test to require `SetShutter` with the
selected open/closed mode and to reject `SetShutterEx`.

## Scope

No GUI, Shamrock, acquisition, or configuration changes. Verification consists
of the focused shutter/backend tests followed by the project test suite.
