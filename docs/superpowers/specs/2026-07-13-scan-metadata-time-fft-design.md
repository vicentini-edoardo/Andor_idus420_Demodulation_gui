# Scan Metadata, Time Estimate, and FFT Fixes

## Scope

Fix three scan-mode inconsistencies without changing scan acquisition behavior:

1. Save `ScanGrid.angle_deg` in the HDF5 grid metadata.
2. Estimate scan duration from distances between consecutive planned scan points,
   plus acquisition time and the existing fixed per-point overhead.
3. Apply correct one-sided ROI FFT normalization: do not double DC or the
   even-length Nyquist bin; double interior positive-frequency bins.

The time estimate excludes the initial move from the stage's unknown current
position to the first scan point because the estimate must work before the stage
is connected.

## Implementation

- Extend the existing grid metadata dictionary in `io/save.py`.
- Replace the X-step-only estimate in `gui/panel_scan.py` with the sum of
  Euclidean distances along `ScanGrid.points()`, using the backend's 1.0 µm/s
  default speed.
- Correct the FFT scaling in `ScanPanel._render_latest_roi_fft`.
- Add focused regression tests for each behavior before production changes.

## Verification

Run the focused scan/save tests, then Ruff on changed files. Real hardware is
not required because all three behaviors are deterministic.
