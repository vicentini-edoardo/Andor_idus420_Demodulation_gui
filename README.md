# Andor iDus 420 Demodulation GUI

Desktop GUI for the **Andor iDus 420** spectrometer camera. Supports live spectra, ROI tracking, real-time demodulation, fixed-length acquisitions, and 2-D SNOM raster scans.

## Main Features

- **Camera Settings**: connect to the mock or Andor backend, control cooling, and apply camera settings.
- **Live Spectrum**: view the incoming spectrum, select an ROI, and plot the rolling summed counts from that ROI over time.
- **Demodulation Alignment**: monitor spectrum, ROI time series, FFT magnitude, and running peak amplitude in real time, using a rolling window of the most recent samples that re-demodulates continuously as frames arrive.
- **Acquisition**: capture a fixed duration or frame count and save data as `.npz`, `.h5`, `.txt`, or `.sif` when supported.
- **Scan**: 2-D XY raster scan using the NEA SNOM sample stage — configurable grid, snake or raster order, per-point demodulation, live scan map, and automatic HDF5 export.
- **Mock backend**: run and test the app without hardware.

## Requirements

- Python 3.9+
- PyQt6, NumPy, SciPy, pyqtgraph, h5py
- For real camera hardware: Andor SDK v2 with `pyAndorSDK2` available in the Python environment
- For SNOM scanning: `nea_tools` and `nest_asyncio` (see Install below)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To enable SNOM scanning:

```bash
python -m pip install -e ".[snom]"
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Run

```bash
python -m idus420_gui
```

If you have not installed the package and just want to run from the repo:

```bash
PYTHONPATH=src python -m idus420_gui
```

Use the **Mock** backend unless the camera and Andor SDK are available on the machine.

## Tabs

### Camera Settings

- Connect or disconnect the backend
- Control cooling and temperature target
- Set exposure, shutter, gain, and readout parameters

### Live Spectrum

- Stream the current spectrum continuously
- Drag or type an ROI
- Plot the ROI summed counts versus time

### Demodulation Alignment

- Set trigger frequency and ROI bounds
- Run continuous ROI demodulation over a rolling (sliding) window of the last `Frames / FFT block` samples
- Inspect FFT peak amplitude and frequency in real time, updated as each new frame slides into the window

### Acquisition

- Acquire by duration or by frame count
- Save spectra, ROI time series, and demodulation results

### Scan

- Define a 2-D XY grid (start, step, number of points) in nanometres
- Choose snake (boustrophedon) or left-to-right raster order
- Set per-point frame count for demodulation
- Watch the scan map update live as each point completes
- Results saved automatically as an HDF5 file
- The tab is always accessible; if `nea_tools` is not installed a banner explains what to install

## Tests

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src PYTEST_QT_API=pyqt6 python -m pytest -q
python -m ruff check src tests
```

## Notes

- The app supports **FVB** read mode only.
- `.sif` saving depends on backend support from the Andor SDK.

## License

MIT. See [LICENSE](LICENSE).
