# Andor iDus 420 Demodulation GUI

Desktop GUI for the **Andor iDus 420** spectrometer camera. It supports live spectra, ROI tracking, real-time demodulation, and fixed-length acquisitions.

## Main Features

- **Camera Settings**: connect to the mock or Andor backend, control cooling, and apply camera settings.
- **Spectrometer**: connect to the Andor Shamrock/Kymera spectrograph, choose the grating, and set the central wavelength. The resulting calibration relabels the spectrum plots in nanometers.
- **Live Spectrum**: view the incoming spectrum, select an ROI, and plot the rolling summed counts from that ROI over time.
- **Demodulation Alignment**: stream frames into a rolling ROI window that fills then slides, and monitor the spectrum, rolling time series, FFT magnitude, and running peak amplitude continuously in real time.
- **Acquisition**: capture a fixed duration or frame count and save data as `.npz`, `.h5`, `.txt`, or `.sif` when supported.
- **Mock backend**: run and test the app without hardware.

## Requirements

- Python 3.10+
- PyQt6, NumPy, SciPy, pyqtgraph, h5py
- For the real camera: Andor SDK v2 with `pyAndorSDK2` available in the Python environment
- For the real spectrograph: the Andor Spectrograph SDK with `pyAndorSpectrograph` available in the Python environment

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
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

### Spectrometer

- Connect to the **Mock** or **Andor Shamrock** spectrograph
- Select the installed grating and set the central wavelength
- Set the detector pixel width (26 µm on the iDus 420) used for the wavelength calibration
- Applying the settings broadcasts the per-pixel wavelength calibration so the Live and Demodulation spectra plot in nanometers; without a connected spectrograph the plots fall back to pixel index

### Live Spectrum

- Stream the current spectrum continuously
- Drag or type an ROI (ROI bounds stay in pixels even when the axis shows nm)
- Plot the ROI summed counts versus time

### Demodulation Alignment

- Set trigger frequency and ROI bounds
- Run continuous ROI demodulation over a rolling window of the last `Frames / FFT block` samples
- Tune `Refresh every (frames)` to control how often the rolling window re-demodulates and the plots refresh
- Inspect FFT peak amplitude and frequency in real time

### Acquisition

- Acquire by duration or by frame count
- Save spectra, ROI time series, and demodulation results

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
