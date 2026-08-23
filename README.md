# Eigenfrequency Transformation

This repository contains the analysis code developed for the automated
reconstruction of first-harmonic spectra from Schottky spectra.

## Contents

The repository contains:

- `Eigenfrequency_transformation_short.ipynb` – main analysis notebook
- the Python modules used by the analysis pipeline

The notebook calls the individual functions implemented in the Python
modules and provides the complete analysis workflow.

## Input data

The analysis expects an `.npz` spectrum containing the arrays

- `frequency`
- `amplitude`

The input file is loaded in the notebook using, for example:

```python
spectrum_file = "YOUR_SPECTRUM.npz"

data_spectrum = np.load(spectrum_file)

frequency = data_spectrum["frequency"]
amplitude = data_spectrum["amplitude"]
