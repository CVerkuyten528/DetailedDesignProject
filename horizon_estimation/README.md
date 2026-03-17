# Horizon Estimation

This directory contains the thermal horizon-estimation stack and the assets
required to generate its lookup data locally inside `DetailedDesignProject`.

## Contents

- `minimal_attitude_estimator.py`: runtime estimator
- `image_generation_V3.py`: synthetic thermal frame generator
- `data_generation.py`: batch frame generation entrypoint
- `conversion_table.py`: conversion-table builder
- `horizon_detection_V6.py`: feature extraction helpers
- `FOV_Files/`: calibration angle maps
- `generated/`: recommended output location for generated assets

## Required Python Packages

- `numpy`
- `scipy`
- `matplotlib`

## Interpreter Note

On this Windows machine, `python` resolves to the Microsoft Store alias and
`py` defaults to Python 3.11, which currently does not have the required
packages installed. The commands below are validated with:

```powershell
py -3.10
```

If you later create a project virtual environment with the required packages,
replace `py -3.10` with that interpreter path.

## Recommended Output Paths

- generated frames: `horizon_estimation/generated/frames/`
- roll bands: `horizon_estimation/generated/roll_bands.json`
- conversion table: `horizon_estimation/generated/conversion_table.npy`

The runtime estimator loads its table from
`horizon_estimation/generated/conversion_table.npy`.

## Generation Workflow

Run commands from the project root:

```powershell
cd C:\Users\rune\PycharmProjects\DetailedDesignProject
```

### 1. Generate local lookup data with roll-band discovery

```powershell
py -3.10 -m horizon_estimation.data_generation `
  --profile minimal-local `
  --auto-roll-bands `
  --pitch-start 0 `
  --pitch-end 359 `
  --pitch-step 1 `
  --roll-step 1 `
  --yaw 0 `
  --altitude-km 550 `
  --workers 1 `
  --clear-out-dir
```

This will:

- discover usable local roll bands
- save the bands to `horizon_estimation/generated/roll_bands.json`
- generate `p*_r*.npy` frames in `horizon_estimation/generated/frames/`

### 2. Build the conversion table

```powershell
py -3.10 -m horizon_estimation.conversion_table
```

This writes:

- `horizon_estimation/generated/conversion_table.npy`

The builder now reports batch progress, processing rate, and ETA while it runs.
If you want more frequent updates, lower the batch size:

```powershell
py -3.10 -m horizon_estimation.conversion_table --batch-size 256
```

By default, the conversion-table builder keeps only the canonical lower roll
band when the generated roll values split into two clearly separated bands.
This is the recommended mode for estimator use.

If you want to keep both valid roll bands instead, run:

```powershell
py -3.10 -m horizon_estimation.conversion_table --keep-all-roll-bands
```

### 3. Import the runtime estimator

```powershell
py -3.10 -c "from horizon_estimation.minimal_attitude_estimator import MinimalAttitudeEstimator; print('estimator ok')"
```

## Alternative Modes

### Full roll sweep

```powershell
py -3.10 -m horizon_estimation.data_generation `
  --profile full `
  --pitch-start 0 `
  --pitch-end 359 `
  --pitch-step 1 `
  --roll-start 0 `
  --roll-end 359 `
  --roll-step 1 `
  --yaw 0 `
  --altitude-km 550 `
  --workers 1 `
  --clear-out-dir
```

### Reuse an existing roll-band JSON

```powershell
py -3.10 -m horizon_estimation.data_generation `
  --use-bands-json horizon_estimation/generated/roll_bands.json `
  --pitch-start 0 `
  --pitch-end 359 `
  --pitch-step 1 `
  --roll-step 1 `
  --yaw 0 `
  --altitude-km 550 `
  --workers 1 `
  --clear-out-dir
```

### Discover bands only

```powershell
py -3.10 -m horizon_estimation.data_generation `
  --profile minimal-local `
  --auto-roll-bands `
  --discover-only `
  --yaw 0 `
  --altitude-km 550
```

## When To Regenerate

Regenerate frames and `conversion_table.npy` when any of these change:

- `FOV_Files/*`
- altitude assumptions
- yaw used during table generation
- threshold or feature-extraction logic
- sensor simulation assumptions

## Validation

Syntax check:

```powershell
py -3.10 -m py_compile `
  horizon_estimation\image_generation_V3.py `
  horizon_estimation\data_generation.py `
  horizon_estimation\horizon_detection_V6.py `
  horizon_estimation\conversion_table.py `
  horizon_estimation\minimal_attitude_estimator.py
```

Import smoke test after conversion-table generation:

```powershell
py -3.10 -c "from horizon_estimation.minimal_attitude_estimator import MinimalAttitudeEstimator; print('import ok')"
```
