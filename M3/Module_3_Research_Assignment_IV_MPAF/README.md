# Module 3 Research Assignment
## IV-MPAF: Individual Variation and Missingness-aware Pattern-Adaptive Forecasting

This package contains a reproducible coursework research prototype for developing
and evaluating a time-series forecasting framework that considers:

1. Individual/entity variation (for example, differences between traffic sensors).
2. Missing observations.
3. Changing temporal patterns (concept drift / change in behaviour).

**Dataset:** PEMS-BAY traffic speed data (multiple sensors, 5-minute sampling).

**Important:** This is an implementation and report package, not a claim that the
proposed method has already outperformed the baselines. Run the experiments and
use the generated results in the final report.

## Package contents

```text
Module_3_Research_Assignment_IV_MPAF/
├── README.md
├── requirements.txt
├── config.json
├── data/
│   └── README_DATA.md
├── docs/
│   ├── RESEARCH_REPORT.md
│   ├── EXPERIMENT_PROTOCOL.md
│   └── RESULTS_TEMPLATE.md
├── outputs/
│   └── .gitkeep
└── src/
    ├── run_experiments.py
    └── iv_mpaf.py
```

## 1. Dataset preparation

Place the PEMS-BAY data in `data/` as one of the following:

- `PEMS-BAY.csv`
- `pems_bay.csv`
- `PEMS-BAY.npz` (supported if it contains an array named `data` or a single array)

For CSV, accepted layouts:

**Wide format:** one timestamp column and one column per sensor.

```csv
timestamp, sensor_001, sensor_002, sensor_003
2017-01-01 00:00:00, 65.2, 61.8, 70.1
...
```

**Long format:** timestamp, sensor_id, speed.

```csv
timestamp,sensor_id,speed
2017-01-01 00:00:00,001,65.2
...
```

If you have the original PEMS-BAY HDF5 file, convert it to a wide CSV first.
Example conversion:

```python
import pandas as pd
df = pd.read_hdf("pems-bay.h5")
df.to_csv("data/PEMS-BAY.csv", index_label="timestamp")
```

Check the original file's structure before using this example; adapt the conversion
if its time index or sensor dimensions differ.

Dataset citation:
Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion Convolutional Recurrent
Neural Network: Data-Driven Traffic Forecasting. ICLR.
PEMS-BAY data is commonly distributed with the DCRNN benchmark materials.

## 2. Environment setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run

From the project root:

```bash
python src/run_experiments.py
```

Optional configuration can be changed in `config.json`. The default run uses a
chronological split and controlled missingness/drift experiments. To keep runtime
reasonable, the default configuration uses a subset of sensors and observations.

## 4. Framework design

The prototype implements an entity-aware global gradient-boosting forecaster.
It combines:
- Shared global model learning across sensors.
- Sensor identity as an entity feature.
- Sensor-specific recent level and variability statistics.
- Lagged observations and calendar/time features.
- Missingness indicators and causal forward filling.
- A drift-aware update policy using recent prediction residuals.
- A comparison against non-adaptive global XGBoost and simple seasonal-naive baseline.

The proposed approach is a practical operationalisation of IV-MPAF. It is not
claimed to be the only possible implementation of the conceptual framework.

## 5. Experiments

The script runs:
- Baseline experiment on the original observed series.
- Missing-data robustness experiment with 5%, 10%, 20% and 30% artificial missingness.
- Drift robustness experiment with a controlled level shift in a selected test interval.
- Combined missingness and drift experiment.

All artificial perturbations are applied to copies of the data for evaluation. The
original values are retained as ground truth for scoring where they are available.

## 6. Outputs

Generated under `outputs/`:
- `metrics.csv` — model and scenario metrics.
- `predictions_sample.csv` — a compact sample of forecast rows.
- `missingness_robustness.png`
- `drift_robustness.png`
- `scenario_model_comparison.png`
- `run_summary.txt`

## 7. Interpretation

Lower MAE and RMSE indicate lower point-forecast errors. Compare the proposed
method with baselines under each scenario. Do not assume the proposed method wins.
Discuss trade-offs, computational cost, and limitations.

## 8. Reproducibility

The random seed is fixed in the configuration. The script records the number of
observations, sensors, split sizes and scenario settings. Use the same test window
for all models within a scenario.

## Academic integrity note

The package provides code, methodology and a report template. Numerical results
must come from executing the experiment on the dataset. Do not submit invented
metrics or describe expected improvements as measured findings.
