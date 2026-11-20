# Traffic Volume Forecasting: ARIMA vs XGBoost vs LSTM

A runnable, leakage-aware practical project using the **Metro Interstate Traffic Volume** dataset.

## Forecasting task

Forecast `traffic_volume` at horizons of **1, 6, and 24 hours** using:
- ARIMA (statsmodels)
- XGBoost (xgboost)
- LSTM (TensorFlow/Keras)

The project uses hourly observations, chronological train/validation/test partitions, rolling-origin validation, and direct multi-horizon models for XGBoost and LSTM. ARIMA is fitted separately for each horizon using a recursive forecast from the training history.

## Dataset

The script downloads the UCI Metro Interstate Traffic Volume dataset from:

https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip

Original dataset description: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

The CSV contains hourly traffic volume and exogenous fields including weather, holiday, and timestamp. This implementation deliberately uses **traffic history and calendar features only** for the primary model comparison, so all models can be compared under a common forecasting information set. Weather features are not used because future weather availability would need to be specified and handled consistently.

## Quick start

Python 3.10–3.11 is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python run_experiment.py
```

First run downloads the dataset, prepares hourly data, trains models, evaluates them, and writes outputs to `outputs/` and `reports/`.

For a faster smoke test:

```bash
python run_experiment.py --fast
```

`--fast` reduces the number of rolling origins, XGBoost trees, and LSTM epochs. It is intended to verify the pipeline, not for final reporting.

## Important methodological details

1. **Chronological split:** earliest 70% training, next 15% validation, final 15% test.
2. **No random shuffling:** all splits preserve time order.
3. **Leakage-safe features:** lag and rolling features are constructed from past observations only. Rolling means are shifted by one hour before calculation.
4. **Scaling:** LSTM scaler is fitted on the training segment only. XGBoost does not require scaling.
5. **Validation:** the LSTM uses the chronological validation segment for early stopping. ARIMA order is selected by AIC using training history only; XGBoost uses a fixed, documented configuration. The test segment is held back for final evaluation.
6. **Rolling-origin validation:** forecasts are evaluated at multiple origins within the test period. Each origin only uses observations available up to that time.
7. **Direct multi-horizon forecasting:** XGBoost and LSTM predict the next 24 values in one model call; metrics are reported at steps 1, 6, and 24. ARIMA produces a 24-step forecast and the same steps are extracted.
8. **Baseline:** seasonal-naive forecast uses the value from 24 hours earlier.
9. **Metrics:** MAE, RMSE, WAPE, and sMAPE. WAPE is used instead of ordinary MAPE because traffic volume can be zero.
10. The traffic dataset has a known interruption/gap in its timestamp coverage. The pipeline reindexes to a complete hourly grid and interpolates only short internal gaps (up to 3 hours); longer gaps are left missing and removed from supervised windows. Inspect `reports/data_quality.json`.

## Outputs

- `outputs/clean_hourly.csv` — cleaned hourly series
- `outputs/forecast_predictions.csv` — rolling-origin predictions
- `outputs/metrics_by_model_horizon.csv` — model comparison
- `outputs/metrics_summary.csv` — aggregate metrics
- `outputs/forecast_plot.png` — sample forecast plot
- `outputs/model_comparison.png` — metric comparison
- `outputs/traffic_series.png` — time-series overview
- `outputs/feature_importance.png` — XGBoost feature importance
- `reports/data_quality.json` — data-quality report
- `reports/experiment_report.md` — generated interpretation/report

## Reproducibility

The random seed is fixed at 42. Record your Python and package versions when submitting. Results can vary slightly by platform and TensorFlow version.

## Limitations

This is an academic forecasting benchmark, not a deployed traffic-management system. The models use past traffic and known calendar variables, not live incidents, road closures, or future weather. ARIMA is univariate; XGBoost and LSTM use lagged traffic plus deterministic calendar features. Compare results with these information-set differences clearly stated.
