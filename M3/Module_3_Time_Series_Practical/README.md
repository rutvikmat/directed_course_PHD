# Module 3 Practical: Time-Series Forecasting

## Title
Comparative Time-Series Forecasting Using ARIMA, XGBoost and LSTM

## Aim
Forecast hourly traffic volume using one statistical model (ARIMA), one tree-based
machine-learning model (XGBoost), and one deep-learning model (LSTM), then compare
their performance on the same chronological test period.

## Dataset
Metro Interstate Traffic Volume (UCI Machine Learning Repository)

- Target: `traffic_volume`
- Frequency: hourly
- Predictors available: date/time, weather measurements, weather description and holidays.
- Dataset download is performed automatically by `src/main.py`.

Dataset source:
https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

## Requirements
Python 3.10 or 3.11 is recommended.

Install:
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Run:
```bash
python src/main.py
```

The script downloads the dataset if it is not already present, preprocesses it,
trains the three models, evaluates them on a chronological holdout set, and saves
metrics and forecast plots in `outputs/`.

## Experimental design
- The observations are sorted chronologically.
- The final 20% of observations form the test set.
- No random shuffling is used.
- ARIMA is fitted to the training target series.
- XGBoost uses lagged traffic values and calendar variables.
- LSTM uses fixed-length historical sequences.
- All models are evaluated on the same test timestamps where predictions are available.

## Evaluation metrics
- MAE: mean absolute error.
- RMSE: root mean squared error.
- MAPE: mean absolute percentage error; interpreted cautiously when actual values are zero.
- R²: coefficient of determination.

## Outputs
- `outputs/model_metrics.csv`
- `outputs/forecast_comparison.png`
- `outputs/actual_vs_predicted.png`
- `outputs/experiment_summary.txt`

## Notes
The default configuration uses a manageable subset of the hourly observations to
keep coursework execution practical. Increase `MAX_ROWS` in `src/main.py` for a
larger experiment. LSTM training may take longer depending on the computer.
