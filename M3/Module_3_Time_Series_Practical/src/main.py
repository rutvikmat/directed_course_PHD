from pathlib import Path
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
ZIP_PATH = DATA_DIR / "metro_interstate_traffic_volume.zip"
CSV_PATH = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"

# Set to None to use all available observations.
MAX_ROWS = 18000
TEST_FRACTION = 0.20
LAG_COUNT = 24
SEQUENCE_LENGTH = 24
RANDOM_STATE = 42


def load_dataset():
    if not CSV_PATH.exists():
        print("Downloading UCI Metro Interstate Traffic Volume dataset...")
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV file found in downloaded dataset archive.")
            extracted = zf.extract(csv_names[0], DATA_DIR)
            Path(extracted).replace(CSV_PATH)

    df = pd.read_csv(CSV_PATH)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"])
    df = df.sort_values("date_time").drop_duplicates("date_time")
    df = df.set_index("date_time")

    # Reindex to hourly frequency. Missing target values are time-interpolated.
    hourly_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(hourly_index)
    df.index.name = "date_time"
    df["traffic_volume"] = df["traffic_volume"].interpolate(method="time").ffill().bfill()

    # Keep a manageable recent chronological segment for coursework execution.
    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.iloc[-MAX_ROWS:].copy()

    return df


def add_time_features(frame):
    # Use the target series and timestamp-derived predictors only.
    # This avoids passing raw text weather-description columns to XGBoost.
    out = pd.DataFrame(index=frame.index)
    out["traffic_volume"] = frame["traffic_volume"].astype(float)
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    for lag in range(1, LAG_COUNT + 1):
        out[f"lag_{lag}"] = out["traffic_volume"].shift(lag)
    out["rolling_mean_6"] = out["traffic_volume"].shift(1).rolling(6).mean()
    out["rolling_mean_24"] = out["traffic_volume"].shift(1).rolling(24).mean()
    return out


def metric_row(name, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    nonzero = np.abs(actual) > 1e-8
    mape = (np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
            if nonzero.any() else np.nan)
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": rmse,
        "MAPE_percent": mape,
        "R2": r2_score(actual, predicted)
    }


def fit_arima(train, test):
    # ARIMA(2,1,2) is used as a baseline specification for the practical.
    model = ARIMA(train, order=(2, 1, 2))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    return np.asarray(forecast)


def fit_xgboost(df, split_position):
    supervised = add_time_features(df).dropna()
    train = supervised.iloc[supervised.index < df.index[split_position]]
    test = supervised.iloc[supervised.index >= df.index[split_position]]
    feature_cols = [c for c in supervised.columns if c != "traffic_volume"]

    model = XGBRegressor(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=4
    )
    model.fit(train[feature_cols], train["traffic_volume"])
    pred = model.predict(test[feature_cols])
    return test.index, test["traffic_volume"].values, pred


def fit_lstm(train_values, test_values):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported. Install requirements.txt and retry."
        ) from exc

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).flatten()
    combined = np.concatenate([train_scaled, scaler.transform(test_values.reshape(-1, 1)).flatten()])

    X_train, y_train = [], []
    for i in range(SEQUENCE_LENGTH, len(train_scaled)):
        X_train.append(train_scaled[i-SEQUENCE_LENGTH:i])
        y_train.append(train_scaled[i])

    X_train = np.asarray(X_train).reshape(-1, SEQUENCE_LENGTH, 1)
    y_train = np.asarray(y_train)

    model = Sequential([
        LSTM(64, input_shape=(SEQUENCE_LENGTH, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(monitor="loss", patience=4, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[early_stop])

    # Recursive one-step forecasting: each prediction becomes the next input.
    history = list(train_scaled[-SEQUENCE_LENGTH:])
    predictions_scaled = []
    for _ in range(len(test_values)):
        x = np.asarray(history[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        predictions_scaled.append(p)
        history.append(p)

    predictions = scaler.inverse_transform(np.asarray(predictions_scaled).reshape(-1, 1)).flatten()
    return predictions


def main():
    print("Loading and preprocessing dataset...")
    df = load_dataset()
    y = df["traffic_volume"].astype(float)
    split = int(len(df) * (1 - TEST_FRACTION))
    train_y, test_y = y.iloc[:split], y.iloc[split:]
    print(f"Observations: {len(df)} | Train: {len(train_y)} | Test: {len(test_y)}")

    print("Training ARIMA...")
    arima_pred = fit_arima(train_y, test_y)

    print("Training XGBoost...")
    xgb_index, xgb_actual, xgb_pred = fit_xgboost(df, split)

    print("Training LSTM...")
    lstm_pred = fit_lstm(train_y.values, test_y.values)

    # Align all forecasts to the XGBoost test timestamps for a fair comparison.
    common_index = xgb_index
    common_actual = y.loc[common_index].values
    arima_aligned = pd.Series(arima_pred, index=test_y.index).loc[common_index].values
    lstm_aligned = pd.Series(lstm_pred, index=test_y.index).loc[common_index].values

    metrics = pd.DataFrame([
        metric_row("ARIMA", common_actual, arima_aligned),
        metric_row("XGBoost", common_actual, xgb_pred),
        metric_row("LSTM", common_actual, lstm_aligned)
    ]).sort_values("RMSE")
    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    predictions = pd.DataFrame({
        "actual": common_actual,
        "ARIMA": arima_aligned,
        "XGBoost": xgb_pred,
        "LSTM": lstm_aligned
    }, index=common_index)
    predictions.index.name = "date_time"
    predictions.to_csv(OUT_DIR / "forecast_predictions.csv")

    # Plot a readable subset of the test interval.
    plot_data = predictions.iloc[:min(500, len(predictions))]
    plt.figure(figsize=(14, 6))
    plt.plot(plot_data.index, plot_data["actual"], label="Actual", linewidth=2)
    for col in ["ARIMA", "XGBoost", "LSTM"]:
        plt.plot(plot_data.index, plot_data[col], label=col, alpha=0.85)
    plt.title("Actual vs Predicted Traffic Volume")
    plt.xlabel("Date and time")
    plt.ylabel("Traffic volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "forecast_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(metrics["Model"], metrics["RMSE"])
    plt.title("RMSE Comparison Across Forecasting Models")
    plt.ylabel("RMSE (lower is better)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_rmse_comparison.png", dpi=160)
    plt.close()

    best = metrics.iloc[0]
    summary = (
        "EXPERIMENT SUMMARY\n"
        "==================\n"
        f"Dataset: Metro Interstate Traffic Volume\n"
        f"Target: traffic_volume (hourly)\n"
        f"Total observations used: {len(df)}\n"
        f"Training observations: {len(train_y)}\n"
        f"Testing observations: {len(test_y)}\n\n"
        f"Best model by RMSE: {best['Model']}\n"
        f"Best RMSE: {best['RMSE']:.4f}\n\n"
        "Interpretation: Lower MAE and RMSE indicate smaller forecast errors. "
        "R-squared describes the proportion of variance explained, while MAPE "
        "expresses percentage error and should be interpreted cautiously when "
        "actual traffic volume is zero or close to zero.\n"
        "Use the saved metrics and plots to write the final observations based "
        "on the actual run; do not assume a model is best before examining results.\n"
    )
    (OUT_DIR / "experiment_summary.txt").write_text(summary, encoding="utf-8")

    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\nResults saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
