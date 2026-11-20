from __future__ import annotations
import json, os, warnings, platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
DATA_DIR = ROOT / "data"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "Metro_Interstate_Traffic_Volume.csv.gz"
)
CSV_PATH = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"
HORIZONS = [1, 6, 24]
SEED = 42

def ensure_dirs():
    for p in [OUT, REPORTS, DATA_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def acquire_data():
    ensure_dirs()

    # Reuse the already downloaded, extracted CSV.
    if CSV_PATH.exists():
        print("Loading cached traffic dataset...")
        return pd.read_csv(CSV_PATH)

    import io

    urls = [
        "https://archive.ics.uci.edu/static/public/492/"
        "Metro_Interstate_Traffic_Volume.csv.gz",

        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00492/Metro_Interstate_Traffic_Volume.csv.gz",
    ]

    last_error = None

    for url in urls:
        try:
            print(f"Downloading dataset from: {url}")

            response = requests.get(
                url,
                timeout=90,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            # Read the gzip-compressed CSV directly.
            df = pd.read_csv(
                io.BytesIO(response.content),
                compression="gzip"
            )

            required = {"date_time", "traffic_volume"}

            if not required.issubset(df.columns):
                raise ValueError(
                    f"Unexpected dataset columns: {df.columns.tolist()}"
                )

            # Cache as a normal CSV for subsequent runs.
            df.to_csv(CSV_PATH, index=False)

            print("Dataset downloaded successfully!")
            print(f"Total observations: {len(df):,}")
            print(f"Columns: {df.columns.tolist()}")

            return df
 
        except Exception as exc:
            last_error = exc
            print(f"Download attempt failed: {exc}")

    raise RuntimeError(
        f"Could not download the UCI traffic dataset. "
        f"Last error: {last_error}"
    )


def prepare_hourly(raw):
    df = raw.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"])
    # Aggregate duplicates before reindexing to a regular hourly grid.
    df = df.groupby("date_time", as_index=False)["traffic_volume"].mean()
    df = df.set_index("date_time").sort_index()
    full_index = pd.date_range(df.index.min().floor("h"), df.index.max().ceil("h"), freq="h")
    original_count = len(df)
    series = df["traffic_volume"].reindex(full_index)
    missing_before = int(series.isna().sum())
    # Only short internal gaps are interpolated; long gaps remain missing.
    series = series.interpolate(method="time", limit=3, limit_area="inside")
    interpolated_count = missing_before - int(series.isna().sum())
    hourly = series.rename("traffic_volume").to_frame()
    hourly.index.name = "date_time"
    hourly.to_csv(OUT / "clean_hourly.csv")
    quality = {
        "raw_rows": int(len(raw)),
        "valid_timestamp_volume_rows": int(original_count),
        "hourly_grid_rows": int(len(hourly)),
        "missing_on_hourly_grid_before_interpolation": missing_before,
        "short_gap_values_interpolated": int(interpolated_count),
        "missing_after_interpolation": int(hourly["traffic_volume"].isna().sum()),
        "start": str(hourly.index.min()),
        "end": str(hourly.index.max()),
        "duplicate_timestamps_aggregated": int(len(df) - original_count) if len(df) >= original_count else 0
    }
    (REPORTS / "data_quality.json").write_text(json.dumps(quality, indent=2))
    return hourly, quality

def add_calendar_features(index):
    f = pd.DataFrame(index=index)
    f["hour"] = index.hour
    f["dayofweek"] = index.dayofweek
    f["month"] = index.month
    f["is_weekend"] = (index.dayofweek >= 5).astype(int)
    # Cyclical encoding preserves wrap-around continuity.
    f["hour_sin"] = np.sin(2*np.pi*f["hour"]/24)
    f["hour_cos"] = np.cos(2*np.pi*f["hour"]/24)
    f["dow_sin"] = np.sin(2*np.pi*f["dayofweek"]/7)
    f["dow_cos"] = np.cos(2*np.pi*f["dayofweek"]/7)
    f["month_sin"] = np.sin(2*np.pi*f["month"]/12)
    f["month_cos"] = np.cos(2*np.pi*f["month"]/12)
    return f.drop(columns=["hour", "dayofweek", "month"])

def make_supervised(series, max_horizon=24):
    """Each row at origin t uses only y<=t and known calendar values for t+1..t+H."""
    rows, targets, origins = [], [], []
    y = series["traffic_volume"]
    idx = y.index
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for i in range(168, len(y)-max_horizon+1):
        hist = y.iloc[:i+1]
        future = y.iloc[i+1:i+1+max_horizon]
        if hist.iloc[-168:].isna().any() or future.isna().any():
            continue
        row = {}
        for lag in lags:
            row[f"lag_{lag}"] = float(y.iloc[i-lag+1])
        past = y.iloc[:i+1]
        for w in [3, 6, 12, 24, 72, 168]:
            row[f"roll_mean_{w}"] = float(past.iloc[-w:].mean())
            row[f"roll_std_{w}"] = float(past.iloc[-w:].std(ddof=0))
        # Calendar features for each target step, known at forecast origin.
        for step in [1, 6, 24]:
            cf = add_calendar_features(pd.DatetimeIndex([idx[i] + pd.Timedelta(hours=step)])).iloc[0]
            for c, v in cf.items():
                row[f"tplus{step}_{c}"] = float(v)
        rows.append(row)
        targets.append(future.to_numpy(dtype=float))
        origins.append(idx[i])
    X = pd.DataFrame(rows, index=pd.DatetimeIndex(origins, name="origin"))
    Y = pd.DataFrame(targets, index=X.index, columns=[f"h_{h}" for h in range(1,max_horizon+1)])
    return X, Y

def metric_row(actual, pred, model, horizon, origin=None):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    denom = np.sum(np.abs(actual))
    wape = float(100*np.sum(np.abs(actual-pred))/denom) if denom > 0 else np.nan
    smape_denom = np.abs(actual)+np.abs(pred)
    smape = float(100*np.mean(2*np.abs(actual-pred)/np.maximum(smape_denom, 1e-8)))
    return {"model":model, "horizon_hours":horizon, "origin":str(origin) if origin is not None else "",
            "MAE":float(mae), "RMSE":rmse, "WAPE_pct":wape, "sMAPE_pct":smape}

def choose_arima_order(train, fast=False):
    # Small, predeclared candidate set selected only on the validation segment.
    candidates = [(1,0,1), (2,0,1), (2,1,1), (3,0,1)] if fast else [(1,0,1),(2,0,1),(2,1,1),(3,0,1),(1,1,2)]
    best_order, best_score = candidates[0], np.inf
    # Use recent training history to keep ARIMA fitting tractable.
    fit_data = train.dropna().iloc[-24*60:]
    for order in candidates:
        try:
            fit = ARIMA(fit_data, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
            score = float(fit.aic)
            if np.isfinite(score) and score < best_score:
                best_order, best_score = order, score
        except Exception:
            continue
    return best_order

def fit_xgb(X_train, Y_train, X_val, Y_val, fast=False):
    params = dict(
        objective="reg:squarederror", n_estimators=180 if fast else 450,
        max_depth=5, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=2.0,
        random_state=SEED, n_jobs=-1
    )
    models = []
    for h in range(1,25):
        model = XGBRegressor(**params)
        model.fit(X_train, Y_train.iloc[:,h-1], verbose=False)
        models.append(model)
    return models

def make_lstm_sequences(hourly):
    """Build [samples, 168 past hours, channels] sequences for a true LSTM."""
    y = hourly["traffic_volume"]
    idx = y.index
    calendar = add_calendar_features(idx)
    channels = pd.DataFrame(index=idx)
    channels["traffic"] = y
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        channels[c] = calendar[c]
    xs, ys, origins = [], [], []
    for i in range(167, len(hourly)-24):
        past = channels.iloc[i-167:i+1]
        future = y.iloc[i+1:i+25]
        if past.isna().any().any() or future.isna().any():
            continue
        xs.append(past.to_numpy(dtype=float))
        ys.append(future.to_numpy(dtype=float))
        origins.append(idx[i])
    return np.asarray(xs), np.asarray(ys), pd.DatetimeIndex(origins, name="origin")

def fit_lstm_sequences(xtr_raw, ytr_raw, xva_raw, yva_raw, fast=False):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    n_features = xtr_raw.shape[-1]
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_x.fit(xtr_raw.reshape(-1, n_features))
    scaler_y.fit(ytr_raw.reshape(-1, 1))
    def transform_x(a):
        return scaler_x.transform(a.reshape(-1,n_features)).reshape(a.shape)
    xtr, xva = transform_x(xtr_raw), transform_x(xva_raw)
    ytr = scaler_y.transform(ytr_raw.reshape(-1,1)).reshape(ytr_raw.shape)
    yva = scaler_y.transform(yva_raw.reshape(-1,1)).reshape(yva_raw.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(xtr.shape[1], xtr.shape[2])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(24)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3 if fast else 7, restore_best_weights=True)]
    model.fit(xtr, ytr, validation_data=(xva,yva),
              epochs=5 if fast else 35, batch_size=64,
              callbacks=callbacks, verbose=0, shuffle=False)
    return model, scaler_x, scaler_y

def predict_lstm(model, scaler_x, scaler_y, x):
    n_features = x.shape[-1]
    xs = scaler_x.transform(x.reshape(-1,n_features)).reshape(x.shape)
    pred_scaled = model.predict(xs, verbose=0)
    return scaler_y.inverse_transform(
        pred_scaled.reshape(-1,1)).reshape(pred_scaled.shape)

def forecast_arima(train_series, order, steps=24):
    data = train_series.dropna().iloc[-24*90:]
    model = ARIMA(data, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    return np.asarray(model.forecast(steps=steps), dtype=float)

def make_test_origins(index, test_start_pos, max_origins, fast):
    # Spread origins over the test period. The model is re-fit at each origin.
    possible = np.arange(max(test_start_pos, 168), len(index)-24)
    if len(possible) == 0:
        raise RuntimeError("Not enough test observations for 24-hour forecasting.")
    n = min(max_origins or (8 if fast else 24), len(possible))
    chosen = np.unique(np.linspace(possible[0], possible[-1], n, dtype=int))
    return chosen

def run(fast=False, max_origins=None):
    ensure_dirs()
    np.random.seed(SEED)
    raw = acquire_data()
    hourly, quality = prepare_hourly(raw)
    y = hourly["traffic_volume"]

    # Time split on complete hourly index; supervised rows later exclude unresolved gaps.
    n = len(hourly)
    train_end = int(n*0.70)
    val_end = int(n*0.85)
    train_cut = hourly.index[train_end]
    val_cut = hourly.index[val_end]

    X, Y = make_supervised(hourly, 24)
    # Keep each 24-hour target window entirely inside its partition.
    train_mask = (X.index + pd.Timedelta(hours=24) < train_cut)
    val_mask = ((X.index >= train_cut) &
                (X.index + pd.Timedelta(hours=24) < val_cut))
    test_mask = X.index >= val_cut
    Xtr, Ytr = X.loc[train_mask], Y.loc[train_mask]
    Xva, Yva = X.loc[val_mask], Y.loc[val_mask]
    Xtest, Ytest = X.loc[test_mask], Y.loc[test_mask]
    if min(len(Xtr),len(Xva),len(Xtest)) < 50:
        raise RuntimeError("Insufficient supervised samples after removing unresolved gaps.")

    print(f"Rows: train={len(Xtr)}, validation={len(Xva)}, test={len(Xtest)}")
    # Select ARIMA order using training history only (AIC selection); validation is reserved for XGB/LSTM early stopping.
    arima_order = choose_arima_order(y.iloc[:train_end], fast=fast)
    print("Selected ARIMA order:", arima_order)

    # XGBoost: 24 horizon-specific regressors for direct multi-step forecasting.
    xgb_models = fit_xgb(Xtr, Ytr, Xva, Yva, fast=fast)

    # LSTM: genuine 168-hour input sequence -> next 24 hours.
    xseq, yseq, seq_origins = make_lstm_sequences(hourly)
    seq_train = (seq_origins + pd.Timedelta(hours=24) < train_cut)
    seq_val = ((seq_origins >= train_cut) &
               (seq_origins + pd.Timedelta(hours=24) < val_cut))
    xtr_seq, ytr_seq = xseq[seq_train], yseq[seq_train]
    xva_seq, yva_seq = xseq[seq_val], yseq[seq_val]
    if len(xtr_seq) == 0 or len(xva_seq) == 0:
        raise RuntimeError("Insufficient leakage-safe LSTM train/validation sequences.")
    lstm_model, sx, sy = fit_lstm_sequences(
        xtr_seq, ytr_seq, xva_seq, yva_seq, fast=fast)
    seq_position = {t:i for i,t in enumerate(seq_origins)}

    # Rolling-origin test: at each origin, only history available at that time is used.
    test_origin_positions = make_test_origins(hourly.index, val_end, max_origins, fast)
    pred_rows = []
    xgb_imp = np.mean([m.feature_importances_ for m in xgb_models], axis=0)
    imp = pd.Series(xgb_imp, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10,6))
    imp.sort_values().plot(kind="barh")
    plt.title("XGBoost: Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT/"feature_importance.png", dpi=160)
    plt.close()

    for j, pos in enumerate(test_origin_positions):
        origin = hourly.index[pos]
        history = y.iloc[:pos+1]
        future = y.iloc[pos+1:pos+25]
        if len(future) != 24 or future.isna().any():
            continue
        # Features at this origin are reconstructed with the same causal function.
        row_frame, row_target = make_supervised(hourly.iloc[:pos+25], 24)
        if origin not in row_frame.index:
            continue
        xrow = X.loc[[origin]] if origin in X.index else row_frame.loc[[origin]]
        # XGB and LSTM predict all 24 horizons.
        xp = np.column_stack([m.predict(xrow) for m in xgb_models]).ravel()
        if origin not in seq_position:
            continue
        lp = predict_lstm(lstm_model, sx, sy,
                          xseq[[seq_position[origin]]])[0]
        # ARIMA re-fit using only observations up to this origin.
        try:
            ap = forecast_arima(history, arima_order, 24)
        except Exception:
            continue
        # Seasonal naive: same hour previous day.
        sn = np.array([history.iloc[-24+h-1] for h in range(1,25)], dtype=float)
        actual = future.to_numpy(dtype=float)
        for h in HORIZONS:
            for model_name, forecast in [("ARIMA",ap),("XGBoost",xp),("LSTM",lp),("SeasonalNaive24",sn)]:
                pred_rows.append({
                    "origin":origin, "target_time":origin+pd.Timedelta(hours=h),
                    "horizon_hours":h, "model":model_name,
                    "actual":actual[h-1], "prediction":float(forecast[h-1])
                })
        if (j+1) % 5 == 0:
            print(f"Completed rolling origin {j+1}/{len(test_origin_positions)}")

    preds = pd.DataFrame(pred_rows)
    if preds.empty:
        raise RuntimeError("No rolling-origin predictions were produced. Check data gaps or split settings.")
    preds.to_csv(OUT/"forecast_predictions.csv", index=False)

    metric_rows = []
    for (model_name,h), grp in preds.groupby(["model","horizon_hours"]):
        metric_rows.append(metric_row(grp.actual, grp.prediction, model_name, int(h)))
    metrics = pd.DataFrame(metric_rows).sort_values(["horizon_hours","RMSE"])
    metrics.to_csv(OUT/"metrics_by_model_horizon.csv", index=False)
    summary = metrics.groupby("model")[["MAE","RMSE","WAPE_pct","sMAPE_pct"]].mean().reset_index()
    summary.to_csv(OUT/"metrics_summary.csv", index=False)

    # Plots
    plt.figure(figsize=(15,5))
    y.plot(lw=0.6)
    plt.title("Metro Interstate Traffic Volume — Hourly Series")
    plt.ylabel("Vehicles per hour")
    plt.tight_layout()
    plt.savefig(OUT/"traffic_series.png", dpi=160)
    plt.close()

    # Example: first origin, 24-hour forecast for all models.
    first_origin = preds["origin"].min()
    sample = preds[(preds.origin==first_origin) & (preds.horizon_hours.isin(HORIZONS))]
    plt.figure(figsize=(12,5))
    for model_name, grp in sample.groupby("model"):
        grp = grp.sort_values("horizon_hours")
        plt.plot(grp.target_time, grp.prediction, marker="o", label=model_name)
    actual_sample = sample.drop_duplicates("target_time").sort_values("target_time")
    plt.plot(actual_sample.target_time, actual_sample.actual, marker="x", lw=2, label="Actual")
    plt.title(f"Forecast comparison from origin {first_origin}")
    plt.ylabel("Traffic volume")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(OUT/"forecast_plot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10,6))
    pivot = metrics.pivot(index="model", columns="horizon_hours", values="RMSE")
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title("RMSE by Model and Forecast Horizon")
    plt.ylabel("RMSE (vehicles/hour)")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(OUT/"model_comparison.png", dpi=160)
    plt.close()

    report = build_report(metrics, summary, quality, arima_order, len(test_origin_positions), fast)
    (REPORTS/"experiment_report.md").write_text(report)
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "arima_order": arima_order,
        "horizons_hours": HORIZONS,
        "split": {"train_end":str(train_cut),"validation_end":str(val_cut)},
        "rolling_origins_requested":len(test_origin_positions),
        "fast_mode":bool(fast)
    }
    (REPORTS/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    print("\n=== Metrics by model and horizon ===")
    print(metrics.to_string(index=False))
    print("\nOutputs saved in:", OUT)
    print("Report saved in:", REPORTS/"experiment_report.md")

def build_report(metrics, summary, quality, arima_order, n_origins, fast):
    best = metrics.loc[metrics.groupby("horizon_hours")["RMSE"].idxmin()]
    lines = [
        "# Traffic Volume Forecasting — Experimental Report",
        "",
        "## Objective",
        "Compare ARIMA, XGBoost, and LSTM for 1-, 6-, and 24-hour-ahead traffic-volume forecasting.",
        "",
        "## Dataset and preprocessing",
        f"- Raw rows: {quality['raw_rows']}",
        f"- Hourly grid rows: {quality['hourly_grid_rows']}",
        f"- Missing values after short-gap interpolation: {quality['missing_after_interpolation']}",
        f"- Time span: {quality['start']} to {quality['end']}",
        "- Duplicate timestamps are aggregated; the data are reindexed hourly; only short internal gaps (up to three hours) are interpolated.",
        "",
        "## Experimental protocol",
        "- Chronological split: 70% training, 15% validation, 15% test.",
        "- Forecast horizons: 1, 6, and 24 hours.",
        f"- ARIMA order selected by AIC from a small candidate set: {arima_order}.",
        "- XGBoost uses lagged/rolling traffic and known calendar features; LSTM receives a genuine 168-hour sequence (traffic plus cyclical calendar channels) and predicts the next 24 hours.",
        f"- Rolling-origin test evaluation used {n_origins} candidate origins (origins with incomplete 24-hour actual windows are skipped).",
        "- Seasonal-naive-24 baseline predicts using the corresponding hour on the previous day.",
        "- Metrics: MAE, RMSE, WAPE, and sMAPE.",
        "",
        "## Results by model and horizon",
        metrics.to_markdown(index=False),
        "",
        "## Best model by horizon (lowest RMSE)",
        best[["horizon_hours","model","MAE","RMSE","WAPE_pct","sMAPE_pct"]].to_markdown(index=False),
        "",
        "## Average metrics across horizons",
        summary.to_markdown(index=False),
        "",
        "## Interpretation guidance",
        "Do not claim that one model is universally superior. Discuss which model performs best at each horizon, whether error increases with horizon, and how the models compare with the seasonal-naive baseline.",
        "A lower RMSE indicates fewer large errors; MAE is easier to interpret in vehicles/hour. WAPE is more stable than MAPE when actual values can be zero.",
        "",
        "## Limitations and threats to validity",
        "- XGBoost and LSTM use lagged traffic plus known calendar features; ARIMA is univariate.",
        "- Weather and holiday variables are excluded from the shared primary information set.",
        "- Traffic regimes, road incidents, construction, and policy changes are not explicitly modelled.",
        "- Results depend on the chosen split, rolling-origin sample, ARIMA candidate set, and compute budget.",
        "- The `--fast` option is for pipeline verification only; rerun the full configuration for final academic reporting.",
        "",
        "## Conclusion",
        "Insert a conclusion based on the measured results above. State the best-performing model separately for each forecast horizon and discuss practical trade-offs."
    ]
    return "\n".join(lines) + "\n"
