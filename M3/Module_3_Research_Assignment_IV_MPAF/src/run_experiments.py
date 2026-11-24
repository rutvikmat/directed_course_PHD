from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from iv_mpaf import make_panel_features, IVMPAF

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_panel(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nSee data/README_DATA.md for required format."
        )
    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        arr = archive["data"] if "data" in archive else archive[archive.files[0]]
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        idx = pd.date_range("2017-01-01", periods=arr.shape[0], freq="5min")
        return pd.DataFrame(arr, index=idx,
                            columns=[f"sensor_{i}" for i in range(arr.shape[1])])

    df = pd.read_csv(path)
    if CFG["timestamp_column"] not in df.columns:
        # If timestamp is saved as the first unnamed/index column, use it.
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().mean() > 0.8:
            df = df.rename(columns={first: CFG["timestamp_column"]})
        else:
            raise ValueError("Could not identify timestamp column; see data instructions.")

    df[CFG["timestamp_column"]] = pd.to_datetime(df[CFG["timestamp_column"]], errors="coerce")
    df = df.dropna(subset=[CFG["timestamp_column"]]).set_index(CFG["timestamp_column"]).sort_index()
    if {"sensor_id", CFG["target_column"]}.issubset(df.columns):
        panel = df.pivot_table(index=df.index, columns="sensor_id",
                               values=CFG["target_column"], aggfunc="mean")
    else:
        panel = df.select_dtypes(include=[np.number])
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    panel = panel.iloc[:, :CFG["max_sensors"]]
    if CFG["max_observations"]:
        panel = panel.iloc[:CFG["max_observations"]]
    return panel.astype(float)


def inject_missing(panel, rate, seed):
    rng = np.random.default_rng(seed)
    corrupted = panel.copy()
    mask = rng.random(corrupted.shape) < rate
    corrupted = corrupted.mask(mask)
    return corrupted, mask


def inject_drift(panel, start_fraction, magnitude):
    changed = panel.copy()
    start = int(len(changed) * start_fraction)
    # Apply a sensor-specific level shift to a subset of sensors for a controlled test.
    n = max(1, changed.shape[1] // 3)
    cols = list(changed.columns[:n])
    for col in cols:
        scale = max(float(changed[col].median()), 1.0)
        changed.iloc[start:, changed.columns.get_loc(col)] += magnitude * scale
    return changed, start


def metric(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p)))
    }


def run_one(panel, scenario, missing_rate=0.0, drift=False):
    # Preserve ground truth for evaluation, while model receives corrupted series.
    truth = panel.copy()
    observed = panel.copy()
    if missing_rate:
        observed, _ = inject_missing(observed, missing_rate, CFG["random_seed"])
    drift_start = None
    if drift:
        observed, drift_start = inject_drift(
            observed, CFG["drift_start_fraction"], CFG["drift_magnitude"]
        )
        truth, _ = inject_drift(truth, CFG["drift_start_fraction"], CFG["drift_magnitude"])

    n = len(observed)
    train_end_i = int(n * CFG["train_fraction"])
    valid_end_i = int(n * (CFG["train_fraction"] + CFG["validation_fraction"]))
    train_end = observed.index[train_end_i]
    valid_end = observed.index[valid_end_i]

    features = make_panel_features(observed, CFG["lags"], CFG["rolling_windows"])
    model = IVMPAF(
        CFG["lags"], CFG["rolling_windows"],
        CFG["xgboost_estimators"], CFG["adaptive_recent_window"],
        CFG["random_seed"]
    ).fit(features, train_end, valid_end)
    pred_df = model.predict_adaptive(features, train_end, valid_end)

    # Evaluate against ground truth at forecast timestamps, not corrupted inputs.
    if pred_df.empty:
        raise RuntimeError("No test predictions were produced.")
    pred_df["truth"] = [
        truth.loc[t, s] for t, s in zip(pred_df["timestamp"], pred_df["sensor_name"])
    ]
    pred_df = pred_df.dropna(subset=["truth"])

    # Non-adaptive global XGBoost baseline, trained only on pre-test observations.
    usable = features.dropna(subset=["target"])
    cols = [c for c in usable.columns if c not in ("target", "sensor_name")]
    tr = usable[usable.index < valid_end]
    te = usable[usable.index >= valid_end].copy()
    baseline = RandomForestRegressor(
        n_estimators=120, max_depth=12, min_samples_leaf=3,
        random_state=CFG["random_seed"], n_jobs=-1
    )
    baseline.fit(tr[cols], tr["target"])
    te_pred = baseline.predict(te[cols])
    te["rf_prediction"] = te_pred
    te["truth"] = [
        truth.loc[t, s] for t, s in zip(te.index, te["sensor_name"])
    ]
    te = te.dropna(subset=["truth"])

    results = []
    m = metric(pred_df["truth"], pred_df["prediction"])
    results.append({"scenario": scenario, "model": "IV-MPAF", **m})
    m = metric(te["truth"], te["rf_prediction"])
    results.append({"scenario": scenario, "model": "RandomForest", **m})

    # Seasonal-naive benchmark: last observed value from one day earlier (288 steps).
    naive_rows = []
    for sensor in observed.columns:
        series = observed[sensor].ffill().bfill()
        test_idx = series.index[valid_end_i:]
        actual = truth.loc[test_idx, sensor]
        naive = series.shift(288).loc[test_idx]
        valid = actual.notna() & naive.notna()
        if valid.any():
            naive_rows.extend(zip(actual[valid].values, naive[valid].values))
    if naive_rows:
        a, p = zip(*naive_rows)
        results.append({"scenario": scenario, "model": "SeasonalNaive", **metric(a, p)})

    pred_df["scenario"] = scenario
    return results, pred_df, drift_start


def main():
    path = ROOT / CFG["data_file"]
    panel = load_panel(path)
    panel = panel.dropna(axis=1, how="all")
    print(f"Loaded panel: {panel.shape[0]} timestamps x {panel.shape[1]} sensors")
    print(f"Time range: {panel.index.min()} to {panel.index.max()}")

    scenarios = [("Original", 0.0, False)]
    scenarios += [(f"Missing_{int(r*100)}pct", r, False) for r in CFG["missing_rates"]]
    scenarios += [("Drift", 0.0, True), ("Combined_20pct_Drift", 0.20, True)]

    all_metrics, all_predictions = [], []
    for name, missing, drift in scenarios:
        print(f"Running scenario: {name}")
        rows, preds, _ = run_one(panel, name, missing, drift)
        all_metrics.extend(rows)
        all_predictions.append(preds)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.head(5000).to_csv(OUT / "predictions_sample.csv", index=False)

    # Plot missingness robustness.
    miss = metrics[metrics["scenario"].str.startswith("Missing_")]
    if not miss.empty:
        plt.figure(figsize=(8, 5))
        for model_name in miss["model"].unique():
            sub = miss[miss["model"] == model_name]
            plt.plot(sub["scenario"], sub["RMSE"], marker="o", label=model_name)
        plt.title("Forecasting Error Under Different Missingness Rates")
        plt.ylabel("RMSE")
        plt.xlabel("Scenario")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "missingness_robustness.png", dpi=160)
        plt.close()

    drift = metrics[metrics["scenario"].isin(["Drift", "Combined_20pct_Drift"])]
    if not drift.empty:
        plt.figure(figsize=(8, 5))
        for model_name in drift["model"].unique():
            sub = drift[drift["model"] == model_name]
            plt.plot(sub["scenario"], sub["RMSE"], marker="o", label=model_name)
        plt.title("Forecasting Error Under Temporal Drift")
        plt.ylabel("RMSE")
        plt.xlabel("Scenario")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "drift_robustness.png", dpi=160)
        plt.close()

    original = metrics[metrics["scenario"] == "Original"]
    if not original.empty:
        plt.figure(figsize=(7, 5))
        plt.bar(original["model"], original["RMSE"])
        plt.title("Baseline Model Comparison")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.savefig(OUT / "scenario_model_comparison.png", dpi=160)
        plt.close()

    summary = [
        "IV-MPAF RESEARCH EXPERIMENT SUMMARY",
        "===================================",
        f"Rows: {len(panel)}",
        f"Sensors: {panel.shape[1]}",
        f"Training fraction: {CFG['train_fraction']}",
        f"Validation fraction: {CFG['validation_fraction']}",
        "",
        "Metrics are in metrics.csv. Interpret actual results; no winning model is assumed.",
        "Artificial missingness and level shifts are controlled stress tests.",
        "The drift experiment is synthetic and should be reported as such."
    ]
    (OUT / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Outputs saved to {OUT}")


if __name__ == "__main__":
    main()
