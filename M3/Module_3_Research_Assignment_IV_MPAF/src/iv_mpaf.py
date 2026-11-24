from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def make_panel_features(panel: pd.DataFrame, lags, rolling_windows):
    """Create causal, per-sensor supervised features from a time x sensor panel."""
    records = []
    for sensor_idx, sensor in enumerate(panel.columns):
        s = panel[sensor]
        frame = pd.DataFrame(index=panel.index)
        frame["target"] = s
        frame["sensor_id"] = sensor_idx
        frame["hour"] = panel.index.hour
        frame["dayofweek"] = panel.index.dayofweek
        frame["minute"] = panel.index.minute
        frame["time_sin"] = np.sin(2 * np.pi * (panel.index.hour * 60 + panel.index.minute) / 1440)
        frame["time_cos"] = np.cos(2 * np.pi * (panel.index.hour * 60 + panel.index.minute) / 1440)
        frame["dow_sin"] = np.sin(2 * np.pi * panel.index.dayofweek / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * panel.index.dayofweek / 7)

        # Preserve missingness information before causal filling.
        frame["was_missing_current"] = s.isna().astype(int)
        filled = s.ffill()
        if filled.notna().any():
            filled = filled.fillna(filled.median())
        else:
            filled = filled.fillna(0.0)

        for lag in lags:
            frame[f"lag_{lag}"] = filled.shift(lag)
            frame[f"lag_missing_{lag}"] = s.shift(lag).isna().astype(int)
        for window in rolling_windows:
            frame[f"rolling_mean_{window}"] = filled.shift(1).rolling(window, min_periods=1).mean()
            frame[f"rolling_std_{window}"] = filled.shift(1).rolling(window, min_periods=2).std().fillna(0)
        # Sensor-specific historical level and variability, computed only from past values.
        frame["sensor_history_mean"] = filled.shift(1).expanding(min_periods=1).mean()
        frame["sensor_history_std"] = filled.shift(1).expanding(min_periods=2).std().fillna(0)
        frame["target_observed"] = s.notna().astype(int)
        frame["sensor_name"] = str(sensor)
        records.append(frame)

    return pd.concat(records).sort_index(kind="stable")


class IVMPAF:
    """
    Individual Variation and Missingness-aware Pattern-Adaptive Forecasting.

    A global XGBoost model learns shared patterns; sensor_id and sensor-history
    features represent entity-specific behaviour. Drift adaptation is implemented
    as periodic recent-window refitting when recent residual error exceeds a
    threshold relative to validation error.
    """
    def __init__(self, lags, rolling_windows, n_estimators=180,
                 recent_window=1500, random_state=42):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.n_estimators = n_estimators
        self.recent_window = recent_window
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
        self.validation_rmse = None

    def _new_model(self):
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            reg_lambda=2.0,
            random_state=self.random_state,
            n_jobs=4
        )

    def fit(self, features, train_end, validation_end):
        usable = features.dropna(subset=["target"])
        cols = [c for c in usable.columns
                if c not in ("target", "sensor_name")]
        self.feature_cols = cols
        train = usable[usable.index < train_end]
        valid = usable[(usable.index >= train_end) & (usable.index < validation_end)]
        self.model = self._new_model()
        self.model.fit(train[cols], train["target"])
        if len(valid):
            pred = self.model.predict(valid[cols])
            self.validation_rmse = float(np.sqrt(np.mean((valid["target"].values - pred) ** 2)))
        else:
            self.validation_rmse = None
        return self

    def predict_adaptive(self, features, train_end, validation_end, drift_threshold=1.35):
        usable = features.dropna(subset=["target"])
        test = usable[usable.index >= validation_end].copy()
        if test.empty:
            return pd.DataFrame(columns=["actual", "prediction", "sensor_name"])

        # Walk-forward prediction: predictions use features formed from past observations.
        # Refit only after the recent residual window crosses a validation-scaled threshold.
        preds = []
        history_features = usable[usable.index < validation_end]
        recent_errors = []
        base = self.validation_rmse if self.validation_rmse and self.validation_rmse > 0 else 1.0
        current_model = self.model
        seen = []
        for timestamp, row in test.iterrows():
            # Convert prediction features explicitly to numeric values.
            x = pd.DataFrame(
                [row[self.feature_cols].to_dict()],
                columns=self.feature_cols
            ).astype(float)

            prediction = float(current_model.predict(x)[0])
            actual = float(row["target"])
            preds.append({
                "timestamp": timestamp,
                "sensor_name": row["sensor_name"],
                "actual": actual,
                "prediction": prediction
            })
            recent_errors.append(abs(actual - prediction))
            seen.append(row)

            # Update after observing the current target; it only affects later predictions.
            if len(recent_errors) >= 30:
                recent_mae = float(np.mean(recent_errors[-30:]))
                if recent_mae > drift_threshold * base:
                    recent_frame = pd.DataFrame(seen[-self.recent_window:])
                    if len(recent_frame) >= 50:
                        current_model = self._new_model()
                        current_model.fit(recent_frame[self.feature_cols],
                                           recent_frame["target"])
                    recent_errors = []

        return pd.DataFrame(preds)
