"""Leakage-aware electricity forecasting utilities."""
from pathlib import Path
import io
import zipfile
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
RESULTS = ROOT / 'results'
URL = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
FEATURE_LAGS = sorted(set(range(24)) | {24, 47, 48, 167})

def download_dataset():
    DATA.mkdir(parents=True, exist_ok=True)
    target = DATA / 'household_power_consumption.txt'
    if target.exists():
        return target
    print('Downloading dataset from UCI...')
    response = requests.get(URL, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        matches = [n for n in z.namelist() if n.endswith('household_power_consumption.txt')]
        if not matches:
            raise RuntimeError('Expected text file not found in UCI archive')
        with z.open(matches[0]) as source, target.open('wb') as dest:
            import shutil
            shutil.copyfileobj(source, dest)
    return target

def load_hourly(max_hours=None):
    path = download_dataset()
    df = pd.read_csv(path, sep=';', usecols=['Date', 'Time', 'Global_active_power'],
                     na_values=['?'], low_memory=False)
    idx = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    power = pd.to_numeric(df['Global_active_power'], errors='coerce')
    series = pd.Series(power.to_numpy(), index=idx, name='power_kw').sort_index()
    hourly = series.resample('h').mean()
    # Missing hour means are retained until causal imputation; no backward fill.
    hourly = hourly.ffill()
    hourly = hourly.dropna()
    if max_hours:
        hourly = hourly.iloc[-int(max_hours):]
    if len(hourly) < 1200:
        raise ValueError('At least 1200 hourly observations are needed')
    return hourly

def make_features(series):
    """Features at timestamp t forecast y(t+1), without using y(t+1)."""
    f = pd.DataFrame(index=series.index)
    for lag in FEATURE_LAGS:
        f[f'lag_{lag}'] = series.shift(lag)
    for window in [6, 24, 168]:
        f[f'rolling_mean_{window}'] = series.rolling(window, min_periods=window).mean()
        f[f'rolling_std_{window}'] = series.rolling(window, min_periods=window).std()
    next_time = f.index + pd.Timedelta(hours=1)
    f['hour_sin'] = np.sin(2*np.pi*next_time.hour/24)
    f['hour_cos'] = np.cos(2*np.pi*next_time.hour/24)
    f['weekday_sin'] = np.sin(2*np.pi*next_time.dayofweek/7)
    f['weekday_cos'] = np.cos(2*np.pi*next_time.dayofweek/7)
    f['weekend'] = (next_time.dayofweek >= 5).astype(int)
    f['seasonal_naive'] = series.shift(23)  # target at t+1, same hour previous day
    f['target'] = series.shift(-1)
    return f.dropna().copy()

def chronological_splits(frame):
    n = len(frame)
    a, b = int(n*.70), int(n*.85)
    return np.arange(a), np.arange(a,b), np.arange(b,n)

def fit_state_model(series, train_last_timestamp, maxiter=80):
    """Fit structural model only on the training history."""
    training = series.loc[:train_last_timestamp]
    model = sm.tsa.UnobservedComponents(training, level='local level', seasonal=24)
    fit = model.fit(disp=False, maxiter=maxiter)
    # Filter full observed history using fixed TRAINING parameters. Filtered state
    # at t depends only on observations <= t, never future observations.
    full_model = sm.tsa.UnobservedComponents(series, level='local level', seasonal=24)
    filtered = full_model.filter(fit.params)
    states = np.asarray(filtered.filtered_state).T
    state_names = [f'state_{i}' for i in range(states.shape[1])]
    state_frame = pd.DataFrame(states, index=series.index, columns=state_names)
    # One-step-ahead prediction at t+1 uses state updated through t.
    forecast = np.asarray(filtered.forecasts[0], dtype=float)
    # filtered.forecasts[t] predicts observation at t using data through t-1;
    # shift to align forecast for t+1 with feature time t.
    state_frame['ssm_forecast_next'] = pd.Series(forecast, index=series.index).shift(-1)
    return fit, state_frame

def create_models(seed=42, fast=False):
    return {
        'RandomForest': RandomForestRegressor(n_estimators=80 if fast else 200,
            max_depth=14, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=180 if fast else 450, max_depth=5,
            learning_rate=.05, subsample=.85, colsample_bytree=.9,
            objective='reg:squarederror', tree_method='hist', n_jobs=4, random_state=seed),
    }

def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'sMAPE': float(np.mean(200*np.abs(y-pred)/(np.abs(y)+np.abs(pred)+1e-8)))}

def save_csv(frame, name):
    RESULTS.mkdir(exist_ok=True)
    frame.to_csv(RESULTS / name, index=False)

def build_context(max_hours=None):
    series = load_hourly(max_hours)
    base = make_features(series)
    tr, va, te = chronological_splits(base)
    cutoff = base.index[tr[-1]]
    print(f'Fitting state-space model on observations through {cutoff}...')
    state_fit, state_features = fit_state_model(series, cutoff)
    state_features = state_features.reindex(base.index)
    base = base.join(state_features).dropna()
    # Splits are recalculated on aligned data, preserving chronology.
    tr, va, te = chronological_splits(base)
    base_cols = [c for c in base.columns if c not in ['target'] and not c.startswith('state_') and c != 'ssm_forecast_next']
    hybrid_cols = base_cols + [c for c in base.columns if c.startswith('state_')] + ['ssm_forecast_next']
    return base, tr, va, te, base_cols, hybrid_cols, state_fit
