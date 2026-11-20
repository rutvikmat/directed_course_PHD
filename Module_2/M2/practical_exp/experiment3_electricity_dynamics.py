
"""
EXPERIMENT 3
REAL-WORLD DYNAMICAL SYSTEM IDENTIFICATION
AND ELECTRICITY DEMAND FORECASTING

Dataset:
UCI Electricity Load Diagrams 2011-2014

Models:
1. ARX Dynamical System
2. Random Forest Regressor

Analysis:
- Dataset preprocessing
- System identification
- Parameter estimation
- Stability analysis
- Forecast comparison
- Residual analysis
- Visualization

Python: 3.13 compatible
"""

import shutil
import zipfile
import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.80

CONSUMER_ID = "MT_001"

RESAMPLING_INTERVAL = "1h"

RANDOM_FOREST_TREES = 250

RANDOM_FOREST_DEPTH = 14

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "electricity_data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ZIP_PATH = DATA_DIR / "electricity.zip"

TEMP_ZIP = DATA_DIR / "electricity_download.tmp"

DATA_FILE = DATA_DIR / "LD2011_2014.txt"


# Official UCI dataset URLs
DATA_URLS = [
    (
        "https://archive.ics.uci.edu/static/public/321/"
        "electricityloaddiagrams20112014.zip"
    ),
    (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00321/"
        "LD2011_2014.txt.zip"
    )
]


# ==========================================================
# 2. DATASET DOWNLOAD AND VALIDATION
# ==========================================================

def download_dataset():

    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:

        print("\nDataset already exists.")
        print("Using:", DATA_FILE)

        return

    print("\nDATASET DOWNLOAD")
    print("----------------")

    session = requests.Session()

    session.headers.update({
        "User-Agent": "Mozilla/5.0 Python Research Project"
    })

    downloaded = False

    for url in DATA_URLS:

        try:

            print("\nTrying URL:")
            print(url)

            with session.get(
                url,
                stream=True,
                timeout=(30, 180)
            ) as response:

                response.raise_for_status()

                with open(TEMP_ZIP, "wb") as file:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if chunk:
                            file.write(chunk)

            # Validate downloaded archive
            if not zipfile.is_zipfile(TEMP_ZIP):

                print("Invalid ZIP response.")

                with open(TEMP_ZIP, "rb") as file:
                    preview = file.read(200)

                print("Response preview:", preview[:100])

                TEMP_ZIP.unlink(missing_ok=True)

                continue

            # Accept validated archive
            shutil.move(
                str(TEMP_ZIP),
                str(ZIP_PATH)
            )

            downloaded = True

            print("\nDataset downloaded successfully.")

            break

        except Exception as error:

            print("\nDownload error:", error)

            TEMP_ZIP.unlink(missing_ok=True)

    session.close()

    if not downloaded:

        raise RuntimeError(
            "\nUnable to download a valid dataset.\n"
            "Download manually from:\n"
            "https://archive.ics.uci.edu/dataset/321/"
            "electricityloaddiagrams20112014"
        )

    # ======================================================
    # EXTRACT DATASET
    # ======================================================

    print("\nExtracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        members = [
            name
            for name in archive.namelist()
            if Path(name).name == "LD2011_2014.txt"
        ]

        if not members:

            raise FileNotFoundError(
                "LD2011_2014.txt not found in ZIP archive."
            )

        with archive.open(members[0]) as source:

            with open(DATA_FILE, "wb") as destination:

                shutil.copyfileobj(
                    source,
                    destination
                )

    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:

        raise RuntimeError("Dataset extraction failed.")

    print("Dataset extracted successfully.")

    print(
        "Extracted size:",
        round(DATA_FILE.stat().st_size / (1024 ** 2), 2),
        "MB"
    )


# ==========================================================
# 3. LOAD ELECTRICITY DATASET
# ==========================================================

def load_electricity_data():

    download_dataset()

    print("\nLOADING DATASET")
    print("---------------")

    # Read column names only
    header = pd.read_csv(
        DATA_FILE,
        sep=";",
        nrows=0
    )

    columns = list(header.columns)

    if len(columns) < 2:

        raise ValueError(
            "Invalid electricity dataset structure."
        )

    timestamp_column = columns[0]

    # Select requested consumer or first available consumer
    if CONSUMER_ID in columns:

        consumer = CONSUMER_ID

    else:

        consumer = columns[1]

        print(
            f"Requested consumer unavailable. "
            f"Using {consumer} instead."
        )

    print("Selected consumer:", consumer)

    # Read only required columns
    df = pd.read_csv(
        DATA_FILE,
        sep=";",
        decimal=",",
        usecols=[
            timestamp_column,
            consumer
        ]
    )

    # Parse timestamps
    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    df[consumer] = pd.to_numeric(
        df[consumer],
        errors="coerce"
    )

    df = df.dropna(
        subset=[timestamp_column]
    )

    df = df.sort_values(
        timestamp_column
    )

    # Handle duplicate timestamps if present
    series = df.groupby(
        timestamp_column
    )[consumer].mean()

    series.index.name = "timestamp"

    # Convert 15-minute data into hourly averages
    hourly = series.resample(
        RESAMPLING_INTERVAL
    ).mean()

    print("\nOriginal observations:", len(series))
    print("Hourly observations:", len(hourly))

    # Interpolate short missing gaps
    hourly = hourly.interpolate(
        method="time",
        limit=2
    )

    remaining_missing = int(hourly.isna().sum())

    print(
        "Missing hourly observations:",
        remaining_missing
    )

    if remaining_missing > 0:

        raise ValueError(
            "Long missing-data gaps remain after "
            "interpolation. Inspect the dataset before "
            "performing equally spaced time-series analysis."
        )

    if not np.isfinite(hourly.values).all():

        raise ValueError(
            "Non-finite values detected in electricity data."
        )

    if len(hourly) < 1000:

        raise ValueError(
            "Insufficient observations for this experiment."
        )

    return hourly


# ==========================================================
# 4. DATASET INFORMATION AND EXPORT
# ==========================================================

def inspect_dataset(series):

    print("\nDATASET INFORMATION")
    print("===================")

    print("Total observations:", len(series))

    print("Start:", series.index.min())

    print("End:", series.index.max())

    print("Mean demand:", series.mean())

    print("Standard deviation:", series.std())

    print("Minimum demand:", series.min())

    print("Maximum demand:", series.max())

    print("Missing values:", series.isna().sum())

    print("Sampling frequency: Hourly")


# ==========================================================
# 5. CREATE DYNAMICAL SYSTEM FEATURES
# ==========================================================

def create_features(series):

    values = series.to_numpy(dtype=float)

    timestamps = series.index

    # Predict demand at time t using demand at t-1
    target = values[1:]

    target_times = timestamps[1:]

    previous_demand = values[:-1]

    # Daily periodicity
    hour = (
        target_times.hour
        + target_times.minute / 60
    )

    daily_sin = np.sin(
        2 * np.pi * hour / 24
    )

    daily_cos = np.cos(
        2 * np.pi * hour / 24
    )

    # Weekly periodicity
    weekly_hour = (
        target_times.dayofweek * 24
        + hour
    )

    weekly_sin = np.sin(
        2 * np.pi * weekly_hour / 168
    )

    weekly_cos = np.cos(
        2 * np.pi * weekly_hour / 168
    )

    # ARX feature matrix
    X = np.column_stack([
        previous_demand,
        np.ones(len(target)),
        daily_sin,
        daily_cos,
        weekly_sin,
        weekly_cos
    ])

    return X, target, target_times


# ==========================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================================

def chronological_split(series, X, target):

    split = int(
        TRAIN_RATIO * len(series)
    )

    # Target observation positions begin at index 1
    target_positions = np.arange(
        1,
        len(series)
    )

    train_mask = target_positions < split

    test_mask = target_positions >= split

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = target[train_mask]
    y_test = target[test_mask]

    print("\nTRAIN-TEST SPLIT")
    print("================")

    print("Training observations:", len(y_train))

    print("Testing observations:", len(y_test))

    print("Training percentage:", TRAIN_RATIO * 100)

    print("Testing percentage:", (1 - TRAIN_RATIO) * 100)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_mask
    )


# ==========================================================
# 7. DYNAMICAL SYSTEM IDENTIFICATION
# ==========================================================

def estimate_dynamical_system(X_train, y_train):

    print("\nDYNAMICAL SYSTEM IDENTIFICATION")
    print("===============================")

    # Least-squares parameter estimation
    parameters, residuals, rank, singular_values = (
        np.linalg.lstsq(
            X_train,
            y_train,
            rcond=None
        )
    )

    a, b0, b1, b2, b3, b4 = parameters

    print("\nEstimated Parameters")

    print("--------------------")

    print("a  =", a)

    print("b0 =", b0)

    print("b1 =", b1)

    print("b2 =", b2)

    print("b3 =", b3)

    print("b4 =", b4)

    print("Matrix rank:", rank)

    if rank < X_train.shape[1]:

        print(
            "WARNING: Design matrix is rank deficient."
        )

    return parameters


# ==========================================================
# 8. STABILITY ANALYSIS
# ==========================================================

def stability_analysis(parameters):

    print("\nSTABILITY ANALYSIS")
    print("==================")

    a = parameters[0]

    # Scalar state transition matrix
    A = np.array([
        [a]
    ])

    eigenvalues = np.linalg.eigvals(A)

    spectral_radius = np.max(
        np.abs(eigenvalues)
    )

    print("System matrix A:")

    print(A)

    print("Eigenvalues:", eigenvalues)

    print("Spectral radius:", spectral_radius)

    if spectral_radius < 1:

        stability = "ASYMPTOTICALLY STABLE"

    elif spectral_radius > 1:

        stability = "UNSTABLE"

    else:

        stability = "MARGINAL / NON-ASYMPTOTIC"

    print("Stability:", stability)

    return eigenvalues, stability


# ==========================================================
# 9. RANDOM FOREST MODEL
# ==========================================================

def train_random_forest(
    X_train,
    X_test,
    y_train
):

    print("\nRANDOM FOREST TRAINING")
    print("======================")

    # Both models use the same lag and seasonal information
    # The constant intercept column is unnecessary for RF
    feature_columns = [0, 2, 3, 4, 5]

    X_train_rf = X_train[:, feature_columns]

    X_test_rf = X_test[:, feature_columns]

    model = RandomForestRegressor(
        n_estimators=RANDOM_FOREST_TREES,
        max_depth=RANDOM_FOREST_DEPTH,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train_rf,
        y_train
    )

    predictions = model.predict(
        X_test_rf
    )

    print("Random Forest training completed.")

    return model, predictions


# ==========================================================
# 10. MODEL EVALUATION
# ==========================================================

def evaluate_model(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    valid = denominator > 1e-8

    smape = np.mean(
        200 * np.abs(
            actual[valid] - predicted[valid]
        ) / denominator[valid]
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE (%)": smape
    }


def compare_models(
    actual,
    dynamic_predictions,
    ml_predictions
):

    dynamic_results = evaluate_model(
        actual,
        dynamic_predictions
    )

    ml_results = evaluate_model(
        actual,
        ml_predictions
    )

    results = pd.DataFrame([
        {
            "Model": "Dynamical ARX",
            **dynamic_results
        },
        {
            "Model": "Random Forest",
            **ml_results
        }
    ])

    print("\nFORECAST PERFORMANCE COMPARISON")
    print("================================")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    # Relative RMSE improvement
    if ml_results["RMSE"] > 0:

        improvement = (
            (
                ml_results["RMSE"]
                - dynamic_results["RMSE"]
            )
            / ml_results["RMSE"]
        ) * 100

        print(
            "\nARX RMSE improvement relative to RF (%):",
            round(improvement, 3)
        )

    return results


# ==========================================================
# 11. RESIDUAL ANALYSIS
# ==========================================================

def residual_analysis(
    actual,
    dynamic_predictions,
    ml_predictions
):

    print("\nRESIDUAL ANALYSIS")
    print("=================")

    dynamic_residuals = (
        actual - dynamic_predictions
    )

    ml_residuals = (
        actual - ml_predictions
    )

    print(
        "ARX residual mean:",
        np.mean(dynamic_residuals)
    )

    print(
        "ARX residual standard deviation:",
        np.std(dynamic_residuals)
    )

    print(
        "RF residual mean:",
        np.mean(ml_residuals)
    )

    print(
        "RF residual standard deviation:",
        np.std(ml_residuals)
    )

    if len(dynamic_residuals) > 2:

        correlation = np.corrcoef(
            dynamic_residuals[:-1],
            dynamic_residuals[1:]
        )[0, 1]

        print(
            "ARX lag-1 residual correlation:",
            correlation
        )

    return (
        dynamic_residuals,
        ml_residuals
    )


# ==========================================================
# 12. VISUALIZATION
# ==========================================================

def create_visualizations(
    series,
    test_times,
    actual,
    dynamic_predictions,
    ml_predictions,
    eigenvalues,
    dynamic_residuals,
    ml_residuals
):

    print("\nGENERATING VISUALIZATIONS")
    print("=========================")

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 18)
    )

    # ------------------------------------------------------
    # GRAPH 1: Historical electricity demand
    # ------------------------------------------------------

    axes[0].plot(
        series.index,
        series.values,
        linewidth=0.6
    )

    axes[0].set_title(
        "Historical Electricity Demand"
    )

    axes[0].set_xlabel("Date")

    axes[0].set_ylabel("Demand (kW)")

    axes[0].grid(True)

    # ------------------------------------------------------
    # GRAPH 2: Forecast comparison
    # ------------------------------------------------------

    display_count = min(
        168,
        len(actual)
    )

    axes[1].plot(
        test_times[:display_count],
        actual[:display_count],
        label="Actual demand"
    )

    axes[1].plot(
        test_times[:display_count],
        dynamic_predictions[:display_count],
        label="Dynamical ARX"
    )

    axes[1].plot(
        test_times[:display_count],
        ml_predictions[:display_count],
        label="Random Forest"
    )

    axes[1].set_title(
        "Forecast Comparison: First Seven Test Days"
    )

    axes[1].set_xlabel("Time")

    axes[1].set_ylabel("Demand (kW)")

    axes[1].legend()

    axes[1].grid(True)

    # ------------------------------------------------------
    # GRAPH 3: Stability analysis
    # ------------------------------------------------------

    axes[2].scatter(
        np.real(eigenvalues),
        np.imag(eigenvalues),
        s=100,
        label="Eigenvalue"
    )

    # Unit circle for discrete-time stability
    circle = plt.Circle(
        (0, 0),
        1,
        fill=False,
        linestyle="--",
        label="Unit circle"
    )

    axes[2].add_patch(circle)

    axes[2].axhline(
        0,
        linestyle=":",
        linewidth=0.8
    )

    axes[2].axvline(
        0,
        linestyle=":",
        linewidth=0.8
    )

    axes[2].set_xlim(-1.5, 1.5)

    axes[2].set_ylim(-1.5, 1.5)

    axes[2].set_aspect("equal")

    axes[2].set_title(
        "Discrete-Time Eigenvalue Stability"
    )

    axes[2].set_xlabel("Real part")

    axes[2].set_ylabel("Imaginary part")

    axes[2].legend()

    axes[2].grid(True)

    # ------------------------------------------------------
    # GRAPH 4: Residual analysis
    # ------------------------------------------------------

    axes[3].plot(
        test_times,
        dynamic_residuals,
        label="ARX residuals",
        alpha=0.7
    )

    axes[3].plot(
        test_times,
        ml_residuals,
        label="Random Forest residuals",
        alpha=0.7
    )

    axes[3].axhline(
        0,
        linestyle="--"
    )

    axes[3].set_title(
        "Forecast Residual Analysis"
    )

    axes[3].set_xlabel("Time")

    axes[3].set_ylabel("Residual (kW)")

    axes[3].legend()

    axes[3].grid(True)

    plt.tight_layout()

    output_path = (
        DATA_DIR / "experiment3_results.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Saved:", output_path)


# ==========================================================
# 13. MAIN EXPERIMENT
# ==========================================================

def main():

    print("\n")
    print("=" * 65)

    print(
        "EXPERIMENT 3: ELECTRICITY DEMAND "
        "DYNAMICAL SYSTEM IDENTIFICATION"
    )

    print("=" * 65)

    # Load real-world observations
    electricity = load_electricity_data()

    inspect_dataset(electricity)

    # Export processed observations
    electricity.to_csv(
        DATA_DIR / "hourly_electricity.csv",
        header=["demand"]
    )

    # Plot historical demand
    plt.figure(figsize=(13, 5))

    plt.plot(
        electricity.index,
        electricity.values,
        linewidth=0.6
    )

    plt.title(
        "Hourly Electricity Consumption"
    )

    plt.xlabel("Date")

    plt.ylabel("Demand (kW)")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        DATA_DIR / "electricity_demand.png",
        dpi=300
    )

    plt.show()

    # Create features
    X, target, target_times = create_features(
        electricity
    )

    # Chronological split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_mask
    ) = chronological_split(
        electricity,
        X,
        target
    )

    test_times = target_times[test_mask]

    # Dynamical system parameter estimation
    parameters = estimate_dynamical_system(
        X_train,
        y_train
    )

    # Dynamical model predictions
    dynamic_forecast = (
        X_test @ parameters
    )

    # Stability analysis
    eigenvalues, stability = stability_analysis(
        parameters
    )

    # Machine learning model
    rf_model, ml_forecast = train_random_forest(
        X_train,
        X_test,
        y_train
    )

    # Forecast evaluation
    results = compare_models(
        y_test,
        dynamic_forecast,
        ml_forecast
    )

    # Export comparison
    results.to_csv(
        DATA_DIR / "model_comparison.csv",
        index=False
    )

    # Residual analysis
    (
        dynamic_residuals,
        ml_residuals
    ) = residual_analysis(
        y_test,
        dynamic_forecast,
        ml_forecast
    )

    # Export predictions
    predictions = pd.DataFrame({

        "timestamp": test_times,

        "actual_demand": y_test,

        "dynamical_ARX": dynamic_forecast,

        "random_forest": ml_forecast,

        "ARX_residual": dynamic_residuals,

        "RF_residual": ml_residuals

    })

    predictions.to_csv(
        DATA_DIR / "forecast_predictions.csv",
        index=False
    )

    # Export estimated coefficients
    coefficient_names = [
        "a",
        "intercept",
        "daily_sin",
        "daily_cos",
        "weekly_sin",
        "weekly_cos"
    ]

    coefficient_table = pd.DataFrame({

        "Parameter": coefficient_names,

        "Estimated_Value": parameters

    })

    coefficient_table.to_csv(
        DATA_DIR / "estimated_parameters.csv",
        index=False
    )

    # Visualizations
    create_visualizations(
        electricity,
        test_times,
        y_test,
        dynamic_forecast,
        ml_forecast,
        eigenvalues,
        dynamic_residuals,
        ml_residuals
    )

    # Final report
    print("\n")
    print("=" * 65)

    print("EXPERIMENT 3 COMPLETED SUCCESSFULLY")

    print("=" * 65)

    print("\nFINAL RESULTS")

    print("Selected consumer:", CONSUMER_ID)

    print("Estimated coefficient a:", parameters[0])

    print("Stability classification:", stability)

    print("\nGENERATED FILES")

    for file in sorted(DATA_DIR.iterdir()):

        if file.is_file():

            print(
                f"{file.name} "
                f"({file.stat().st_size / 1024:.2f} KB)"
            )

    print("\nAll experiment outputs saved in:")

    print(DATA_DIR.resolve())


# ==========================================================
# 14. EXECUTION ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nExperiment interrupted by user.")

    except Exception as error:

        print("\nEXPERIMENT FAILED")

        print("Error:", str(error))

        raise
