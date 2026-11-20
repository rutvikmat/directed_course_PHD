
from __future__ import annotations

import hashlib
import json
import platform
import zipfile
import warnings

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

CONFIG_PATH = ROOT / "config" / "benchmark.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}"
    )

CONFIG = json.loads(CONFIG_PATH.read_text())


# ============================================================
# 2. DATASET SOURCES
# ============================================================

# Primary and alternative PEMS-BAY sources.
TRAFFIC_URLS = [
    "https://zenodo.org/records/5146275/files/PEMS-BAY.csv?download=1",
    "https://zenodo.org/records/4264445/files/PEMS-BAY.csv?download=1",
]

ENERGY_URL = (
    "https://archive.ics.uci.edu/static/public/321/"
    "electricityloaddiagrams20112014.zip"
)

NOAA_BASE = (
    "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all"
)

USER_AGENT = "GAF-TS-Research-Benchmark/1.0"

EXPECTED_TRAFFIC_MD5 = CONFIG["traffic_md5"]


# ============================================================
# 3. DIRECTORY MANAGEMENT
# ============================================================

def dirs():

    for directory in [RAW, PROC, OUT, REPORTS]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 4. HTTP SESSION WITH AUTOMATIC RETRIES
# ============================================================

def create_session(retries=5):

    retry_strategy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,

        backoff_factor=2,

        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],

        allowed_methods=frozenset(["GET"]),

        respect_retry_after_header=True
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter
    )

    session.mount(
        "http://",
        adapter
    )

    session.headers.update({
        "User-Agent": USER_AGENT
    })

    return session


# ============================================================
# 5. GENERIC DOWNLOAD FUNCTION
# ============================================================

def get(url, path, retries=8, force=False):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".part")

    if (
        not force
        and path.exists()
        and path.stat().st_size > 1000
    ):
        print(f"Using existing file: {path.name}")
        return path

    if path.exists():
        path.unlink()

    last_error = None

    for attempt in range(1, retries + 1):

        if temp_path.exists():
            temp_path.unlink()

        print(
            f"\nDownload attempt {attempt}/{retries}: {path.name}"
        )

        try:

            with requests.get(
                url,
                stream=True,
                timeout=(30, 180),
                headers={
                    "User-Agent": "GAF-TS-Research-Benchmark/1.0",
                    "Accept-Encoding": "identity"
                }
            ) as response:

                response.raise_for_status()

                expected_size = response.headers.get(
                    "Content-Length"
                )

                expected_size = (
                    int(expected_size)
                    if expected_size
                    else None
                )

                downloaded = 0

                with open(temp_path, "wb") as file:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if chunk:

                            file.write(chunk)
                            downloaded += len(chunk)

                            if expected_size:

                                percentage = (
                                    downloaded / expected_size
                                ) * 100

                                print(
                                    f"\rProgress: {percentage:.1f}%",
                                    end="",
                                    flush=True
                                )

                print()

            if not temp_path.exists():
                raise RuntimeError("Temporary download file missing.")

            actual_size = temp_path.stat().st_size

            if actual_size == 0:
                raise RuntimeError("Downloaded file is empty.")

            if (
                expected_size is not None
                and actual_size != expected_size
            ):
                raise RuntimeError(
                    f"Incomplete download: received {actual_size} "
                    f"bytes; expected {expected_size} bytes."
                )

            temp_path.replace(path)

            print(f"Download successful: {path.name}")
            print(
                f"File size: {path.stat().st_size / (1024 * 1024):.2f} MB"
            )

            return path

        except (
            requests.RequestException,
            OSError,
            RuntimeError
        ) as exc:

            last_error = exc

            print(f"\nDownload interrupted: {exc}")

            if temp_path.exists():
                temp_path.unlink()

            if attempt < retries:

                delay = min(
                    5 * (2 ** (attempt - 1)),
                    60
                )

                print(
                    f"Retrying the complete download in {delay} seconds..."
                )

                time.sleep(delay)

    raise RuntimeError(
        f"Download failed after {retries} attempts: {url}\n"
        f"Last error: {last_error}"
    ) from last_error

# ============================================================
# 6. CHECKSUM UTILITIES
# ============================================================

def md5(path):

    digest = hashlib.md5()

    with open(path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            digest.update(block)

    return digest.hexdigest()


def download_verified_traffic(path):

    """
    Download PEMS-BAY from primary or alternative sources.

    Each downloaded file is validated using the expected MD5.
    Invalid files are deleted before trying the next source.
    """

    path = Path(path)

    # Check whether a valid copy already exists.
    if path.exists() and path.stat().st_size > 1000:

        existing_hash = md5(path)

        if existing_hash == EXPECTED_TRAFFIC_MD5:

            print(
                "\nExisting PEMS-BAY file passed MD5 validation."
            )

            return path

        print(
            "\nExisting PEMS-BAY file has an invalid checksum."
        )

        path.unlink()

    errors = []

    for index, url in enumerate(TRAFFIC_URLS, start=1):

        print(
            f"\nPEMS-BAY source attempt {index}/{len(TRAFFIC_URLS)}"
        )

        try:

            get(
                url,
                path,
                retries=5,
                force=True
            )

            actual_hash = md5(path)

            print(
                f"Downloaded MD5: {actual_hash}"
            )

            print(
                f"Expected MD5:   {EXPECTED_TRAFFIC_MD5}"
            )

            if actual_hash != EXPECTED_TRAFFIC_MD5:

                path.unlink(missing_ok=True)

                raise ValueError(
                    "PEMS-BAY checksum validation failed."
                )

            print(
                "\nPEMS-BAY checksum validation successful."
            )

            return path

        except Exception as exc:

            errors.append(str(exc))

            print(
                f"Source {index} failed: {exc}"
            )

            path.unlink(missing_ok=True)

    raise RuntimeError(
        "Unable to download a valid PEMS-BAY dataset.\n"
        + "\n".join(errors)
    )


# ============================================================
# 7. TABLE STORAGE UTILITIES
# ============================================================

def save_table(df, stem):

    parquet_path = PROC / f"{stem}.parquet"

    try:

        df.to_parquet(
            parquet_path,
            index=False
        )

        return str(
            parquet_path.relative_to(ROOT)
        )

    except Exception as exc:

        warnings.warn(
            f"Parquet unavailable; saving CSV instead: {exc}"
        )

        csv_path = PROC / f"{stem}.csv"

        df.to_csv(
            csv_path,
            index=False
        )

        return str(
            csv_path.relative_to(ROOT)
        )


def read_table(stem):

    parquet_path = PROC / f"{stem}.parquet"

    csv_path = PROC / f"{stem}.csv"

    if parquet_path.exists():

        return pd.read_parquet(parquet_path)

    if csv_path.exists():

        return pd.read_csv(
            csv_path,
            parse_dates=["timestamp"]
        )

    raise FileNotFoundError(
        f"Processed dataset not found: {stem}. "
        "Run python main.py --prepare first."
    )


# ============================================================
# 8. PEMS-BAY TRAFFIC DATASET
# ============================================================

def download_traffic():

    path = RAW / "PEMS-BAY.csv"

    download_verified_traffic(path)

    df = pd.read_csv(
        path,
        index_col=0
    )

    # The published file normally uses its first column
    # as the timestamp index.
    idx = pd.to_datetime(
        df.index,
        errors="coerce"
    )

    if idx.notna().mean() < 0.95:

        df0 = pd.read_csv(path)

        time_col = next(
            (
                column
                for column in df0.columns
                if (
                    "time" in str(column).lower()
                    or "date" in str(column).lower()
                )
            ),
            None
        )

        if time_col is None:

            raise ValueError(
                "Cannot identify timestamp column in PEMS-BAY CSV."
            )

        idx = pd.to_datetime(
            df0.pop(time_col),
            errors="coerce"
        )

        df = df0

    df.index = idx

    df = df.loc[
        df.index.notna()
    ]

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # DCRNN-style preprocessing convention:
    # zero traffic speed readings are treated as missing.
    df = df.mask(df <= 0)

    # Deterministic selection of up to 40 sensors.
    columns = sorted(
        df.columns.astype(str)
    )

    rng = np.random.default_rng(
        CONFIG["seed"]
    )

    selected = sorted(
        rng.choice(
            columns,
            size=min(40, len(columns)),
            replace=False
        ).tolist()
    )

    # Ensure selected column labels match the dataframe labels.
    df.columns = df.columns.astype(str)

    df = df[selected]

    long = (
        df.rename_axis("timestamp")
        .reset_index()
        .melt(
            id_vars="timestamp",
            var_name="entity_id",
            value_name="target"
        )
    )

    long["domain"] = "traffic"

    return long.dropna(
        subset=["timestamp"]
    )


# ============================================================
# 9. UCI ELECTRICITY DATASET
# ============================================================

def download_energy():

    path = RAW / "electricityloaddiagrams20112014.zip"

    get(
        ENERGY_URL,
        path,
        retries=8
    )

    # Verify the downloaded ZIP before extracting.
    if not zipfile.is_zipfile(path):

        path.unlink(missing_ok=True)

        raise RuntimeError(
            "Electricity dataset download is not a valid ZIP file. "
            "Please rerun preparation."
        )

    with zipfile.ZipFile(path) as archive:

        corrupted_file = archive.testzip()

        if corrupted_file is not None:

            path.unlink(missing_ok=True)

            raise RuntimeError(
                f"Corrupted ZIP member detected: {corrupted_file}"
            )

        names = [
            name
            for name in archive.namelist()
            if (
                name.lower().endswith(".txt")
                and "ld2011_2014" in name.lower()
            )
        ]

        if not names:

            raise ValueError(
                "Could not find LD2011_2014.txt in the UCI archive."
            )

        with archive.open(names[0]) as file:

            df = pd.read_csv(
                file,
                sep=";",
                decimal=",",
                low_memory=False
            )

    time_col = df.columns[0]

    df[time_col] = pd.to_datetime(
        df[time_col],
        errors="coerce"
    )

    df = (
        df.dropna(subset=[time_col])
        .set_index(time_col)
        .sort_index()
    )

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    long = (
        df.rename_axis("timestamp")
        .reset_index()
        .melt(
            id_vars="timestamp",
            var_name="entity_id",
            value_name="target"
        )
    )

    long["domain"] = "energy"

    return long.dropna(
        subset=["timestamp"]
    )
# ============================================================
# 10. NOAA GHCN-DAILY DATASET
# ============================================================

def parse_ghcn_dly(path, station):

    """
    Parse NOAA GHCN-Daily .dly fixed-width files.

    Record format:
    ID       11 characters
    YEAR      4 characters
    MONTH     2 characters
    ELEMENT   4 characters
    DAYS     31 x 8 characters

    Extracts TMAX and converts tenths of Celsius to Celsius.
    """

    widths = (
        [11, 4, 2, 4]
        + [8] * 31
    )

    names = (
        [
            "station",
            "year",
            "month",
            "element"
        ]
        + [
            f"d{day}"
            for day in range(1, 32)
        ]
    )

    raw = pd.read_fwf(
        path,
        widths=widths,
        names=names,
        dtype=str
    )

    raw = raw[
        raw["element"] == "TMAX"
    ]

    records = []

    for _, row in raw.iterrows():

        try:

            year = int(row["year"])
            month = int(row["month"])

        except (ValueError, TypeError):

            continue

        if not 1 <= month <= 12:
            continue

        for day in range(1, 32):

            token = row.get(
                f"d{day}"
            )

            if (
                not isinstance(token, str)
                or len(token) < 8
            ):

                continue

            try:

                value = int(
                    token[:5]
                )

                # GHCN quality flag is the seventh character.
                qflag = token[6:7]

                # -9999 represents missing observations.
                # Non-empty quality flags are excluded.
                if value == -9999:
                    continue

                if qflag.strip():
                    continue

                date = pd.Timestamp(
                    year=year,
                    month=month,
                    day=day
                )

                records.append(
                    (
                        date,
                        station,
                        value / 10.0
                    )
                )

            except (
                ValueError,
                TypeError,
                OverflowError
            ):

                # Invalid calendar dates are skipped.
                continue

    return pd.DataFrame(
        records,
        columns=[
            "timestamp",
            "entity_id",
            "target"
        ]
    )


def download_weather():

    pieces = []

    stations = CONFIG["noaa_stations"]

    for index, station in enumerate(
        stations,
        start=1
    ):

        print(
            f"\nNOAA station {index}/{len(stations)}: {station}"
        )

        path = RAW / f"{station}.dly"

        url = f"{NOAA_BASE}/{station}.dly"

        get(
            url,
            path,
            retries=5
        )

        station_data = parse_ghcn_dly(
            path,
            station
        )

        print(
            f"Valid TMAX records: {len(station_data)}"
        )

        pieces.append(
            station_data
        )

    df = pd.concat(
        pieces,
        ignore_index=True
    )

    df["domain"] = "weather"

    return df


# ============================================================
# 11. DATASET PREPARATION
# ============================================================

def prepare_all():

    dirs()

    print("\n" + "=" * 65)
    print("GAF-TS MULTI-DOMAIN DATA PREPARATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    print("\n[1/3] Preparing PEMS-BAY traffic...")

    traffic = download_traffic()

    print(
        "Traffic shape:",
        traffic.shape
    )

    print(
        "Traffic entities:",
        traffic["entity_id"].nunique()
    )

    traffic_path = save_table(
        traffic,
        "traffic"
    )

    # --------------------------------------------------------
    # Electricity
    # --------------------------------------------------------

    print("\n[2/3] Preparing UCI electricity...")

    energy = download_energy()

    print(
        "Energy shape:",
        energy.shape
    )

    print(
        "Energy entities:",
        energy["entity_id"].nunique()
    )

    energy_path = save_table(
        energy,
        "energy"
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    print("\n[3/3] Preparing NOAA weather...")

    weather = download_weather()

    print(
        "Weather shape:",
        weather.shape
    )

    print(
        "Weather entities:",
        weather["entity_id"].nunique()
    )

    weather_path = save_table(
        weather,
        "weather"
    )

    # --------------------------------------------------------
    # Dataset manifest
    # --------------------------------------------------------

    manifest = {}

    datasets = [
        (
            "traffic",
            traffic,
            traffic_path
        ),
        (
            "energy",
            energy,
            energy_path
        ),
        (
            "weather",
            weather,
            weather_path
        )
    ]

    for name, df, processed_path in datasets:

        manifest[name] = {

            "processed_path": processed_path,

            "entity_sampling": (
                "seeded deterministic subset for laptop-scale benchmark"
                if name in ("traffic", "energy")
                else "fixed five NOAA stations"
            ),

            "seed": CONFIG["seed"],

            "rows": int(len(df)),

            "entities": int(
                df["entity_id"].nunique()
            ),

            "start": str(
                df["timestamp"].min()
            ),

            "end": str(
                df["timestamp"].max()
            ),

            "natural_missing_target_rows": int(
                df["target"].isna().sum()
            ),

            "frequency": {
                "traffic": "5min",
                "energy": "15min",
                "weather": "daily"
            }[name],

            "target_unit": {
                "traffic": "dataset traffic-speed unit",
                "energy": "kW per 15-minute interval",
                "weather": "degrees Celsius (TMAX)"
            }[name]
        }

    manifest["sources"] = {

        "traffic": (
            "Zenodo 10.5281/zenodo.5146275; PEMS-BAY.csv"
        ),

        "energy": (
            "UCI 10.24432/C58C86; ElectricityLoadDiagrams20112014"
        ),

        "weather": (
            "NOAA GHCN-Daily station .dly files; TMAX"
        )
    }

    manifest["raw_file_checksums_md5"] = {

        file.name: md5(file)

        for file in sorted(RAW.iterdir())

        if (
            file.is_file()
            and not file.name.endswith(".part")
        )
    }

    manifest["retrieved_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    manifest_path = (
        REPORTS / "dataset_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        )
    )

    print("\n" + "=" * 65)
    print("DATA PREPARATION COMPLETED")
    print("=" * 65)

    print(
        f"Manifest saved: {manifest_path}"
    )


# ============================================================
# 12. LOAD PROCESSED DATA
# ============================================================

def load_domain(domain):

    df = read_table(domain)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce"
    )

    return df.sort_values(
        [
            "entity_id",
            "timestamp"
        ]
    )


# ============================================================
# 13. CAUSAL FEATURE ENGINEERING
# ============================================================

def make_features(hist, timestamps, domain):

    """
    Generate features using only historical observations.

    No future observations are used in feature construction.
    """

    features = {}

    values = np.asarray(
        hist,
        dtype=float
    )

    # Lag features.
    for lag in [1, 2, 3, 6, 12, 24]:

        features[f"lag_{lag}"] = (
            values[-lag]
            if len(values) >= lag
            else np.nan
        )

    # Rolling statistics.
    for window in [3, 6, 12, 24]:

        subset = values[-window:]

        if np.isfinite(subset).any():

            features[f"mean_{window}"] = np.nanmean(
                subset
            )

            features[f"std_{window}"] = np.nanstd(
                subset
            )

        else:

            features[f"mean_{window}"] = np.nan
            features[f"std_{window}"] = np.nan

    timestamp = pd.Timestamp(
        timestamps
    )

    # Time features.
    features["hour_sin"] = np.sin(
        2 * np.pi * timestamp.hour / 24
    )

    features["hour_cos"] = np.cos(
        2 * np.pi * timestamp.hour / 24
    )

    features["dow_sin"] = np.sin(
        2 * np.pi * timestamp.dayofweek / 7
    )

    features["dow_cos"] = np.cos(
        2 * np.pi * timestamp.dayofweek / 7
    )

    features["doy_sin"] = np.sin(
        2 * np.pi * timestamp.dayofyear / 365.25
    )

    features["doy_cos"] = np.cos(
        2 * np.pi * timestamp.dayofyear / 365.25
    )

    recent = values[-24:]

    features["missing_fraction_lookback"] = float(
        np.mean(~np.isfinite(recent))
    )

    return features


# ============================================================
# 14. ENTITY-LEVEL WINDOW GENERATION
# ============================================================

def entity_windows(
    g,
    lookback,
    horizons,
    domain,
    max_windows=None
):

    g = (
        g.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")
    )

    # Retain native sampling frequency.
    frequency = {
        "traffic": "5min",
        "energy": "15min",
        "weather": "D"
    }[domain]

    index = pd.date_range(
        g.index.min(),
        g.index.max(),
        freq=frequency
    )

    series = g["target"].reindex(index)

    n = len(series)

    max_horizon = max(horizons)

    # Chronological splits.
    train_end = int(
        n * CONFIG["train_fraction"]
    )

    validation_end = int(
        n * (
            CONFIG["train_fraction"]
            + CONFIG["validation_fraction"]
        )
    )

    ranges = {

        "train": (
            0,
            train_end
        ),

        "validation": (
            train_end,
            validation_end
        ),

        "test": (
            validation_end,
            n
        )
    }

    rows = []

    seasonal_period = {
        "traffic": 288,
        "energy": 96,
        "weather": 7
    }[domain]

    for split, (low, high) in ranges.items():

        first = max(
            low + lookback,
            lookback
        )

        last = high - max_horizon

        if last <= first:
            continue

        candidates = np.arange(
            first,
            last
        )

        # Optional deterministic window sampling.
        if (
            max_windows
            and len(candidates) > max_windows
        ):

            rng = np.random.default_rng(
                CONFIG["seed"]
                + sum(
                    map(
                        ord,
                        str(g["entity_id"].iloc[0])
                    )
                )
            )

            candidates = np.sort(
                rng.choice(
                    candidates,
                    size=max_windows,
                    replace=False
                )
            )

        for position in candidates:

            history = series.iloc[
                position - lookback:position
            ].to_numpy(dtype=float)

            future = series.iloc[
                position + np.array(horizons) - 1
            ].to_numpy(dtype=float)

            # Forecast targets must be available.
            if not np.isfinite(future).all():
                continue

            feature = make_features(
                history,
                index[position],
                domain
            )

            # Seasonal-naive references.
            for horizon in horizons:

                seasonal_position = (
                    position
                    + horizon
                    - 1
                    - seasonal_period
                )

                if (
                    seasonal_position >= 0
                    and np.isfinite(
                        series.iloc[seasonal_position]
                    )
                ):

                    feature[
                        f"seasonal_naive_{horizon}"
                    ] = float(
                        series.iloc[seasonal_position]
                    )

                else:

                    feature[
                        f"seasonal_naive_{horizon}"
                    ] = np.nan

            feature["entity_id"] = str(
                g["entity_id"].iloc[0]
            )

            feature["domain"] = domain

            feature["split"] = split

            feature["origin"] = index[position]

            for horizon, target in zip(
                horizons,
                future
            ):

                feature[
                    f"target_{horizon}"
                ] = target

            rows.append(feature)

    return rows


# ============================================================
# 15. EVALUATION METRICS
# ============================================================

def metrics(y, prediction):

    y = np.asarray(
        y,
        dtype=float
    )

    prediction = np.asarray(
        prediction,
        dtype=float
    )

    valid = (
        np.isfinite(y)
        & np.isfinite(prediction)
    )

    y = y[valid]
    prediction = prediction[valid]

    if len(y) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    mae = float(
        mean_absolute_error(
            y,
            prediction
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y,
                prediction
            )
        )
    )

    denominator = np.sum(
        np.abs(y)
    )

    wape = (
        float(
            100 * np.sum(
                np.abs(y - prediction)
            ) / denominator
        )
        if denominator > 0
        else np.nan
    )

    smape = float(
        100 * np.mean(
            2 * np.abs(y - prediction)
            / np.maximum(
                np.abs(y) + np.abs(prediction),
                1e-8
            )
        )
    )

    return mae, rmse, wape, smape


# ============================================================
# 16. SYNTHETIC MISSINGNESS
# ============================================================

def apply_synthetic_missingness(
    X,
    rate,
    seed,
    block=False
):

    x = np.asarray(
        X,
        dtype=float
    ).copy()

    rng = np.random.default_rng(seed)

    if x.size == 0:
        return x

    if block:

        for row in range(len(x)):

            start = int(
                rng.integers(
                    0,
                    max(1, x.shape[1] - 3)
                )
            )

            width = max(
                1,
                int(
                    round(
                        x.shape[1] * rate
                    )
                )
            )

            x[
                row,
                start:min(
                    x.shape[1],
                    start + width
                )
            ] = np.nan

    else:

        mask = rng.random(
            x.shape
        ) < rate

        x[mask] = np.nan

    return x


# ============================================================
# 17. BENCHMARK EXECUTION
# ============================================================

def run_benchmark(
    domain="all",
    max_entities=None,
    max_windows=None
):

    dirs()

    domains = (
        ["traffic", "energy", "weather"]
        if domain == "all"
        else [domain]
    )

    for dom in domains:

        if dom not in [
            "traffic",
            "energy",
            "weather"
        ]:

            raise ValueError(
                f"Unsupported domain: {dom}"
            )

    horizons = CONFIG["horizons"]

    all_metrics = []
    all_predictions = []
    coverage = []

    for dom in domains:

        print(
            f"\n{'=' * 55}\n"
            f"BENCHMARK: {dom.upper()}\n"
            f"{'=' * 55}"
        )

        df = load_domain(dom)

        entities = sorted(
            df["entity_id"].astype(str).unique()
        )

        entity_limit = (
            max_entities
            if max_entities is not None
            else CONFIG["max_entities_default"]
        )

        if entity_limit is not None:

            entities = entities[:entity_limit]

        df = df[
            df["entity_id"].astype(str).isin(
                entities
            )
        ]

        rows = []

        domain_lookback = {
            "traffic": 294,
            "energy": 102,
            "weather": 13
        }[dom]

        window_limit = (
            max_windows
            if max_windows is not None
            else CONFIG["max_windows_default"]
        )

        for entity, group in df.groupby(
            "entity_id",
            sort=True
        ):

            print(
                f"Building windows: {entity}"
            )

            rows.extend(
                entity_windows(
                    group,
                    domain_lookback,
                    horizons,
                    dom,
                    max_windows=window_limit
                )
            )

        data = pd.DataFrame(rows)

        if data.empty:

            print(
                f"No eligible windows for {dom}."
            )

            continue

        feature_cols = [
            column
            for column in data.columns
            if column.startswith(
                (
                    "lag_",
                    "mean_",
                    "std_",
                    "hour_",
                    "dow_",
                    "doy_",
                    "missing_fraction_",
                    "seasonal_naive_"
                )
            )
        ]

        train = data[
            data["split"] == "train"
        ]

        validation = data[
            data["split"] == "validation"
        ]

        test = data[
            data["split"] == "test"
        ]

        if train.empty or test.empty:

            print(
                f"Insufficient train/test windows for {dom}."
            )

            continue

        coverage.append({

            "domain": dom,

            "entities_used": len(entities),

            "windows_train": len(train),

            "windows_validation": len(validation),

            "windows_test": len(test),

            "natural_missing_rate": float(
                df["target"].isna().mean()
            ),

            "frequency": {
                "traffic": "5min",
                "energy": "15min",
                "weather": "daily"
            }[dom]
        })

        # ----------------------------------------------------
        # Forecasting models
        # ----------------------------------------------------

        for horizon in horizons:

            print(
                f"\nForecast horizon: {horizon}"
            )

            target_col = f"target_{horizon}"

            X_train = train[feature_cols]
            y_train = train[target_col]

            X_test = test[feature_cols]
            y_test = test[target_col]

            # Imputer is fitted only on training data.
            imputer = SimpleImputer(
                strategy="median"
            )

            X_train_i = imputer.fit_transform(
                X_train
            )

            X_test_i = imputer.transform(
                X_test
            )

            model = XGBRegressor(

                n_estimators=120,

                max_depth=5,

                learning_rate=0.04,

                subsample=0.85,

                colsample_bytree=0.85,

                objective="reg:squarederror",

                random_state=CONFIG["seed"],

                n_jobs=4
            )

            model.fit(
                X_train_i,
                y_train
            )

            xgb_prediction = model.predict(
                X_test_i
            )

            # Persistence baseline.
            persistence_prediction = (
                test["lag_1"]
                .fillna(
                    y_train.median()
                )
                .to_numpy()
            )

            # Seasonal naive baseline.
            seasonal_prediction = (
                test[
                    f"seasonal_naive_{horizon}"
                ].to_numpy()
            )

            model_predictions = [

                (
                    "XGBoost",
                    xgb_prediction
                ),

                (
                    "Persistence",
                    persistence_prediction
                ),

                (
                    "SeasonalNaive",
                    seasonal_prediction
                )
            ]

            actual = y_test.to_numpy()

            for model_name, prediction in model_predictions:

                valid = (
                    np.isfinite(actual)
                    & np.isfinite(prediction)
                )

                if valid.sum() == 0:
                    continue

                mae, rmse, wape, smape = metrics(
                    actual[valid],
                    prediction[valid]
                )

                all_metrics.append({

                    "domain": dom,

                    "model": model_name,

                    "horizon_steps": horizon,

                    "horizon_native": (
                        {
                            "traffic": 5,
                            "energy": 15,
                            "weather": 1440
                        }[dom] * horizon
                    ),

                    "horizon_unit": "minutes",

                    "n_test": int(
                        valid.sum()
                    ),

                    "MAE": mae,

                    "RMSE": rmse,

                    "WAPE_pct": wape,

                    "sMAPE_pct": smape
                })

                if model_name == "XGBoost":

                    sample = test.loc[
                        valid,
                        [
                            "entity_id",
                            "origin",
                            target_col
                        ]
                    ].copy()

                    sample["domain"] = dom

                    sample["model"] = model_name

                    sample["horizon_steps"] = horizon

                    sample["actual"] = actual[valid]

                    sample["prediction"] = prediction[valid]

                    all_predictions.append(
                        sample.drop(
                            columns=[target_col]
                        )
                    )

        # ----------------------------------------------------
        # Missingness robustness experiment
        # ----------------------------------------------------

        print(
            "\nRunning missingness robustness experiment..."
        )

        horizon = 1

        target_col = f"target_{horizon}"

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_i = imputer.fit_transform(
            train[feature_cols]
        )

        X_test = test[feature_cols].to_numpy()

        model = XGBRegressor(

            n_estimators=120,

            max_depth=5,

            learning_rate=0.04,

            subsample=0.85,

            colsample_bytree=0.85,

            objective="reg:squarederror",

            random_state=CONFIG["seed"],

            n_jobs=4
        )

        model.fit(
            X_train_i,
            train[target_col]
        )

        robustness_results = []

        actual = test[target_col].to_numpy()

        for rate in CONFIG["mcar_rates"]:

            corrupted = apply_synthetic_missingness(

                X_test,

                rate,

                CONFIG["seed"] + int(rate * 1000)
            )

            corrupted_i = imputer.transform(
                corrupted
            )

            prediction = model.predict(
                corrupted_i
            )

            mae, rmse, wape, smape = metrics(
                actual,
                prediction
            )

            robustness_results.append({

                "domain": dom,

                "missing_rate": rate,

                "mechanism": "MCAR_engineered_features",

                "horizon_steps": 1,

                "MAE": mae,

                "RMSE": rmse,

                "WAPE_pct": wape,

                "sMAPE_pct": smape
            })

        pd.DataFrame(
            robustness_results
        ).to_csv(
            OUT / f"missingness_{dom}.csv",
            index=False
        )

        print(
            f"\n{dom.upper()} completed."
        )

        print(
            f"Train windows: {len(train)}"
        )

        print(
            f"Validation windows: {len(validation)}"
        )

        print(
            f"Test windows: {len(test)}"
        )

    # ========================================================
    # 18. SAVE BENCHMARK RESULTS
    # ========================================================

    if not all_metrics:

        raise RuntimeError(
            "No benchmark metrics generated. "
            "Run python main.py --prepare first."
        )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_df.to_csv(
        OUT / "benchmark_metrics.csv",
        index=False
    )

    coverage_df = pd.DataFrame(
        coverage
    )

    coverage_df.to_csv(
        OUT / "domain_coverage.csv",
        index=False
    )

    if all_predictions:

        predictions_df = pd.concat(
            all_predictions,
            ignore_index=True
        )

        predictions_df.to_csv(
            OUT / "predictions_sample.csv",
            index=False
        )

    # Combine domain-wise missingness reports.
    robustness_files = list(
        OUT.glob("missingness_*.csv")
    )

    if robustness_files:

        robustness_df = pd.concat(
            [
                pd.read_csv(file)
                for file in robustness_files
            ],
            ignore_index=True
        )

        robustness_df.to_csv(
            OUT / "missingness_robustness.csv",
            index=False
        )

    # Generate plots.
    plot_metrics(
        metrics_df
    )

    # Generate research report.
    write_report(
        metrics_df,
        coverage
    )

    # ========================================================
    # 19. RUN METADATA
    # ========================================================

    metadata = {

        "run_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "python": platform.python_version(),

        "numpy": np.__version__,

        "pandas": pd.__version__,

        "seed": CONFIG["seed"],

        "domain": domain,

        "max_entities": max_entities,

        "max_windows": max_windows,

        "lookback_steps": {

            "traffic": 294,

            "energy": 102,

            "weather": 13
        },

        "horizons_steps": horizons,

        "split_fractions": [

            CONFIG["train_fraction"],

            CONFIG["validation_fraction"],

            1
            - CONFIG["train_fraction"]
            - CONFIG["validation_fraction"]
        ],

        "models": [

            "Persistence",

            "SeasonalNaive",

            "XGBoost"
        ],

        "note": (
            "Benchmark v1 baseline. "
            "Adaptive GAF-TS is not implemented in this version."
        )
    }

    (
        REPORTS / "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        )
    )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65)

    print(
        "\nBenchmark metrics:"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print(
        "\nResults saved inside:",
        OUT
    )


# ============================================================
# 20. VISUALIZATION
# ============================================================

def plot_metrics(metrics_df):

    for metric in [
        "RMSE",
        "MAE"
    ]:

        pivot = metrics_df.pivot_table(

            index=[
                "domain",
                "horizon_steps"
            ],

            columns="model",

            values=metric
        )

        if pivot.empty:
            continue

        ax = pivot.plot(

            kind="bar",

            figsize=(12, 5)
        )

        ax.set_title(
            f"Multi-domain baseline comparison — {metric}"
        )

        ax.set_ylabel(metric)

        ax.set_xlabel(
            "Domain / horizon steps"
        )

        plt.tight_layout()

        plt.savefig(

            OUT / f"benchmark_{metric.lower()}.png",

            dpi=160
        )

        plt.close()


# ============================================================
# 21. RESEARCH REPORT GENERATION
# ============================================================

def write_report(metrics_df, coverage):

    lines = [

        "# GAF-TS Multi-Domain Benchmark Report",

        "",

        "## 1. Scope",

        (
            "This report establishes the baseline benchmark "
            "for traffic, electricity and weather forecasting."
        ),

        (
            "It does not claim that the adaptive GAF-TS "
            "architecture is complete."
        ),

        "",

        "## 2. Dataset Coverage",

        pd.DataFrame(
            coverage
        ).to_markdown(
            index=False
        ),

        "",

        "## 3. Forecasting Results",

        metrics_df.to_markdown(
            index=False
        ),

        "",

        "## 4. Interpretation",

        (
            "- Compare models within each domain and horizon. "
            "Raw MAE/RMSE values are not directly comparable "
            "across different units."
        ),

        (
            "- Electricity data are described as having no "
            "missing values by UCI; some clients' pre-activation "
            "periods are represented as zero."
        ),

        (
            "- Artificial masking is a controlled robustness "
            "test, not natural missingness."
        ),

        (
            "- Synthetic missingness in v1 is applied to "
            "engineered test input features, not raw source "
            "observations."
        ),

        (
            "- Weather observations retain naturally missing "
            "dates and quality-flagged values are excluded."
        ),

        (
            "- Traffic zero readings are treated as missing "
            "following common DCRNN preprocessing conventions. "
            "This assumption should be checked in a sensitivity analysis."
        ),

        (
            "- The benchmark uses entity-wise chronological "
            "splits and does not shuffle timestamps."
        ),

        "",

        "## 5. Limitations",

        (
            "- Benchmark v1 includes Persistence, "
            "SeasonalNaive and XGBoost baselines."
        ),

        (
            "- ARIMA, LSTM, entity embeddings, regime detection "
            "and adaptive updating are future stages."
        ),

        (
            "- This implementation samples entities and windows "
            "for tractable local execution."
        ),

        (
            "- The three datasets have different frequencies "
            "and units. Horizons are expressed in native steps."
        ),

        "",

        "## 6. Next Experiments",

        "1. Add ARIMA and additional statistical baselines.",

        "2. Add global LSTM and entity embedding models.",

        "3. Compare MCAR, MAR and block missingness.",

        "4. Add ADWIN/change-point detection and measure detection delay.",

        "5. Add adaptive updates and cross-domain transfer.",

        "6. Evaluate per-entity forecasting performance.",

        "",

        "## 7. Reproducibility",

        (
            "Dataset checksums, run configuration, "
            "software versions and benchmark outputs are "
            "recorded in the generated manifest and metadata files."
        )
    ]

    report_path = (
        REPORTS / "benchmark_report.md"
    )

    report_path.write_text(
        "\n".join(lines) + "\n"
    )

    print(
        f"\nResearch report generated: {report_path}"
    )
