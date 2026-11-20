# GAF-TS Multi-Domain Benchmark (v1.0)

A reproducible benchmark foundation for **traffic, electricity, and weather** time-series forecasting. This is the benchmark/baseline stage; it does not claim that the proposed adaptive GAF-TS method has already been validated.

## Research purpose

Establish a common, auditable experimental protocol across three domains before developing:
1. entity-aware representations,
2. missingness-aware learning,
3. regime-change detection,
4. selective adaptive updates.

## Datasets and sources

### 1) PEMS-BAY — traffic
- 325 traffic sensors, 5-minute traffic-speed observations.
- Zenodo record: https://doi.org/10.5281/zenodo.5146275
- File: `PEMS-BAY.csv`
- DCRNN reference: https://github.com/liyaguang/DCRNN
- Direct file: `https://zenodo.org/records/5146275/files/PEMS-BAY.csv?download=1`
- Published file checksum (MD5): `c8dea58987a5882e946217c22fdb8256`

### 2) ElectricityLoadDiagrams20112014 — energy
- 370 clients, 15-minute consumption, 2011–2014.
- UCI record: https://doi.org/10.24432/C58C86
- Direct archive: `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip`
- Original file is semicolon-delimited; the UCI description states that the source contains no missing values. Do not describe artificial gaps as naturally occurring missing data.
- Values are kW per 15 minutes. Preserve the original unit in this benchmark; optional energy conversion is not performed.

### 3) NOAA GHCN-Daily — weather
- Daily climate observations from stations worldwide.
- Official documentation: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- Station file directory: https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/
- Five fixed US stations are downloaded by default:
  `USW00094728`, `USW00023174`, `USW00012918`, `USW00013874`, `USW00024233`
- Target: daily maximum temperature (`TMAX`), converted from tenths of °C to °C.
- GHCN-Daily is irregular/incomplete across stations; missing days remain missing in the canonical series.

## Canonical schema

Every dataset is converted to a long table:

| column | meaning |
|---|---|
| `timestamp` | timestamp in UTC-naive source time convention |
| `entity_id` | sensor, client, or weather station |
| `target` | traffic speed / electricity load / daily maximum temperature |
| `domain` | traffic / energy / weather |

Original frequencies are retained: 5 minutes, 15 minutes, and daily. The benchmark does **not** force all domains onto one sampling interval.

## Reproducible protocol

- Seed: 42.
- Chronological split per entity: 70% train, 15% validation, 15% test.
- Windows are generated within each entity; windows crossing split boundaries are excluded.
- Default lookback: 24 steps; horizon: 1, 3, and 6 steps.
- This means horizons are domain-relative: traffic = 5/15/30 minutes; energy = 15/45/90 minutes; weather = 1/3/6 days.
- Baselines: persistence, seasonal-naive (daily cycle for traffic/energy; weekly cycle for weather), and XGBoost direct horizon models.
- Metrics: MAE, RMSE, WAPE, sMAPE. MAPE is avoided because zero/near-zero values can make it unstable.
- Scaling is fit on training observations only.
- No random shuffling.
- Natural missingness is reported separately from synthetic missingness.
- The v1 robustness check applies MCAR corruption to engineered test input features; it does not yet simulate raw-series MCAR/MAR/block outages. Ground-truth future targets are never masked.
- The default benchmark run is a reproducibility/smoke benchmark. It is not a claim of statistical significance or state-of-the-art performance.

## Install

Python 3.11 recommended (the user's previous TensorFlow/Python 3.13 issue is avoided; TensorFlow is not required for this benchmark version).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Download and preprocess

```bash
python main.py --prepare
```

This downloads the source files, verifies the PEMS-BAY MD5, parses each dataset, and writes:
- `data/processed/traffic.parquet`
- `data/processed/energy.parquet`
- `data/processed/weather.parquet`
- `reports/dataset_manifest.json`

If parquet support is unavailable, CSV is also written as a fallback.

## Run baseline benchmark

```bash
python main.py --benchmark
```

Quick smoke run:

```bash
python main.py --benchmark --max-entities 8 --max-windows 300
```

Run only one domain:

```bash
python main.py --benchmark --domain traffic
python main.py --benchmark --domain energy
python main.py --benchmark --domain weather
```

## Outputs

- `outputs/benchmark_metrics.csv`
- `outputs/predictions_sample.csv`
- `outputs/domain_coverage.csv`
- `outputs/benchmark_rmse.png`
- `outputs/missingness_robustness.csv`
- `reports/benchmark_report.md`
- `reports/dataset_manifest.json`
- `reports/run_metadata.json`

## Reproducibility notes

- The first preparation run requires internet access and sufficient disk space. Electricity source archive is approximately 249 MB compressed and expands to a much larger text file. For laptop-scale processing, v1 selects 40 traffic sensors and 32 electricity clients deterministically using seed 42; NOAA uses five fixed stations.
- Raw files are not included in this ZIP.
- `data/raw/` and `data/processed/` are excluded from version control by `.gitignore`.
- Store downloaded source checksums, source URLs, date of retrieval, package versions, split timestamps, seeds and configuration in the run metadata.
- Re-run with the same raw files, package versions and configuration to reproduce the benchmark.
- PEMS-BAY and Electricity are not redistributed in this project; download them from their cited sources and follow their dataset terms.
