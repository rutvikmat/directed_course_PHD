# Module 4: Explainable AI, Trustworthy Prediction and Model Reliability

**Complete coursework package** — practical implementation, reproducible experiments, generated results, report, and research assignment.

## Scope
This package uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset bundled with scikit-learn. The task is binary classification (malignant vs benign) using:
- Logistic Regression as the simpler interpretable reference model.
- Random Forest as the primary non-linear ensemble model.
- SHAP explanations, when the `shap` package is installed.
- Permutation importance as a model-agnostic cross-check.
- Calibration, subgroup performance, and explanation stability experiments.

**Important:** This is an educational benchmark using a public, de-identified diagnostic dataset. It is not a clinical decision-support system, does not establish clinical utility, and must not be used for diagnosis or treatment.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
python src/run_all.py
```

The pipeline downloads no data. It loads the dataset shipped with scikit-learn. Outputs are written to `results/`.

## Main outputs
- `results/model_metrics.csv`
- `results/classification_reports.txt`
- `results/calibration_metrics.csv`
- `results/subgroup_metrics.csv`
- `results/permutation_importance.csv`
- `results/shap_global_importance.csv` and SHAP plots (if SHAP is installed)
- `results/explanation_stability.csv`
- `results/experiment_summary.json`

## Reproducibility
- Fixed random seed: 42
- Stratified train/test split: 80/20
- Preprocessing is fitted on training data only.
- All reported test metrics use the held-out test set.
- The test set is not used for model fitting or threshold selection.

## Folder guide
- `src/`: executable Python pipeline.
- `data/`: dataset notes (the dataset is loaded from scikit-learn).
- `results/`: generated tables, plots, and logs.
- `report/`: coursework report.
- `research_assignment/`: proposed explanation reliability research method.
- `docs/`: syllabus mapping and submission checklist.

## Re-run
From the project root:

```bash
python src/run_all.py
```

If SHAP cannot be imported due to a local dependency conflict, the pipeline still completes the non-SHAP experiments and reports the issue. Install a compatible SHAP/scikit-learn/numpy combination and rerun to generate SHAP plots.
