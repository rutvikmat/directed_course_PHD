# Module 5 Practical: Explainable, Fair and Privacy-Aware Healthcare Risk Prediction

**Purpose:** A reproducible educational prototype covering healthcare-style data preparation, predictive modelling, evaluation, explanations, subgroup analysis, and a privacy-preserving extension.

## Important scope statement
The default dataset is **synthetic**, generated locally with a fixed random seed. It contains no real patient records and is not suitable for clinical use. The sensitive-looking attributes (age and sex) are simulated for demonstrating subgroup evaluation. Results are methodological demonstrations, not medical evidence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
python src/run_pipeline.py
```

Outputs are written to `results/`:
- model comparison and per-model classification reports
- confusion matrices and ROC curves
- feature importance and permutation importance
- subgroup metrics by age band and sex
- a clearly labelled privacy-extension simulation
- run summary and reproducibility metadata

## Dataset options
By default, the pipeline generates a deterministic synthetic dataset. To use a CSV:
```bash
python src/run_pipeline.py --data data/your_dataset.csv --target outcome
```
The CSV must contain a binary target column (`0`/`1`). All other columns are treated as candidate predictors. If demographic columns named `age` and/or `sex` exist, subgroup analysis uses them. Do not use identifiable patient data. Only use datasets whose licence and ethics conditions permit your intended work.

## Methods
- Logistic Regression, Random Forest, and HistGradientBoosting (scikit-learn; no external download required).
- Stratified train/test split and preprocessing fitted only on training data.
- Accuracy, precision, recall, F1, ROC-AUC, Brier score, confusion matrix and classification report.
- Permutation importance and model-agnostic local explanation using feature perturbation. If SHAP is installed, an optional SHAP summary plot is attempted; the pipeline remains functional without it.
- Subgroup metrics: sample count, prevalence, selection rate, TPR, FPR, precision and accuracy.
- Privacy extension: a **toy clipped-and-noised aggregate demonstration**. It is not a complete differentially private training algorithm and does not claim an epsilon/delta guarantee. See `docs/PRIVACY_EXTENSION.md`.

## Reproducibility
The random seed is fixed in `src/config.py`. The script saves package versions, data dimensions, split sizes and metrics in `results/run_metadata.json`. Re-running with the same environment and seed should reproduce the synthetic data and split.

## Ethical and clinical limitations
This is not a diagnostic tool, does not provide patient-specific medical advice, and has no prospective or external clinical validation. Synthetic subgroup metrics do not establish real-world fairness. A clinical system would require suitable representative data, clinical outcome definition, calibration and decision-curve assessment, external and prospective validation, governance approval, security review, human oversight and monitoring for distribution shift.
