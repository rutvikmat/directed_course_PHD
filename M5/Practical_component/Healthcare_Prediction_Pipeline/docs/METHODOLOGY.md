# Methodology

## Problem formulation
Given a feature vector representing a synthetic patient's health-related indicators, predict a binary risk outcome. The output is a research demonstration and must not be interpreted as a diagnosis.

## Data and preprocessing
The default data are generated with `sklearn.datasets.make_classification`, augmented with synthetic age and sex fields and controlled missingness. Demographic attributes are excluded from the primary predictor set and retained for subgroup auditing. Numeric features use median imputation and standardisation; categorical features use most-frequent imputation and one-hot encoding. Preprocessing is fitted inside the training pipeline to reduce leakage.

## Experimental design
A stratified 75/25 train/test split is used with a fixed seed. Three classifiers are compared. Report discrimination, threshold-based performance and calibration proxy (Brier score). The highest test ROC-AUC is selected only to produce the example explanation and subgroup report. For a publishable study, choose models on validation data or nested cross-validation and evaluate once on a locked external test cohort.

## Explainability
Permutation importance estimates the decrease in ROC-AUC when a feature is shuffled. The local perturbation file changes one feature to a typical value and records the probability change. These are associations/sensitivity summaries, not causal effects. Optional SHAP output is attempted when the package is available.

## Subgroup analysis
The selected model is evaluated across age bands and sex categories. Metrics include sample size, prevalence, selection rate, accuracy, sensitivity, false-positive rate and positive predictive value. Differences must be interpreted alongside uncertainty, subgroup sample sizes, outcome prevalence and clinical context. No single fairness metric is universally sufficient.

## Limitations
Synthetic data do not represent clinical populations. No temporal, geographic or external validation is performed. There is no clinician review, prospective study, clinical utility assessment or regulatory evaluation.
