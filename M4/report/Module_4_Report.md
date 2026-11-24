# Module 4 Report: Explainable AI, Trustworthy Prediction and Model Reliability


## Abstract
This work studies explainability and reliability in healthcare-oriented machine learning. A binary classification task is implemented using the Wisconsin Diagnostic Breast Cancer dataset distributed with scikit-learn. Logistic Regression is used as a comparatively interpretable baseline and Random Forest as a non-linear ensemble model. Performance is assessed using accuracy, precision, recall, F1-score, ROC-AUC, Brier score and log loss. Explainability is examined through model coefficients, permutation feature importance and SHAP (when available). Reliability checks include probability calibration, subgroup error analysis using feature-defined strata, and prediction sensitivity to small input perturbations. The work emphasises that explanations are descriptions of model behaviour rather than causal evidence, and that good benchmark performance does not establish clinical validity.

## 1. Introduction
Machine learning can identify complex patterns in healthcare data, but predictive performance alone is insufficient for high-impact decisions. Clinicians and patients may need to understand the factors associated with a prediction, the uncertainty of the prediction, and the conditions under which the model may fail. Explainable AI (XAI) provides methods for inspecting model behaviour. Trustworthy prediction additionally requires calibration, robustness, fairness assessment, and transparent communication of limitations.

## 2. Aim and objectives
**Aim:** To compare an interpretable baseline and a tree-ensemble model, and assess the quality and reliability of their explanations in a healthcare classification benchmark.

**Objectives:**
1. Prepare a reproducible healthcare classification dataset.
2. Train Logistic Regression and Random Forest models.
3. Compare discrimination and probability-quality metrics.
4. Generate global and local explanations using SHAP where supported.
5. Compare SHAP/permutation importance with Logistic Regression coefficients.
6. Examine calibration, feature-defined subgroup performance, and sensitivity to small perturbations.
7. Discuss the limits of explanation methods and clinical translation.

## 3. Dataset and preprocessing
The Wisconsin Diagnostic Breast Cancer dataset contains 569 observations and 30 numeric image-derived features. The binary target is encoded as 0 (malignant) and 1 (benign). The data are split into training and test subsets using an 80:20 stratified split and a fixed random seed of 42. Median imputation is included in each pipeline. Logistic Regression additionally uses standardisation. Preprocessing is fitted within the training pipeline to avoid leakage.

## 4. Models
### 4.1 Logistic Regression
Logistic Regression is used as the simpler reference model. Its coefficients indicate the direction and relative strength of feature associations on the standardised scale. Coefficients are conditional on the other variables and should not be interpreted causally.

### 4.2 Random Forest
Random Forest combines predictions from multiple decision trees. It can represent non-linear relationships and feature interactions, but its internal structure is less directly interpretable than a single linear model.

## 5. Evaluation methodology
The held-out test set is evaluated using:
- Accuracy, precision, recall and F1-score.
- ROC-AUC for ranking/discrimination.
- Brier score and log loss for probability quality.
- Calibration curve comparing predicted probabilities with observed frequencies.
- Feature-defined subgroup analysis to identify performance variation across mean-radius quartiles.

Subgroup analysis here is an error-analysis demonstration, not a complete fairness audit: mean radius is not a protected attribute, and the dataset does not provide sufficient demographic or clinical context for a meaningful equity assessment.

## 6. Explainability methodology
### 6.1 Permutation feature importance
Feature values are shuffled one column at a time. The reduction in ROC-AUC estimates the model's reliance on that feature for this dataset and evaluation sample. Correlated features can share or mask importance.

### 6.2 SHAP
SHAP assigns additive feature contributions relative to a reference value. Global importance is summarised using mean absolute SHAP values. A waterfall plot is generated for one test observation when supported by the installed SHAP version. SHAP values explain the fitted model output; they do not prove that a feature causes the outcome.

### 6.3 Logistic Regression coefficients
Standardised coefficients provide a transparent baseline. Their magnitudes are not directly interchangeable with SHAP values because the methods explain different model forms and output scales.

## 7. Trustworthiness and reliability checks
### 7.1 Calibration
Brier score, log loss and a calibration curve are reported. A well-calibrated model should assign probabilities that correspond to observed frequencies over groups of predictions. Calibration may change across populations and time.

### 7.2 Robustness and sensitivity
The experiment perturbs one test observation with increasing small Gaussian noise and records the change in predicted probability. This is a simple local sensitivity diagnostic. It is not a complete distribution-shift experiment and does not establish robustness under real hospital-to-hospital changes.

### 7.3 Subgroup performance
Performance is stratified by quartiles of mean radius. The purpose is to demonstrate how aggregate metrics can conceal variation. A real fairness assessment should use clinically and ethically relevant subgroups, adequate sample sizes, and appropriate protected or clinically relevant attributes.

### 7.4 Explanation stability
The supplied perturbation experiment reports prediction sensitivity and the model's top global features. Because global Random Forest feature importance is fixed for a fitted model, it cannot be treated as a valid local explanation-stability score. The research assignment therefore proposes a stronger repeated local-SHAP stability protocol using rank correlation, top-k overlap, and attribution magnitude change.

## 8. Results
Run `python src/run_all.py`. The generated numerical results are saved in `results/`. Insert the values from `model_metrics.csv` into the table below after execution.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9561 | 0.9855 | 0.9444 | 0.9645 | 0.9954 | 0.0260 |
| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9931 | 0.0340 |

### Observed results from the included run

The fixed 80:20 stratified split produced the following metrics (positive class: benign):

| Logistic Regression | 0.9561 | 0.9855 | 0.9444 | 0.9645 | 0.9954 | 0.0260 |
| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9931 | 0.0340 |

On this split, Logistic Regression achieved slightly higher accuracy and ROC-AUC and a lower Brier score than Random Forest. Random Forest had slightly higher benign-class recall. These are single-split results and should not be interpreted as proof of general superiority; repeated cross-validation and external validation would be needed for stronger conclusions.

Discuss:
1. Which model has stronger ROC-AUC?
2. Which model has lower Brier score and log loss?
3. Do the most important features agree across SHAP, permutation importance and Logistic Regression coefficients?
4. Are there differences in subgroup recall?
5. Does a small perturbation materially alter an individual prediction?

Do not claim that one model is superior based on a single metric. Consider discrimination, calibration, errors, interpretability and the intended use.

## 9. Discussion
The experiment illustrates the trade-off between a relatively transparent model and a more flexible ensemble. Post-hoc explanations can help inspect the ensemble, but they add another layer of approximation and assumptions. Correlated features may divide attribution, and explanations may be unstable under perturbations or changes in the background distribution. Calibration and subgroup analysis are essential because a high ROC-AUC does not guarantee reliable individual probabilities or equitable performance.

## 10. Limitations
1. The dataset is small, historical and narrowly defined.
2. It is not a prospective or multi-hospital validation cohort.
3. It lacks the demographic and clinical variables needed for a robust fairness assessment.
4. The perturbation test is a limited sensitivity analysis.
5. SHAP implementation details and output conventions vary by software version.
6. Model explanations are not causal explanations.
7. No clinician or patient usability study was conducted.

## 11. Conclusion
The practical demonstrates a reproducible workflow for comparing an interpretable Logistic Regression baseline with Random Forest and inspecting predictions using feature attribution. Trustworthy healthcare prediction requires more than accuracy: calibration, robustness, subgroup performance, explanation stability and clinical usefulness must also be assessed. The proposed research assignment extends the practical by defining a repeated perturbation-based protocol for measuring explanation stability and reliability. The results are educational and must not be interpreted as clinical evidence.

## References
1. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?” Explaining the Predictions of Any Classifier. *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
4. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232.
5. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *Proceedings of ICML*.
6. Doshi-Velez, F., & Kim, B. (2017). Towards A Rigorous Science of Interpretable Machine Learning. arXiv:1702.08608.
7. scikit-learn developers. *scikit-learn documentation: Model evaluation, calibration, permutation importance and datasets*.
8. SHAP developers. *SHAP documentation*.
