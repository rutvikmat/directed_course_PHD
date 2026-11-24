# Research Assignment: Assessing Stability, Reliability and Clinical Usefulness of XAI Explanations

**Module:** Explainable AI, Trustworthy Prediction and Model Reliability

## 1. Proposed title
A Perturbation- and Subgroup-Based Framework for Evaluating the Stability and Reliability of SHAP Explanations in Healthcare Prediction Models

## 2. Problem statement
Healthcare prediction models may produce accurate outputs while providing explanations that are unstable, misleading or difficult for intended users to interpret. A feature attribution can change substantially after a small input perturbation, a change in the reference/background sample, or a shift in the patient population. There is therefore a need to evaluate explanations systematically rather than assuming that a visual explanation is automatically trustworthy.

## 3. Aim
To propose and evaluate a reproducible framework for assessing the stability, model fidelity, robustness and potential clinical usefulness of feature-attribution explanations.

## 4. Objectives
1. Generate local SHAP explanations for a trained healthcare prediction model.
2. Measure explanation stability under clinically plausible input perturbations.
3. Measure sensitivity to background/reference sample selection.
4. Compare explanations across model seeds and model families.
5. Evaluate explanation changes under simulated distribution shift.
6. Define a structured assessment of clinical plausibility and usefulness for future expert review.

## 5. Proposed framework
The framework has five stages.

### Stage A: Baseline model and explanation
Train a predictive model using a fixed training/test protocol. Generate local SHAP explanations for a pre-specified sample of test instances. Record the model probability, predicted class, base value, SHAP values and feature ranking.

### Stage B: Input perturbation stability
For each selected instance, create repeated perturbed versions of the input. Perturbations must remain within the observed feature range and should be clinically plausible. For each replicate, recompute SHAP values.

For instance i and perturbation replicate r, let the attribution vector be:
`phi(i,r) = [phi_1, phi_2, ..., phi_p]`.

Compare each perturbed attribution vector with the original attribution vector using:
- Spearman rank correlation of feature rankings.
- Top-k feature overlap (e.g., k=5).
- Normalised L1 attribution change.

A possible normalised change is:
`NLA = sum_j |phi_j(r) - phi_j(0)| / (sum_j |phi_j(0)| + epsilon)`.

### Stage C: Reference/background sensitivity
Repeat SHAP computation using multiple background samples drawn from the training data. Compare rankings and attribution magnitudes. This tests whether the explanation depends strongly on the selected reference population.

### Stage D: Model and distribution robustness
Train the same model with multiple random seeds and compare explanations for the same observations. Separately simulate a covariate shift in a controlled manner, document the transformation, and assess:
- Predictive performance change.
- Calibration change.
- Attribution ranking change.
- Subgroup-specific explanation changes.

Synthetic shifts are stress tests, not substitutes for external validation using an independent hospital or temporal cohort.

### Stage E: Clinical usefulness assessment
A future clinician-facing evaluation should use a pre-defined protocol and appropriately qualified reviewers. Reviewers can rate each explanation on:
- Clinical plausibility.
- Clarity and comprehensibility.
- Relevance to the clinical question.
- Actionability (where appropriate).
- Whether uncertainty and limitations are communicated.
- Whether the explanation risks encouraging an incorrect causal interpretation.

Use a structured Likert scale and report inter-rater agreement. Do not claim clinical usefulness from feature-attribution scores alone.

## 6. Evaluation measures

| Dimension | Measure | Interpretation |
|---|---|---|
| Rank stability | Spearman correlation | Agreement in feature ordering |
| Top-feature stability | Top-k overlap/Jaccard | Consistency of most influential features |
| Attribution magnitude | Normalised L1 change | Size of attribution changes |
| Model fidelity | Local surrogate fidelity or deletion/insertion test | Whether explanation reflects model behaviour |
| Predictive reliability | Brier score, log loss, calibration curve | Quality of predicted probabilities |
| Robustness | Performance and attribution change under shift | Sensitivity to changed data |
| Subgroup consistency | Metrics and explanation stability by subgroup | Whether behaviour varies across groups |
| Human usefulness | Reviewer ratings and inter-rater agreement | Whether intended users find explanations useful |

## 7. Hypotheses
- H1: Explanations generated for clinically plausible small perturbations will show higher rank stability than explanations generated under larger perturbations.
- H2: The choice of SHAP background sample will affect attribution magnitude and may affect feature ranking.
- H3: Explanation instability may increase under simulated distribution shift.
- H4: High predictive performance does not necessarily imply stable or clinically useful explanations.

## 8. Experimental design
1. Use a healthcare dataset with a clearly defined target and documented provenance.
2. Split data at patient level where repeated observations exist; avoid leakage.
3. Train at least one interpretable baseline and one non-linear model.
4. Select a fixed evaluation cohort before explanation analysis.
5. Generate baseline SHAP values.
6. Apply pre-specified perturbation levels and repeat the explanation procedure.
7. Repeat using multiple background samples and random seeds.
8. Calculate stability and fidelity measures.
9. Report confidence intervals using bootstrap resampling where appropriate.
10. Conduct expert review only after ethics, governance and data-access approvals where required.

## 9. Expected outcomes
- A reproducible explanation-stability evaluation protocol.
- Quantitative evidence about perturbation and background-sample sensitivity.
- A comparison of model performance and explanation stability.
- Identification of cases where explanation confidence should be qualified.
- Recommendations for reporting XAI explanations in healthcare studies.

## 10. Ethical and methodological considerations
- Do not use the model for patient care.
- Protect privacy and follow data governance requirements.
- Avoid presenting predictive associations as causes.
- Document dataset limitations and subgroup sample sizes.
- Do not infer fairness from a single proxy subgroup.
- Clinical review requires qualified participants and appropriate approvals.
- External validation is necessary before any clinical deployment claim.

## 11. Conclusion
The proposed framework treats explanation quality as an empirical property to be tested. It combines perturbation stability, reference sensitivity, model fidelity, robustness, calibration and prospective human assessment. This approach can help distinguish visually persuasive explanations from explanations that are stable, faithful and potentially useful in a clinical workflow.
