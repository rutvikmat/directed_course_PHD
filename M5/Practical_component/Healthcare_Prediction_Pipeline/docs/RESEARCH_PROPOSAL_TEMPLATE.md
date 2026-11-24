# Short Doctoral Research Proposal Template

## Title
Privacy-Aware and Fair Predictive Modelling for Healthcare Risk Assessment

## 1. Background and problem
Healthcare prediction models may improve risk stratification, but centralised use of sensitive longitudinal data creates privacy risks. Predictive performance may also vary across demographic subgroups. A model with good aggregate performance can still have unequal error rates or poor calibration for underrepresented groups.

## 2. Research gap
Many prototype studies report predictive metrics without jointly evaluating subgroup performance, explainability and the privacy-utility trade-off. Federated learning can reduce the need to pool raw records, but it does not independently prevent leakage from model updates.

## 3. Aim and objectives
Aim: Develop and evaluate an explainable, subgroup-aware healthcare prediction framework with a formally specified privacy-preserving extension.

Objectives:
1. Define a clinically meaningful prediction target and prediction horizon.
2. Establish reproducible baseline models and evaluation protocols.
3. Evaluate feature contributions and explanation stability.
4. quantify subgroup discrimination, calibration and error-rate differences.
5. Implement federated training and/or DP-SGD with a documented privacy accountant.
6. Compare privacy-utility-fairness trade-offs and document limitations.

## 4. Proposed algorithmic contribution
A candidate contribution is a privacy-aware federated training workflow with contribution clipping, secure aggregation and optional differentially private optimisation, combined with subgroup-aware monitoring. The novelty must be established through a systematic literature review; this proposal does not claim that the combination is unprecedented.

## 5. Experimental design
Use a legally accessible, ethically approved dataset with a defined cohort, outcome, prediction time and horizon. Compare logistic regression, tree-based models and a suitable temporal baseline if longitudinal data are available. Use patient-level splitting, temporal validation and external validation where possible. Report ROC-AUC, PR-AUC, calibration, sensitivity, specificity, subgroup metrics, confidence intervals, privacy budget, communication cost and ablation results.

## 6. Ethical considerations
Obtain institutional ethics review and data access approvals before accessing patient data. Apply data minimisation, role-based access, encryption, retention limits and a documented consent/legal basis. Do not re-identify records. Assess representativeness and possible harms. Keep the system as decision support with clinician oversight, not autonomous diagnosis.

## 7. Reproducibility plan
Version-control code; pin dependencies; document dataset provenance and licence; publish preprocessing and split logic; fix random seeds; log configurations; preserve environment metadata; report all experiments, including negative results; and provide a data statement. Share only data that may legally and ethically be redistributed.

## 8. Expected contribution and limitations
Expected outputs include a reproducible benchmark, explainability and subgroup audit, and a quantified privacy-utility evaluation. Limitations may include dataset shift, small subgroup sizes, missingness bias, residual privacy risks and limited generalisability.
