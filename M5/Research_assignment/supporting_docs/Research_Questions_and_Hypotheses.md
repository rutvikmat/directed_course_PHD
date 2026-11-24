# Research questions and hypotheses

## Main research question
How do privacy-preserving learning mechanisms affect predictive utility, explanation behaviour and subgroup fairness in healthcare risk prediction?

## Supporting questions
1. How do baseline algorithms compare on discrimination and calibration?
2. How stable are global and local explanations across resamples and model variants?
3. How do error rates and calibration differ across adequately represented subgroups?
4. What predictive utility is retained at specified differential privacy budgets?
5. How does federated training compare with centralised training in utility, communication cost and leakage risk?

## Candidate hypotheses (to refine after dataset selection)
- H1: Baseline model performance differs across algorithms and metrics.
- H2: Subgroup performance differences remain after aggregate performance is considered.
- H3: Increasing privacy protection may reduce predictive utility, with the magnitude dependent on the dataset and privacy configuration.
- H4: Federated training can approach centralised baseline performance under some data-partition conditions, but does not alone establish privacy.

These are hypotheses to test, not established findings.
