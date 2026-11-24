# Reproducibility guidance

Record dataset provenance and permissions, cohort logic, target and prediction horizon, patient-level split identifiers (kept securely), preprocessing, software versions, random seeds and all model configurations.

For DP-SGD, record:
- protected unit and adjacency definition;
- per-example clipping norm;
- noise multiplier and sampling rate;
- number of optimisation steps and batch size;
- target delta and accountant implementation/version;
- computed epsilon and assumptions.

Do not infer formal differential privacy from adding arbitrary noise to a statistic. Federated learning alone is not a privacy guarantee. Document secure aggregation status and residual model-update leakage risks.

Use `Experiment_Log_Template.csv` to record runs. Share only artefacts and code permitted by the dataset agreement and ethics approval.
