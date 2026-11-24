# Experimental Protocol

## Research question
Does an entity-aware, missingness-aware and drift-adaptive forecasting framework
improve robustness over non-adaptive forecasting baselines when time series differ
between entities, contain missing observations, and undergo temporal changes?

## Hypotheses
- H1: Entity-specific information can improve performance when entities have distinct baselines or temporal profiles.
- H2: Explicit missingness indicators and causal imputation can improve robustness over imputation without missingness information.
- H3: Drift-triggered recent-window updating can reduce post-change forecasting error compared with a static model.
- H4: The combined framework may provide more stable performance under simultaneous missingness and drift.

These are hypotheses to test, not established conclusions.

## Data
Use PEMS-BAY traffic speeds, where each sensor is treated as an entity and each
timestamp is a sequential observation. Preserve chronological order.

## Split
- First 70%: training.
- Next 10%: validation and drift-threshold calibration.
- Final 20%: testing.
No random shuffling is permitted.

## Conditions
1. Original observed data.
2. Artificial missingness at 5%, 10%, 20%, and 30%.
3. Controlled temporal level shift from 80% of the series onward.
4. Combined 20% missingness and temporal level shift.

## Models
- Seasonal naive benchmark (previous-day value).
- Random Forest global baseline.
- IV-MPAF: global XGBoost with sensor identity, entity-history statistics,
  lagged values, missingness flags and residual-triggered recent-window refitting.

## Metrics
MAE and RMSE are primary point-forecast metrics. Report metrics by scenario and,
where possible, by sensor. Report update frequency and runtime if collected.

## Fairness and leakage controls
- Split by time, never randomly.
- Features use past observations only.
- Fit preprocessing/imputation parameters using past data only.
- Do not use future target observations to construct predictors.
- Compare models on matching timestamps and entities.
- Preserve the uncorrupted target as evaluation ground truth in synthetic missingness tests.

## Reproducibility
Record software versions, configuration, dataset version, sensor subset, date range,
random seed, and any modifications to the code.
