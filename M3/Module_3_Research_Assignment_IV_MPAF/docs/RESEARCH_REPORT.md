# Research Assignment Report

## Title
Development of an Adaptive Time-Series Forecasting Framework Incorporating
Individual Variation, Missing Data and Changing Temporal Patterns

**Proposed framework:** IV-MPAF – Individual Variation and Missingness-aware
Pattern-Adaptive Forecasting Framework

## Abstract
Real-world time-series data commonly exhibit differences between entities,
incomplete observations and temporal behaviour that changes over time. These
characteristics can reduce the reliability of conventional forecasting models.
This study proposes IV-MPAF, a forecasting framework that combines shared
learning across entities with entity-specific information, missingness-aware
features and drift-triggered model updating. PEMS-BAY traffic-speed data is used
as a multi-entity case study, with individual sensors treated as separate
forecasting entities. The framework is evaluated against seasonal-naive and
Random Forest baselines under original, missingness, temporal-drift and combined
conditions. The study uses chronological evaluation to avoid temporal leakage.
The effectiveness of the framework is determined from the experimental results.

## 1. Introduction
Time-series forecasting is used in transportation, energy, healthcare, industrial
monitoring and environmental analysis. Many real-world applications involve
multiple entities whose temporal patterns are not identical. In addition,
observations may be missing because of sensor failures, transmission problems or
irregular measurement. Temporal patterns can also change due to external events,
seasonal transitions or changes in system behaviour.

A forecasting method that assumes complete and stable data may perform poorly
when these conditions occur. This research therefore investigates an adaptive
forecasting framework that accounts for individual variation, missing data and
changing temporal patterns.

## 2. Problem Statement
Conventional forecasting approaches may fail to represent entity-specific
behaviour, may be sensitive to incomplete observations, and may lose accuracy
when temporal relationships change. A unified framework is required to combine
shared temporal learning, entity-specific information, explicit missingness
handling and adaptation to temporal changes.

## 3. Aim
To develop and evaluate an adaptive time-series forecasting framework that
accounts for individual variation, missing observations and changing temporal
patterns.

## 4. Objectives
1. Analyse individual variation across multiple time-series entities.
2. Investigate missing-data handling strategies for temporal observations.
3. Detect temporal changes using residual monitoring and change-sensitive statistics.
4. Develop an entity-aware and missingness-aware forecasting model with adaptive updating.
5. Compare the proposed approach with established forecasting baselines.
6. Evaluate accuracy and robustness under separate and combined challenges.

## 5. Dataset
The PEMS-BAY traffic-speed dataset is selected as a multi-sensor time-series
benchmark. Each traffic sensor is treated as an individual entity. Traffic speed
is modelled as a function of historical speed observations and temporal features.
The data is sampled at five-minute intervals in its commonly distributed form.
The exact file version, date range, number of sensors and preprocessing decisions
must be recorded after loading the dataset.

## 6. Proposed Framework: IV-MPAF
IV-MPAF has four functional components:

### 6.1 Entity-aware representation
A global model is trained across multiple sensors. Sensor identity and
sensor-specific historical statistics allow the model to represent shared
relationships while retaining entity-level variation.

### 6.2 Missingness-aware processing
Missing observations are identified before imputation. Causal forward filling
and historical fallback values are used to construct predictors, while missingness
flags preserve information about unavailable observations. Future values are not
used to fill past observations.

### 6.3 Temporal pattern monitoring
Forecast residuals are monitored over time. A sustained increase in recent error
relative to validation error is treated as a possible indication of temporal
change.

### 6.4 Adaptive model updating
When the drift threshold is crossed, the model is refitted using a recent
historical window. The update uses observations available up to that point and
affects subsequent forecasts.

## 7. Methodology
The experiment follows a chronological train-validation-test split. The training
period is used for initial model fitting, the validation period is used to
estimate baseline error, and the test period is used for final evaluation.
Controlled missingness and a synthetic level shift are introduced to assess
robustness. The original target values are retained for scoring in the artificial
missingness experiments.

## 8. Experimental Conditions
- Original data.
- 5%, 10%, 20% and 30% artificial missingness.
- Synthetic level shift in a subset of sensors during the test period.
- Combined 20% missingness and synthetic level shift.

The synthetic shift is a controlled stress test, not a claim that the original
dataset contains that exact event.

## 9. Baseline Models
- Seasonal naive: previous-day observation.
- Random Forest: global tree-based regression baseline.
- IV-MPAF: proposed entity-aware, missingness-aware and adaptive model.

## 10. Evaluation Metrics
MAE and RMSE are used as primary point-forecast metrics. Results should be
reported for each scenario and, where practical, separately for individual
sensors. Lower values indicate lower forecast error.

## 11. Results
**Complete this section only after running the experiment.**

Insert:
- `outputs/metrics.csv`
- `outputs/missingness_robustness.png`
- `outputs/drift_robustness.png`
- `outputs/scenario_model_comparison.png`

Discuss:
1. Which model has the lowest MAE and RMSE on the original data?
2. How does error change as missingness increases?
3. Does adaptive updating reduce error after the synthetic shift?
4. Which sensors show the greatest variation?
5. What trade-offs appear in runtime and update frequency?

Do not assume IV-MPAF performs best. Report measured results, including cases where
a baseline performs better.

## 12. Discussion
Interpret the results in relation to the hypotheses. Discuss whether entity
features contribute useful information, whether missingness indicators help,
and whether residual-triggered updating responds appropriately to temporal
changes. Identify possible confounding effects and limitations.

## 13. Limitations
- A limited sensor subset may not represent the complete network.
- Artificial missingness may not reproduce real sensor-failure mechanisms.
- A synthetic level shift is a simplified representation of concept drift.
- The prototype uses a global gradient-boosting model rather than a fully
  probabilistic deep architecture.
- Thresholds and update windows require sensitivity analysis.
- Performance on PEMS-BAY does not establish generalisation to other domains.

## 14. Conclusion
This assignment presents a reproducible prototype for evaluating adaptive
multi-entity time-series forecasting under individual variation, missing
observations and changing temporal patterns. The proposed framework combines
shared learning, entity-specific features, missingness indicators and
residual-triggered updates. Its effectiveness must be judged from the actual
experimental results. Future work can investigate additional datasets, more
advanced change-point detection, probabilistic forecasting and transfer across
domains.

## References
1. Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion Convolutional
   Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
   Neural Computation, 9(8), 1735–1780.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
   Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining.
4. Lim, B., Arik, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion
   Transformers for Interpretable Multi-horizon Time Series Forecasting.
   International Journal of Forecasting, 37(4), 1748–1764.
5. Little, R. J. A., & Rubin, D. B. (2019). Statistical Analysis with Missing Data.
   Wiley.
6. Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.
