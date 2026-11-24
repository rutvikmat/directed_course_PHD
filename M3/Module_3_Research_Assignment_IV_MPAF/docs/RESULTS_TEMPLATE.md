# Results Table Template

Populate using `outputs/metrics.csv` after execution.

| Scenario | Model | MAE | RMSE |
|---|---|---:|---:|
| Original | SeasonalNaive | | |
| Original | RandomForest | | |
| Original | IV-MPAF | | |
| Missing 5% | ... | | |
| Missing 10% | ... | | |
| Missing 20% | ... | | |
| Missing 30% | ... | | |
| Drift | ... | | |
| Combined | ... | | |

## Interpretation prompts
- Does error increase with missingness?
- Which model is most robust as missingness increases?
- Does adaptive refitting help after drift?
- Does performance differ across sensors?
- What are the runtime and update-frequency costs?
- Are differences large and consistent enough to justify further statistical testing?

Do not fill this table with illustrative or invented values.
