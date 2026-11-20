# Traffic Volume Forecasting — Experimental Report

## Objective
Compare ARIMA, XGBoost, and LSTM for 1-, 6-, and 24-hour-ahead traffic-volume forecasting.

## Dataset and preprocessing
- Raw rows: 48204
- Hourly grid rows: 52551
- Missing values after short-gap interpolation: 8797
- Time span: 2012-10-02 09:00:00 to 2018-09-30 23:00:00
- Duplicate timestamps are aggregated; the data are reindexed hourly; only short internal gaps (up to three hours) are interpolated.

## Experimental protocol
- Chronological split: 70% training, 15% validation, 15% test.
- Forecast horizons: 1, 6, and 24 hours.
- ARIMA order selected by AIC from a small candidate set: (3, 0, 1).
- XGBoost uses lagged/rolling traffic and known calendar features; LSTM receives a genuine 168-hour sequence (traffic plus cyclical calendar channels) and predicts the next 24 hours.
- Rolling-origin test evaluation used 24 candidate origins (origins with incomplete 24-hour actual windows are skipped).
- Seasonal-naive-24 baseline predicts using the corresponding hour on the previous day.
- Metrics: MAE, RMSE, WAPE, and sMAPE.

## Results by model and horizon
| model           |   horizon_hours | origin   |      MAE |     RMSE |   WAPE_pct |   sMAPE_pct |
|:----------------|----------------:|:---------|---------:|---------:|-----------:|------------:|
| XGBoost         |               1 |          |  109.103 |  126.07  |    2.97752 |     4.49049 |
| LSTM            |               1 |          |  282.632 |  322.445 |    7.71328 |    12.6826  |
| ARIMA           |               1 |          |  691.703 |  993.042 |   18.8772  |    25.5865  |
| SeasonalNaive24 |               1 |          |  566.778 | 1030.52  |   15.4679  |    16.7463  |
| LSTM            |               6 |          |  299.108 |  394.446 |    9.63517 |    13.7145  |
| XGBoost         |               6 |          |  222.33  |  465.353 |    7.16194 |     8.4281  |
| SeasonalNaive24 |               6 |          |  658.778 | 1154.49  |   21.2212  |    23.1119  |
| ARIMA           |               6 |          | 1588.99  | 1829.01  |   51.1861  |    58.0179  |
| XGBoost         |              24 |          |  193.179 |  352.194 |    5.90121 |     7.45635 |
| LSTM            |              24 |          |  370.586 |  427.56  |   11.3206  |    15.6497  |
| SeasonalNaive24 |              24 |          |  617.333 | 1171.61  |   18.8582  |    20.9603  |
| ARIMA           |              24 |          | 1564.11  | 1735.81  |   47.7801  |    54.5129  |

## Best model by horizon (lowest RMSE)
|   horizon_hours | model   |     MAE |    RMSE |   WAPE_pct |   sMAPE_pct |
|----------------:|:--------|--------:|--------:|-----------:|------------:|
|               1 | XGBoost | 109.103 | 126.07  |    2.97752 |     4.49049 |
|               6 | LSTM    | 299.108 | 394.446 |    9.63517 |    13.7145  |
|              24 | XGBoost | 193.179 | 352.194 |    5.90121 |     7.45635 |

## Average metrics across horizons
| model           |      MAE |     RMSE |   WAPE_pct |   sMAPE_pct |
|:----------------|---------:|---------:|-----------:|------------:|
| ARIMA           | 1281.6   | 1519.29  |   39.2811  |    46.0391  |
| LSTM            |  317.442 |  381.484 |    9.55635 |    14.0156  |
| SeasonalNaive24 |  614.296 | 1118.87  |   18.5158  |    20.2728  |
| XGBoost         |  174.871 |  314.539 |    5.34689 |     6.79165 |

## Interpretation guidance
Do not claim that one model is universally superior. Discuss which model performs best at each horizon, whether error increases with horizon, and how the models compare with the seasonal-naive baseline.
A lower RMSE indicates fewer large errors; MAE is easier to interpret in vehicles/hour. WAPE is more stable than MAPE when actual values can be zero.

## Limitations and threats to validity
- XGBoost and LSTM use lagged traffic plus known calendar features; ARIMA is univariate.
- Weather and holiday variables are excluded from the shared primary information set.
- Traffic regimes, road incidents, construction, and policy changes are not explicitly modelled.
- Results depend on the chosen split, rolling-origin sample, ARIMA candidate set, and compute budget.
- The `--fast` option is for pipeline verification only; rerun the full configuration for final academic reporting.

## Conclusion
Insert a conclusion based on the measured results above. State the best-performing model separately for each forecast horizon and discuss practical trade-offs.
