# Results overview

The pipeline completed successfully in the build environment. Metrics are from one fixed stratified 80:20 split; positive class is benign.

| model               |   accuracy |   precision_benign |   recall_benign |   f1_benign |   roc_auc |   brier_score |   log_loss | tn_fp_fn_tp   |
|:--------------------|-----------:|-------------------:|----------------:|------------:|----------:|--------------:|-----------:|:--------------|
| Logistic Regression |   0.95614  |           0.985507 |        0.944444 |    0.964539 |  0.99537  |     0.0259908 |  0.0889804 | 41,1,4,68     |
| Random Forest       |   0.947368 |           0.958333 |        0.958333 |    0.958333 |  0.993056 |     0.0339624 |  0.116989  | 39,3,3,69     |

SHAP status: {'available': True, 'n_explained': 114}

These results are educational and do not establish clinical validity.
