import numpy as np
import pandas as pd
import joblib
from directed_course.Module_2.M2.Dynamical_Structure_Research.src.pipeline import create_models, metrics, save_csv, RESULTS

def run(frame, tr, va, te, base_cols, hybrid_cols, fast=False):
    trainval = np.concatenate([tr,va])
    y = frame['target']
    models = create_models(fast=fast)
    fitted = {}
    preds = pd.DataFrame({'timestamp':frame.index[te], 'actual':y.iloc[te].values})
    preds['SeasonalNaive'] = frame['seasonal_naive'].iloc[te].values
    preds['StateSpace'] = frame['ssm_forecast_next'].iloc[te].values
    for name, model in models.items():
        print('Training', name)
        model.fit(frame.iloc[trainval][base_cols], y.iloc[trainval])
        preds[name] = model.predict(frame.iloc[te][base_cols])
        fitted[name] = model
    print('Training Hybrid XGBoost + state features')
    hybrid = create_models(fast=fast)['XGBoost']
    hybrid.fit(frame.iloc[trainval][hybrid_cols], y.iloc[trainval])
    preds['Hybrid'] = hybrid.predict(frame.iloc[te][hybrid_cols])
    fitted['Hybrid'] = hybrid
    rows = [{'Model':c,**metrics(preds.actual, preds[c])} for c in preds.columns if c not in ['timestamp','actual']]
    results = pd.DataFrame(rows).sort_values('RMSE')
    save_csv(results, 'exp1_accuracy.csv')
    save_csv(preds, 'exp1_predictions.csv')
    RESULTS.mkdir(exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, RESULTS / f'{name}.joblib')
    print(results.to_string(index=False))
    return results, preds, fitted
