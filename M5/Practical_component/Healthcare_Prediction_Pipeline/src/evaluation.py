import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, confusion_matrix,
                             classification_report)

def evaluate_model(name, model, X_test, y_test, output_dir):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred.astype(float)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall_sensitivity": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "brier_score": brier_score_loss(y_test, prob),
    }
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(
        f"{output_dir}/{name.lower().replace(' ', '_')}_classification_report.csv")
    return metrics, pred, prob, confusion_matrix(y_test, pred)

def subgroup_metrics(y_true, y_pred, groups, attribute):
    frame = pd.DataFrame({"y": np.asarray(y_true), "pred": np.asarray(y_pred),
                          "group": pd.Series(groups).astype("string").fillna("Unknown").to_numpy()})
    rows=[]
    for group, d in frame.groupby("group", dropna=False):
        tp=((d.y==1)&(d.pred==1)).sum()
        fp=((d.y==0)&(d.pred==1)).sum()
        fn=((d.y==1)&(d.pred==0)).sum()
        tn=((d.y==0)&(d.pred==0)).sum()
        rows.append({
            "attribute": attribute, "group": str(group), "n": len(d),
            "outcome_prevalence": d.y.mean(), "prediction_selection_rate": d.pred.mean(),
            "accuracy": (tp+tn)/len(d) if len(d) else np.nan,
            "tpr_sensitivity": tp/(tp+fn) if (tp+fn) else np.nan,
            "fpr": fp/(fp+tn) if (fp+tn) else np.nan,
            "precision_ppv": tp/(tp+fp) if (tp+fp) else np.nan
        })
    return rows
