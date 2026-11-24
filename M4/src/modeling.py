import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, brier_score_loss,
                             log_loss, classification_report, confusion_matrix)
from sklearn.calibration import calibration_curve

def evaluate_model(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_benign": precision_score(y_test, pred, zero_division=0),
        "recall_benign": recall_score(y_test, pred, zero_division=0),
        "f1_benign": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "tn_fp_fn_tp": ",".join(map(str, confusion_matrix(y_test, pred).ravel()))
    }

def classification_text(name, model, X_test, y_test):
    return f"===== {name} =====\n" + classification_report(
        y_test, model.predict(X_test), target_names=["malignant", "benign"],
        zero_division=0
    )

def subgroup_evaluation(name, model, X_test, y_test):
    # Educational subgroup proxy based on mean radius quartiles; not a protected attribute.
    # It illustrates stratified error analysis, not a fairness audit.
    radius = X_test["mean radius"]
    groups = pd.qcut(radius, q=4, labels=["Q1_low_radius","Q2","Q3","Q4_high_radius"],
                     duplicates="drop")
    pred = model.predict(X_test)
    rows = []
    for group in groups.unique():
        mask = (groups == group).to_numpy()
        if mask.sum() == 0:
            continue
        rows.append({
            "model": name, "subgroup": str(group), "n": int(mask.sum()),
            "accuracy": float((pred[mask] == y_test.to_numpy()[mask]).mean()),
            "recall_malignant": float(
                ((pred[mask] == 0) & (y_test.to_numpy()[mask] == 0)).sum() /
                max(1, (y_test.to_numpy()[mask] == 0).sum())
            )
        })
    return rows
