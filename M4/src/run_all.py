import os, sys, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from data_utils import load_data, split_data, make_logistic_pipeline, make_rf_pipeline
from modeling import evaluate_model, classification_text, subgroup_evaluation
from explainability import permutation_importance_table, shap_analysis, explanation_stability
from config import RANDOM_STATE, TEST_SIZE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

def main():
    X, y, features = load_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, TEST_SIZE, RANDOM_STATE
    )
    models = {
        "Logistic Regression": make_logistic_pipeline(),
        "Random Forest": make_rf_pipeline()
    }
    metrics, reports, subgroup_rows = [], [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics.append(evaluate_model(name, model, X_test, y_test))
        reports.append(classification_text(name, model, X_test, y_test))
        subgroup_rows.extend(subgroup_evaluation(name, model, X_test, y_test))
        # Calibration curve for both models.
    pd.DataFrame(metrics).to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)
    with open(os.path.join(OUT, "classification_reports.txt"), "w") as f:
        f.write("\n\n".join(reports))
    pd.DataFrame(subgroup_rows).to_csv(os.path.join(OUT, "subgroup_metrics.csv"), index=False)

    cal_rows = []
    plt.figure(figsize=(7, 6))
    plt.plot([0,1], [0,1], linestyle="--", label="Perfect calibration")
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:,1]
        frac, mean = calibration_curve(y_test, prob, n_bins=10, strategy="uniform")
        plt.plot(mean, frac, marker="o", label=name)
        cal_rows.append({"model":name,
                         "brier_score": next(m["brier_score"] for m in metrics if m["model"]==name),
                         "log_loss": next(m["log_loss"] for m in metrics if m["model"]==name)})
    plt.xlabel("Mean predicted probability (benign)")
    plt.ylabel("Observed fraction benign")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "calibration_curve.png"), dpi=180)
    plt.close()
    pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "calibration_metrics.csv"), index=False)

    perm = permutation_importance_table(models["Random Forest"], X_test, y_test, features)
    perm.to_csv(os.path.join(OUT, "permutation_importance.csv"), index=False)
    # Logistic coefficients: standardized coefficients provide a simple
    # interpretable reference; absolute magnitude is not causal importance.
    lr = models["Logistic Regression"].named_steps["model"]
    coef = pd.DataFrame({"feature":features, "coefficient":lr.coef_[0],
                         "absolute_coefficient":abs(lr.coef_[0])}).sort_values(
                             "absolute_coefficient", ascending=False)
    coef.to_csv(os.path.join(OUT, "logistic_coefficients.csv"), index=False)

    shap_status = shap_analysis(models["Random Forest"], X_train, X_test, features, OUT)
    stability = explanation_stability(models["Random Forest"], X_test, features, OUT)

    summary = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn bundled)",
        "n_samples": int(len(X)), "n_features": int(X.shape[1]),
        "train_n": int(len(X_train)), "test_n": int(len(X_test)),
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "models": metrics, "shap": shap_status,
        "stability_rows": int(len(stability)),
        "target_encoding": {"0":"malignant","1":"benign"},
        "caution": "Educational benchmark only; not for clinical use."
    }
    with open(os.path.join(OUT, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Pipeline completed. Results saved to:", OUT)
    print(pd.DataFrame(metrics).to_string(index=False))
    print("SHAP:", shap_status)

if __name__ == "__main__":
    main()
