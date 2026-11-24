import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def permutation_plot(model, X_test, y_test, output_dir, seed=42):
    result = permutation_importance(model, X_test, y_test, n_repeats=8,
                                    random_state=seed, scoring="roc_auc", n_jobs=-1)
    names = np.asarray(X_test.columns)
    order = np.argsort(result.importances_mean)
    plt.figure(figsize=(9, 5))
    plt.barh(names[order], result.importances_mean[order], xerr=result.importances_std[order])
    plt.xlabel("Mean decrease in ROC-AUC after permutation")
    plt.title("Permutation feature importance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/permutation_importance.png", dpi=160)
    plt.close()
    pd.DataFrame({"feature": names, "importance_mean": result.importances_mean,
                  "importance_std": result.importances_std}).sort_values(
                      "importance_mean", ascending=False).to_csv(
                          f"{output_dir}/permutation_importance.csv", index=False)

def local_perturbation_explanation(model, X_test, output_dir, row_index=0):
    """Model-agnostic local sensitivity illustration, not a causal explanation."""
    x = X_test.iloc[[row_index]].copy()
    baseline = float(model.predict_proba(x)[0, 1])
    rows=[]
    for col in X_test.columns:
        changed=x.copy()
        if pd.api.types.is_numeric_dtype(X_test[col]):
            changed[col] = X_test[col].median()
        else:
            mode = X_test[col].mode(dropna=True)
            changed[col] = mode.iloc[0] if len(mode) else "Unknown"
        new=float(model.predict_proba(changed)[0, 1])
        rows.append({"feature": col, "baseline_probability": baseline,
                     "perturbed_probability": new, "change": new-baseline})
    out=pd.DataFrame(rows).sort_values("change", key=lambda s: s.abs(), ascending=False)
    out.to_csv(f"{output_dir}/local_perturbation_explanation.csv", index=False)
    return out

def optional_shap(model, X_test, output_dir):
    try:
        import shap
        # Pipeline-agnostic KernelExplainer can be slow; cap sample sizes.
        sample = X_test.sample(min(80, len(X_test)), random_state=42)
        background = X_test.sample(min(30, len(X_test)), random_state=7)
        explainer = shap.KernelExplainer(lambda z: model.predict_proba(
            pd.DataFrame(z, columns=X_test.columns))[:, 1], background)
        values = explainer.shap_values(sample, nsamples=80)
        shap.summary_plot(values, sample, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary_optional.png", dpi=160, bbox_inches="tight")
        plt.close()
        return "SHAP summary generated."
    except Exception as exc:
        return f"Optional SHAP skipped: {type(exc).__name__}: {exc}"
