import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

def permutation_importance_table(model, X_test, y_test, feature_names, seed=42):
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=20,
        random_state=seed, n_jobs=-1
    )
    return pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

def shap_analysis(rf_pipeline, X_train, X_test, feature_names, out_dir):
    try:
        import shap
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}
    try:
        # Explain the fitted forest on its imputed feature matrix.
        imputer = rf_pipeline.named_steps["imputer"]
        forest = rf_pipeline.named_steps["model"]
        X_bg = pd.DataFrame(imputer.transform(X_train), columns=feature_names)
        X_eval = pd.DataFrame(imputer.transform(X_test), columns=feature_names)
        explainer = shap.TreeExplainer(forest)
        values = explainer.shap_values(X_eval)
        # Binary-class SHAP output differs across SHAP versions.
        if isinstance(values, list):
            vals = values[1]
        elif getattr(values, "ndim", 0) == 3:
            vals = values[:, :, 1]
        else:
            vals = values
        imp = np.abs(vals).mean(axis=0)
        table = pd.DataFrame({"feature": feature_names,
                              "mean_abs_shap": imp}).sort_values(
                                  "mean_abs_shap", ascending=False)
        table.to_csv(f"{out_dir}/shap_global_importance.csv", index=False)
        plt.figure(figsize=(9, 7))
        top = table.head(15).sort_values("mean_abs_shap")
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Mean absolute SHAP value")
        plt.title("Random Forest: global SHAP feature importance")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/shap_global_importance.png", dpi=180)
        plt.close()
        # Waterfall plot for one test case.
        try:
            exp = explainer(X_eval.iloc[[0]])
            if hasattr(exp, "values"):
                ev = exp[0]
                if getattr(ev.values, "ndim", 1) == 2:
                    ev = shap.Explanation(values=ev.values[:, 1],
                                          base_values=ev.base_values[1],
                                          data=ev.data,
                                          feature_names=feature_names)
                shap.plots.waterfall(ev, max_display=12, show=False)
                plt.tight_layout()
                plt.savefig(f"{out_dir}/shap_patient_waterfall.png",
                            dpi=180, bbox_inches="tight")
                plt.close()
        except Exception:
            pass
        return {"available": True, "n_explained": int(len(X_eval))}
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}

def explanation_stability(rf_pipeline, X_test, feature_names, out_dir, seed=42):
    # Perturb one randomly selected test observation slightly, then compare
    # permutation importances on small local samples. This is a practical proxy,
    # not a proof of explanation validity.
    rng = np.random.default_rng(seed)
    imputer = rf_pipeline.named_steps["imputer"]
    forest = rf_pipeline.named_steps["model"]
    X = pd.DataFrame(imputer.transform(X_test), columns=feature_names)
    base = X.iloc[[0]].copy()
    rows = []
    base_pred = float(rf_pipeline.predict_proba(X_test.iloc[[0]])[0, 1])
    base_importance = forest.feature_importances_
    for level in [0.0, 0.01, 0.03, 0.05, 0.10]:
        perturbed = base.copy()
        scale = X.std(axis=0).replace(0, 1)
        noise = rng.normal(0, level, size=perturbed.shape) * scale.to_numpy()
        perturbed.iloc[0, :] = perturbed.iloc[0, :].to_numpy() + noise[0]
        pred = float(forest.predict_proba(perturbed)[0, 1])
        # Global feature importance is constant for a fixed fitted model;
        # rank agreement is therefore 1.0 by construction. Instead, record
        # local prediction sensitivity and top features as a baseline diagnostic.
        top = np.argsort(base_importance)[-5:][::-1]
        rows.append({
            "relative_noise_sd": level,
            "base_probability_benign": base_pred,
            "perturbed_probability_benign": pred,
            "absolute_probability_change": abs(pred-base_pred),
            "top5_global_features": ";".join(feature_names[i] for i in top)
        })
    result = pd.DataFrame(rows)
    result.to_csv(f"{out_dir}/explanation_stability.csv", index=False)
    return result
