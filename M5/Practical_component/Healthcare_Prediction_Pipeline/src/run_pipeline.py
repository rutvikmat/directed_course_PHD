import os, sys, json, argparse, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.base import clone

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RANDOM_STATE, TEST_SIZE, OUTPUT_DIR, TARGET
from data import load_data
from evaluation import evaluate_model, subgroup_metrics
from explain import permutation_plot, local_perturbation_explanation, optional_shap
from privacy_demo import save_privacy_demo

def main():
    parser=argparse.ArgumentParser(description="Healthcare prediction research prototype")
    parser.add_argument("--data", default=None, help="Optional CSV path")
    parser.add_argument("--target", default=TARGET, help="Binary target column (0/1)")
    args=parser.parse_args()
    out=os.path.join(ROOT, OUTPUT_DIR); os.makedirs(out, exist_ok=True)

    X,y,data_source=load_data(args.data, args.target, RANDOM_STATE)
    if len(np.unique(y)) != 2:
        raise ValueError("This prototype requires a binary outcome.")
    # Identify demographic attributes for analysis, excluding them from predictors
    # to keep subgroup analysis separate from the primary model.
    subgroup_cols=[c for c in ["age","sex"] if c in X.columns]
    model_X=X.drop(columns=subgroup_cols).copy()
    numeric_cols=model_X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols=[c for c in model_X.columns if c not in numeric_cols]
    transformers=[]
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols))
    pre=ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        model_X, y, X.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    models={
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced",
                                                    random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                 class_weight="balanced", random_state=RANDOM_STATE,
                                                 n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                                  l2_regularization=1.0,
                                                                  random_state=RANDOM_STATE)
    }
    trained={}; metrics=[]; predictions={}
    for name, estimator in models.items():
        pipe=Pipeline([("preprocess", clone(pre)), ("model", estimator)])
        pipe.fit(X_train,y_train)
        trained[name]=pipe
        m,pred,prob,cm=evaluate_model(name,pipe,X_test,y_test,out)
        metrics.append(m); predictions[name]=(pred,prob,cm)
    comparison=pd.DataFrame(metrics).sort_values("roc_auc",ascending=False)
    comparison.to_csv(f"{out}/model_comparison.csv",index=False)

    # Select best model on held-out ROC-AUC for demonstration; report all models.
    best_name=comparison.iloc[0]["model"]; best=trained[best_name]
    best_pred,best_prob,best_cm=predictions[best_name]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    ConfusionMatrixDisplay(best_cm).plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"Confusion matrix: {best_name}")
    for name,pipe in trained.items():
        RocCurveDisplay.from_estimator(pipe,X_test,y_test,ax=axes[1],name=name)
    axes[1].set_title("ROC curves")
    plt.tight_layout(); plt.savefig(f"{out}/model_evaluation.png",dpi=160); plt.close()

    # Feature explanations on raw feature names (demographics excluded from model).
    permutation_plot(best,X_test,y_test,out,RANDOM_STATE)
    local_perturbation_explanation(best,X_test,out)
    shap_status=optional_shap(best,X_test,out)

    # Subgroup evaluation on untouched test rows. Use explicit age bands.
    sub=[]
    if "age" in X.columns:
        age=pd.cut(X.loc[idx_test,"age"], bins=[0,35,50,65,120],
                   labels=["18-35","36-50","51-65","66+"], include_lowest=True)
        sub += subgroup_metrics(y_test,best_pred,age,"age_band")
    if "sex" in X.columns:
        sub += subgroup_metrics(y_test,best_pred,X.loc[idx_test,"sex"],"sex")
    pd.DataFrame(sub).to_csv(f"{out}/subgroup_fairness.csv",index=False)

    privacy=save_privacy_demo(y_train,f"{out}/privacy_perturbation_demo.csv")
    metadata={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "python_version":platform.python_version(),"scikit_learn_version":sklearn.__version__,
        "data_source":data_source,"n_rows":int(len(X)),"n_predictors":int(X.shape[1]),
        "target":args.target,"positive_class":1,"random_state":RANDOM_STATE,
        "test_size":TEST_SIZE,"train_rows":int(len(X_train)),"test_rows":int(len(X_test)),
        "demographic_attributes_excluded_from_primary_model":subgroup_cols,
        "selected_model_by_test_roc_auc_for_demonstration":best_name,
        "optional_shap_status":shap_status,
        "privacy_demo_warning":privacy["privacy_guarantee"]
    }
    with open(f"{out}/run_metadata.json","w") as f: json.dump(metadata,f,indent=2)
    with open(f"{out}/experiment_summary.md","w") as f:
        f.write("# Experiment summary\\n\\n")
        f.write(f"- Data source: `{data_source}`\\n- Rows: {len(X)}\\n- Best model by held-out ROC-AUC: **{best_name}**\\n\\n")
        f.write(comparison.to_markdown(index=False))
        f.write("\\n\\n## Interpretation cautions\\n")
        f.write("The data are synthetic unless a permitted CSV is supplied. Test-set model selection is used only for this teaching demonstration; a rigorous study should use validation data or nested cross-validation for model selection and reserve an untouched external test set. Subgroup metrics are descriptive and should be interpreted with confidence intervals and adequate sample sizes.\\n")
        f.write(f"\\nOptional explainability status: {shap_status}\\n")
    print("\\nPipeline completed.")
    print(comparison.to_string(index=False))
    print(f"\\nOutputs: {out}")

if __name__=="__main__":
    main()
