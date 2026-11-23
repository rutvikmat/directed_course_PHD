
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")
SEED=42
out="results"; os.makedirs(out,exist_ok=True)
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
# sklearn target: 0 malignant, 1 benign
y=pd.Series(data.target,name="diagnosis")
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=SEED)
models={
 "Logistic Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=5000,C=1.0))]),
 "Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced",random_state=SEED,n_jobs=-1))]),
 "SVM RBF":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",SVC(C=1.0,kernel="rbf",probability=True,class_weight="balanced",random_state=SEED))]),
 "Extra Trees":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced",random_state=SEED,n_jobs=-1))])
}
for m in models.values(): m.fit(Xtr,ytr)
# Controlled test-set shifts. All transforms are applied to held-out test only.
rng=np.random.default_rng(SEED)
scales=Xtr.std(axis=0).replace(0,1)
scenarios={}
scenarios["No shift"] = Xte.copy()
# Covariate mean shift: shift all measurements by +0.5 training SD
scenarios["Covariate mean shift (+0.5 SD)"] = Xte + 0.5*scales
# Feature noise: additive independent Gaussian noise with SD 0.5 training SD
scenarios["Measurement noise (0.5 SD)"] = Xte + pd.DataFrame(rng.normal(0,.5,(len(Xte),Xte.shape[1]))*scales.values,index=Xte.index,columns=Xte.columns)
# Missingness: MCAR 20% entries; median imputation occurs in pipelines
Xm=Xte.copy()
mask=rng.random(Xm.shape)<.20
Xm= Xm.mask(mask)
scenarios["MCAR missingness (20%)"] = Xm
# Prior/prevalence shift: balanced resampling of held-out malignant/benign cases; conditional features unchanged
idx0=yte[yte==0].index.to_numpy(); idx1=yte[yte==1].index.to_numpy()
n=min(len(idx0),len(idx1))
idx_prior=np.concatenate([rng.choice(idx0,n,replace=False),rng.choice(idx1,n,replace=False)])
rng.shuffle(idx_prior)
scenarios["Outcome prevalence shift (balanced sample)"] = Xte.loc[idx_prior]
yscenarios={name:(yte.loc[Xs.index] if name!="Outcome prevalence shift (balanced sample)" else yte.loc[Xs.index]) for name,Xs in scenarios.items()}
# metrics
rows=[]
for scen,Xs in scenarios.items():
    ys=yscenarios[scen]
    for name,model in models.items():
        p=model.predict_proba(Xs)[:,1] # P(benign)
        pred=(p>=.5).astype(int)
        tn,fp,fn,tp=confusion_matrix(ys,pred,labels=[0,1]).ravel()
        rows.append({"scenario":scen,"model":name,"n":len(ys),"accuracy":accuracy_score(ys,pred),
          "balanced_accuracy":balanced_accuracy_score(ys,pred),"f1_benign":f1_score(ys,pred,pos_label=1),
          "roc_auc_benign":roc_auc_score(ys,p),"brier_benign":brier_score_loss(ys,p),
          "malignant_recall":tn/(tn+fn) if tn+fn else np.nan,"false_negatives_malignant_as_benign":fn,
          "false_positives_benign_as_malignant":fp})
metrics=pd.DataFrame(rows)
metrics.to_csv(f"{out}/shift_metrics.csv",index=False)
# Drift detection: proportion of features with KS p<.05 compared with training
drift=[]
for scen,Xs in scenarios.items():
    pvals=[ks_2samp(Xtr[c].dropna(),Xs[c].dropna()).pvalue for c in X.columns]
    drift.append({"scenario":scen,"features_KS_p_lt_0_05":int(np.sum(np.array(pvals)<.05)),
                  "features_tested":len(pvals),"fraction_features_flagged":float(np.mean(np.array(pvals)<.05))})
pd.DataFrame(drift).to_csv(f"{out}/univariate_drift_detection.csv",index=False)
# Confidence / selective prediction for LR and RF; abstain lowest-confidence fraction
select=[]
for scen,Xs in scenarios.items():
    ys=yscenarios[scen]
    for name in ["Logistic Regression","Random Forest"]:
        p=models[name].predict_proba(Xs)[:,1]
        confidence=np.maximum(p,1-p)
        pred=(p>=.5).astype(int)
        for coverage in [.5,.7,.9,1.0]:
            cutoff=np.quantile(confidence,1-coverage)
            keep=confidence>=cutoff
            if keep.sum()==0: continue
            select.append({"scenario":scen,"model":name,"target_coverage":coverage,
                "actual_coverage":keep.mean(),"selective_accuracy":accuracy_score(ys.iloc[np.where(keep)[0]],pred[keep]),
                "retained_n":int(keep.sum())})
pd.DataFrame(select).to_csv(f"{out}/selective_prediction.csv",index=False)
# Reproducibility metadata
meta={"dataset":"Wisconsin Diagnostic Breast Cancer (scikit-learn load_breast_cancer; UCI WDBC)",
      "n":int(len(X)),"features":int(X.shape[1]),"train_n":int(len(Xtr)),"test_n":int(len(Xte)),
      "split":"stratified 80:20","random_state":42,"target_mapping":{"0":"malignant","1":"benign"},
      "models":list(models.keys()),
      "scenarios":list(scenarios.keys()),
      "warning":"All shifts are controlled synthetic transformations of the held-out test set; they are not observed temporal, hospital, scanner or population shifts. Outcome-prevalence shift uses balanced resampling of the held-out cases and changes test prevalence by design."}
with open(f"{out}/experiment_manifest.json","w") as f: json.dump(meta,f,indent=2)
print(metrics.round(4).to_string(index=False))
print("\\nDRIFT\\n",pd.DataFrame(drift).round(3).to_string(index=False))
