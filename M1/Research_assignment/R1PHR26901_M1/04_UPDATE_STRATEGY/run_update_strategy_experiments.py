
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
from scipy.special import logit
warnings.filterwarnings("ignore")
SEED=42
rng=np.random.default_rng(SEED)
os.makedirs("results",exist_ok=True)
d=load_breast_cancer()
X=pd.DataFrame(d.data,columns=d.feature_names); y=pd.Series(d.target,name="diagnosis")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=SEED)
# Hold out an adaptation/calibration subset from the training partition.
X_fit,X_adapt,y_fit,y_adapt=train_test_split(X_train,y_train,test_size=.30,stratify=y_train,random_state=SEED)
std=X_fit.std().replace(0,1)
feature_cols=list(X.columns)
def shift_pair(name, Xa, ya, Xt, yt, seed):
    rr=np.random.default_rng(seed)
    if name=="No shift":
        return Xa.copy(),ya.copy(),Xt.copy(),yt.copy()
    if name=="Covariate mean shift (+0.5 SD)":
        delta=.5*std
        return Xa+delta,ya.copy(),Xt+delta,yt.copy()
    if name=="Measurement noise (0.5 SD)":
        na=rr.normal(0,.5,size=Xa.shape)*std.values
        nt=rr.normal(0,.5,size=Xt.shape)*std.values
        return Xa+na,ya.copy(),Xt+nt,yt.copy()
    if name=="MCAR missingness (20%)":
        ma=rr.random(Xa.shape)<.20; mt=rr.random(Xt.shape)<.20
        return Xa.mask(ma),ya.copy(),Xt.mask(mt),yt.copy()
    if name=="Outcome prevalence shift (balanced sample)":
        def balance(Xs,ys):
            i0=ys[ys==0].index.to_numpy(); i1=ys[ys==1].index.to_numpy()
            n=min(len(i0),len(i1))
            ix=np.concatenate([rr.choice(i0,n,replace=False),rr.choice(i1,n,replace=False)])
            rr.shuffle(ix)
            return Xs.loc[ix].copy(),ys.loc[ix].copy()
        Xa2,ya2=balance(Xa,ya); Xt2,yt2=balance(Xt,yt)
        return Xa2,ya2,Xt2,yt2
    raise ValueError(name)
def shift_fit(name, Xs, ys, seed):
    rr=np.random.default_rng(seed)
    if name=="No shift":
        return Xs.copy(),ys.copy()
    if name=="Covariate mean shift (+0.5 SD)":
        return Xs+.5*std,ys.copy()
    if name=="Measurement noise (0.5 SD)":
        noise=rr.normal(0,.5,size=Xs.shape)*std.values
        return Xs+noise,ys.copy()
    if name=="MCAR missingness (20%)":
        mask=rr.random(Xs.shape)<.20
        return Xs.mask(mask),ys.copy()
    if name=="Outcome prevalence shift (balanced sample)":
        i0=ys[ys==0].index.to_numpy(); i1=ys[ys==1].index.to_numpy()
        n=min(len(i0),len(i1))
        ix=np.concatenate([rr.choice(i0,n,replace=False),rr.choice(i1,n,replace=False)])
        rr.shuffle(ix)
        return Xs.loc[ix].copy(),ys.loc[ix].copy()
    raise ValueError(name)
def make_models(seed):
    return {
      "Logistic Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=5000,C=1.0))]),
      "Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=-1))]),
      "SVM RBF":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",SVC(C=1,kernel="rbf",probability=True,class_weight="balanced",random_state=seed))]),
      "Extra Trees":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=-1))])
    }
def get_p(model,Xs):
    return np.clip(model.predict_proba(Xs)[:,1],1e-6,1-1e-6)
def cal_intercept_slope(ytrue,p):
    # Logistic calibration model: logit(P(Y=1)) = intercept + slope*logit(predicted probability)
    z=logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1)
    cal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=3000)
    cal.fit(z,np.asarray(ytrue))
    return float(cal.intercept_[0]),float(cal.coef_[0,0])
def metrics(ytrue,p):
    ytrue=np.asarray(ytrue); pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(ytrue,pred,labels=[0,1]).ravel()
    ci,cs=cal_intercept_slope(ytrue,p)
    return {"accuracy":accuracy_score(ytrue,pred),"roc_auc":roc_auc_score(ytrue,p),
      "brier_score":brier_score_loss(ytrue,p),"calibration_intercept":ci,
      "calibration_slope":cs,"malignant_recall":tn/(tn+fn) if (tn+fn)>0 else np.nan,
      "malignant_false_negatives":int(fn),"n_test":len(ytrue)}
scenarios=["No shift","Covariate mean shift (+0.5 SD)","Measurement noise (0.5 SD)","MCAR missingness (20%)","Outcome prevalence shift (balanced sample)"]
rows=[]
for scen_i,scen in enumerate(scenarios):
    Xa,ya,Xt,yt=shift_pair(scen,X_adapt,y_adapt,X_test,y_test,SEED+100+scen_i)
    # Use one deterministic, scenario-specific transformed adaptation/test pair across model families.
    for model_name,base_model in make_models(SEED).items():
        base_model.fit(X_fit,y_fit)
        # Strategy A: no update
        p0=get_p(base_model,Xt)
        for strategy,predp in [("No update",p0)]:
            rows.append({"scenario":scen,"model":model_name,"strategy":strategy,**metrics(yt,predp)})
        # Strategy B: probability recalibration on held-out shifted adaptation labels
        pa=get_p(base_model,Xa)
        z_a=logit(pa).reshape(-1,1)
        recal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=3000)
        recal.fit(z_a,ya)
        p_recal=recal.predict_proba(logit(p0).reshape(-1,1))[:,1]
        rows.append({"scenario":scen,"model":model_name,"strategy":"Probability recalibration",**metrics(yt,p_recal)})
        # Strategy C: full retraining on shifted training data (original fit + shifted adaptation)
        Xfit_shift,yfit_shift=shift_fit(scen,X_fit,y_fit,SEED+500+scen_i)
        X_full=pd.concat([Xfit_shift,Xa],axis=0); y_full=pd.concat([yfit_shift,ya],axis=0)
        retrained=make_models(SEED)[model_name]
        retrained.fit(X_full,y_full)
        p_retrain=get_p(retrained,Xt)
        rows.append({"scenario":scen,"model":model_name,"strategy":"Full model retraining",**metrics(yt,p_retrain)})
results=pd.DataFrame(rows)
results.to_csv("results/update_strategy_metrics.csv",index=False)
# Paired delta versus no-update within same scenario/model
wide=results.pivot(index=["scenario","model"],columns="strategy",values=["brier_score","roc_auc","calibration_intercept","calibration_slope","malignant_recall"])
delta=[]
for idx,row in wide.iterrows():
    for strat in ["Probability recalibration","Full model retraining"]:
        rec={"scenario":idx[0],"model":idx[1],"strategy":strat}
        for metric in ["brier_score","roc_auc","calibration_intercept","calibration_slope","malignant_recall"]:
            rec[metric+"_change_vs_no_update"]=row[(metric,strat)]-row[(metric,"No update")]
        delta.append(rec)
pd.DataFrame(delta).to_csv("results/changes_vs_no_update.csv",index=False)
# Save methodology and seed metadata
manifest={"dataset":"Wisconsin Diagnostic Breast Cancer (scikit-learn; 569 x 30)",
 "split":"Stratified 80:20 outer train/test; training partition split 70:30 into model-fit and adaptation/calibration subsets",
 "random_state":42,"target":{"0":"malignant","1":"benign"},
 "strategies":{"No update":"Original model trained on fit subset; no adaptation.",
 "Probability recalibration":"Original model fixed; logistic recalibration fitted on separate labelled adaptation subset transformed with same shift; evaluated on separate shifted test set.",
 "Full model retraining":"Same model family refitted on the full training partition transformed/resampled to the scenario distribution; evaluated on a separate shifted test set."},
 "calibration_metrics":"Calibration intercept and slope from logistic regression of outcome on logit(predicted P(benign)); ideal intercept=0 and slope=1.",
 "scenarios":scenarios,
 "caveats":["Synthetic shifts do not represent real clinical/hospital/time shifts.","The prevalence scenario uses balanced resampling of available labelled samples, not a prospective prevalence-shift cohort.","Adaptation labels are assumed available; this is a supervised updating experiment.","Small test sample means calibration slope/intercept may be unstable.","No external validation or confidence intervals are included."]}
with open("results/experiment_manifest.json","w") as f: json.dump(manifest,f,indent=2)
print(results.round(4).to_string(index=False))
