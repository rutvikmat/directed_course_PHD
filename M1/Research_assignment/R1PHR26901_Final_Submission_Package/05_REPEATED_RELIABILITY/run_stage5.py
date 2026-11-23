
import os, json, warnings
import numpy as np, pandas as pd
from scipy.special import logit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,brier_score_loss,confusion_matrix,accuracy_score
warnings.filterwarnings("ignore")
OUT=os.environ.get("OUT_DIR","results_batch");os.makedirs(OUT,exist_ok=True)
N_SEEDS=int(os.environ.get("N_SEEDS","10"))
SEED_START=int(os.environ.get("SEED_START","100")); SEEDS=list(range(SEED_START,SEED_START+N_SEEDS))
SCENARIOS=["No shift","Covariate mean shift (+0.5 SD)","Measurement noise (0.5 SD)","MCAR missingness (20%)","Outcome prevalence shift (balanced sample)"]
MODELS=["Logistic Regression","Random Forest"]
def build_model(name,seed):
    if name=="Logistic Regression":
        return Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("clf",LogisticRegression(max_iter=5000,C=1.0))])
    if name=="Random Forest":
        return Pipeline([("imp",SimpleImputer(strategy="median")),("clf",RandomForestClassifier(n_estimators=100,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=-1))])
    if name=="SVM RBF":
        return Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("clf",SVC(C=1,kernel="rbf",probability=True,class_weight="balanced",random_state=seed))])
    return Pipeline([("imp",SimpleImputer(strategy="median")),("clf",ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,class_weight="balanced",random_state=seed,n_jobs=-1))])
def balance(X,y,rng):
    i0=y[y==0].index.to_numpy();i1=y[y==1].index.to_numpy();n=min(len(i0),len(i1))
    ix=np.concatenate([rng.choice(i0,n,replace=False),rng.choice(i1,n,replace=False)])
    rng.shuffle(ix)
    return X.loc[ix].copy(),y.loc[ix].copy()
def transform_pair(scen,Xa,ya,Xt,yt,std,rng):
    if scen=="No shift": return Xa.copy(),ya.copy(),Xt.copy(),yt.copy()
    if scen=="Covariate mean shift (+0.5 SD)":
        return Xa+.5*std,ya.copy(),Xt+.5*std,yt.copy()
    if scen=="Measurement noise (0.5 SD)":
        na=rng.normal(0,.5,size=Xa.shape)*std.values
        nt=rng.normal(0,.5,size=Xt.shape)*std.values
        return Xa+na,ya.copy(),Xt+nt,yt.copy()
    if scen=="MCAR missingness (20%)":
        ma=rng.random(Xa.shape)<.20;mt=rng.random(Xt.shape)<.20
        return Xa.mask(ma),ya.copy(),Xt.mask(mt),yt.copy()
    aa=balance(Xa,ya,rng); tt=balance(Xt,yt,rng); return aa[0],aa[1],tt[0],tt[1]
def shift_fit(scen,Xf,yf,std,rng):
    if scen=="No shift":return Xf.copy(),yf.copy()
    if scen=="Covariate mean shift (+0.5 SD)":return Xf+.5*std,yf.copy()
    if scen=="Measurement noise (0.5 SD)":return Xf+rng.normal(0,.5,size=Xf.shape)*std.values,yf.copy()
    if scen=="MCAR missingness (20%)":return Xf.mask(rng.random(Xf.shape)<.20),yf.copy()
    return balance(Xf,yf,rng)
def cal_stats(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); z=logit(p).reshape(-1,1)
    # unpenalized-like calibration logistic model with weak regularization for numerical stability
    c=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000)
    c.fit(z,np.asarray(y))
    return float(c.intercept_[0]),float(c.coef_[0,0])
def calc(y,p):
    y=np.asarray(y);p=np.clip(np.asarray(p),1e-6,1-1e-6);pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    ci,cs=cal_stats(y,p)
    return {"brier":brier_score_loss(y,p),"roc_auc":roc_auc_score(y,p),
            "calibration_intercept":ci,"calibration_slope":cs,
            "malignant_recall":tn/(tn+fn) if tn+fn else np.nan,
            "accuracy":accuracy_score(y,pred),"malignant_false_negatives":int(fn)}
d=load_breast_cancer();X=pd.DataFrame(d.data,columns=d.feature_names);y=pd.Series(d.target,index=X.index)
records=[]
for seed in SEEDS:
    rng=np.random.default_rng(seed)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=seed)
    Xfit,Xadapt,yfit,yadapt=train_test_split(Xtr,ytr,test_size=.30,stratify=ytr,random_state=seed+1000)
    std=Xfit.std().replace(0,1)
    for si,scen in enumerate(SCENARIOS):
        # use same transformed adapt/test samples across models within seed/scenario
        Xa,ya,Xt,yt=transform_pair(scen,Xadapt,yadapt,Xte,yte,std,np.random.default_rng(seed*100+si))
        Xf,yf=shift_fit(scen,Xfit,yfit,std,np.random.default_rng(seed*100+si+50))
        Xfull=pd.concat([Xf,Xa]);yfull=pd.concat([yf,ya])
        for model_name in MODELS:
            model=build_model(model_name,seed);model.fit(Xfit,yfit)
            p0=model.predict_proba(Xt)[:,1]
            vals=calc(yt,p0)
            records.append({"seed":seed,"scenario":scen,"model":model_name,"strategy":"No update",**vals})
            pa=model.predict_proba(Xa)[:,1]
            cal=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000)
            cal.fit(logit(np.clip(pa,1e-6,1-1e-6)).reshape(-1,1),ya)
            pr=cal.predict_proba(logit(np.clip(p0,1e-6,1-1e-6)).reshape(-1,1))[:,1]
            records.append({"seed":seed,"scenario":scen,"model":model_name,"strategy":"Probability recalibration",**calc(yt,pr)})
            retr=build_model(model_name,seed);retr.fit(Xfull,yfull)
            pt=retr.predict_proba(Xt)[:,1]
            records.append({"seed":seed,"scenario":scen,"model":model_name,"strategy":"Full model retraining",**calc(yt,pt)})
raw=pd.DataFrame(records);raw.to_csv(f"{OUT}/seed_level_results.csv",index=False)
metrics=["brier","roc_auc","calibration_intercept","calibration_slope","malignant_recall","accuracy"]
summary=raw.groupby(["scenario","model","strategy"])[metrics].agg(["mean","std"]).reset_index()
summary.to_csv(f"{OUT}/summary_mean_sd.csv",index=False)
# Paired bootstrap over seed IDs; compare strategy minus no-update for each scenario/model.
rng=np.random.default_rng(90210);B=2000
ci_rows=[]
for (scen,model),g in raw.groupby(["scenario","model"]):
    piv=g.pivot(index="seed",columns="strategy",values=metrics)
    for strat in ["Probability recalibration","Full model retraining"]:
        for met in metrics:
            diff=(piv[(met,strat)]-piv[(met,"No update")]).dropna().values
            boot=np.array([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(B)])
            ci_rows.append({"scenario":scen,"model":model,"strategy_vs":"No update","strategy":strat,"metric":met,
             "mean_paired_difference":float(np.mean(diff)),"ci95_lower":float(np.quantile(boot,.025)),
             "ci95_upper":float(np.quantile(boot,.975)),"n_seeds":len(diff),
             "interpretation":"Negative is favourable" if met=="brier" else ("Positive is favourable" if met in ["roc_auc","malignant_recall","accuracy"] else "Closer to 0 (intercept) or 1 (slope) is favourable")})
pd.DataFrame(ci_rows).to_csv(f"{OUT}/paired_bootstrap_differences.csv",index=False)
# scenario/model/strategy average across model families
overall=raw.groupby(["scenario","strategy"])[metrics].mean().reset_index()
overall.to_csv(f"{OUT}/overall_strategy_means.csv",index=False)
manifest={"dataset":"Wisconsin Diagnostic Breast Cancer (569 observations, 30 features)",
"random_seeds":SEEDS,"n_seeds":N_SEEDS,"split":"Stratified 80:20 train/test; then stratified 70:30 fit/adaptation split within training",
"strategies":{"No update":"Fit model on fit subset; unchanged.","Probability recalibration":"Fit logistic calibration mapping on labelled shifted adaptation subset; base model unchanged.","Full model retraining":"Refit same model family on fit subset transformed to shift plus shifted labelled adaptation subset."},
"shifts":SCENARIOS,"target":{"0":"malignant","1":"benign"},
"bootstrap":"2000 paired bootstrap resamples of the 20 seed-level paired differences; percentile 95% intervals.",
"calibration":"Logistic calibration regression of outcome on logit(predicted P(benign)); ideal intercept 0, slope 1.",
"caveats":["Repeated random splits overlap in observations and are not independent external replications.","Bootstrap intervals quantify variability across these random seeds/splits, not population-level clinical uncertainty.","Synthetic shifts are not real temporal/site/population shifts.","Adaptation uses labelled data.","The balanced prevalence condition uses balanced resampling, not a prospective prior-shift design."]}
json.dump(manifest,open(f"{OUT}/experiment_manifest.json","w"),indent=2)
print("completed rows",len(raw))
print("\\nOVERALL\\n",overall.round(3).to_string(index=False))
