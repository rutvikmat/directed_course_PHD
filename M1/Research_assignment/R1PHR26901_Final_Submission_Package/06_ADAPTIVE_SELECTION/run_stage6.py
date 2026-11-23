import os, json, zipfile, warnings
import numpy as np, pandas as pd
from scipy.special import logit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, accuracy_score
warnings.filterwarnings('ignore')
OUT=os.environ.get('OUT_DIR',os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT,exist_ok=True)
SEEDS=list(range(200,210))
SCENARIOS=['No shift','Covariate mean shift (+0.5 SD)','Measurement noise (0.5 SD)','MCAR missingness (20%)','Outcome prevalence shift (balanced sample)']
MODELS=['Logistic Regression','Random Forest']
STRATEGIES=['No update','Probability recalibration','Full model retraining']

def build_model(name,seed):
    if name=='Logistic Regression':
        return Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('clf',LogisticRegression(max_iter=5000,C=1.0))])
    return Pipeline([('imp',SimpleImputer(strategy='median')),('clf',RandomForestClassifier(n_estimators=100,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=-1))])
def balance(X,y,rng):
    i0=np.where(np.asarray(y)==0)[0]; i1=np.where(np.asarray(y)==1)[0]; n=min(len(i0),len(i1))
    ix=np.r_[rng.choice(i0,n,replace=False),rng.choice(i1,n,replace=False)]; rng.shuffle(ix)
    return X.iloc[ix].copy(),y.iloc[ix].copy()
def transform(scen,X,y,std,rng):
    if scen=='No shift': return X.copy(),y.copy()
    if scen=='Covariate mean shift (+0.5 SD)': return X+.5*std,y.copy()
    if scen=='Measurement noise (0.5 SD)': return X+rng.normal(0,.5,size=X.shape)*std.values,y.copy()
    if scen=='MCAR missingness (20%)': return X.mask(rng.random(X.shape)<.20),y.copy()
    return balance(X,y,rng)
def stats(y,p):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-6,1-1e-6); pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    # Calibration regression; regularized fit prevents numerical blow-up on small monitor/test sets.
    z=logit(p).reshape(-1,1); c=LogisticRegression(C=1e4,max_iter=2000).fit(z,y)
    return {'brier':float(brier_score_loss(y,p)), 'roc_auc':float(roc_auc_score(y,p)),
      'calibration_intercept':float(c.intercept_[0]),'calibration_slope':float(c.coef_[0,0]),
      'malignant_recall':float(tn/(tn+fn)) if tn+fn else np.nan,
      'accuracy':float(accuracy_score(y,pred)),'malignant_false_negatives':int(fn)}
def choose_strategy(ref,obs):
    db=obs['brier']-ref['brier']; da=obs['roc_auc']-ref['roc_auc']; dr=obs['malignant_recall']-ref['malignant_recall']
    # Pre-specified rule: severe discrimination/sensitivity or Brier deterioration -> retrain;
    # calibration-dominant deterioration -> recalibrate; otherwise retain model.
    if da <= -0.03 or dr <= -0.10 or db >= 0.05:
        return 'Full model retraining','Severe deterioration: ranking, malignant recall, or Brier threshold crossed'
    if db >= 0.02 or abs(obs['calibration_intercept']) >= 0.50 or obs['calibration_slope'] < 0.80 or obs['calibration_slope'] > 1.20:
        return 'Probability recalibration','Calibration/probability deterioration without severe ranking or recall loss'
    return 'No update','No pre-specified deterioration threshold crossed'

d=load_breast_cancer(); X=pd.DataFrame(d.data,columns=d.feature_names); y=pd.Series(d.target,index=X.index)
rows=[]; decisions=[]
for seed in SEEDS:
  Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=seed)
  Xfit,Xadapt,yfit,yadapt=train_test_split(Xtr,ytr,test_size=.30,stratify=ytr,random_state=seed+1000)
  Xmon,Xupd,ymon,yupd=train_test_split(Xadapt,yadapt,test_size=.50,stratify=yadapt,random_state=seed+2000)
  std=Xfit.std().replace(0,1)
  for si,scen in enumerate(SCENARIOS):
    rng=np.random.default_rng(seed*100+si)
    Xm,ymm=transform(scen,Xmon,ymon,std,rng)
    Xu,yuu=transform(scen,Xupd,yupd,std,np.random.default_rng(seed*100+si+1))
    Xt,ytt=transform(scen,Xte,yte,std,np.random.default_rng(seed*100+si+2))
    Xf,yff=transform(scen,Xfit,yfit,std,np.random.default_rng(seed*100+si+3))
    for model_name in MODELS:
      base=build_model(model_name,seed); base.fit(Xfit,yfit)
      # Reference diagnostics on the same monitoring observations before the synthetic shift.
      pref=base.predict_proba(Xmon)[:,1]
      pmon=base.predict_proba(Xm)[:,1]
      ref=stats(ymon,pref); obs=stats(ymm,pmon)
      selected,reason=choose_strategy(ref,obs)
      decisions.append({'seed':seed,'scenario':scen,'model':model_name,'selected_strategy':selected,'reason':reason,
        'reference_brier':ref['brier'],'monitor_brier':obs['brier'],'delta_brier':obs['brier']-ref['brier'],
        'reference_auc':ref['roc_auc'],'monitor_auc':obs['roc_auc'],'delta_auc':obs['roc_auc']-ref['roc_auc'],
        'reference_malignant_recall':ref['malignant_recall'],'monitor_malignant_recall':obs['malignant_recall'],
        'delta_malignant_recall':obs['malignant_recall']-ref['malignant_recall'],
        'monitor_calibration_intercept':obs['calibration_intercept'],'monitor_calibration_slope':obs['calibration_slope']})
      # Construct each policy using update subset only; monitor subset is not reused for fitting updates.
      for strategy in STRATEGIES:
        if strategy=='No update': model=base; p=model.predict_proba(Xt)[:,1]
        elif strategy=='Probability recalibration':
          pu=base.predict_proba(Xu)[:,1]
          cal=LogisticRegression(C=1e4,max_iter=2000).fit(logit(np.clip(pu,1e-6,1-1e-6)).reshape(-1,1),yuu)
          p=cal.predict_proba(logit(np.clip(base.predict_proba(Xt)[:,1],1e-6,1-1e-6)).reshape(-1,1))[:,1]
        else:
          model=build_model(model_name,seed); model.fit(pd.concat([Xf,Xu]),pd.concat([yff,yuu])); p=model.predict_proba(Xt)[:,1]
        rows.append({'seed':seed,'scenario':scen,'model':model_name,'strategy':strategy,**stats(ytt,p)})
      # adaptive selector outcome reuses the corresponding fixed-policy evaluation row (same fitting protocol)
      # Record separately for easy comparison.
      ar=next(r for r in rows[::-1] if r['seed']==seed and r['scenario']==scen and r['model']==model_name and r['strategy']==selected)
      rows.append({**ar,'strategy':'Adaptive selector'})
raw=pd.DataFrame(rows); dec=pd.DataFrame(decisions)
raw.to_csv(f'{OUT}/stage6_policy_results.csv',index=False); dec.to_csv(f'{OUT}/stage6_selection_diagnostics.csv',index=False)
metrics=['brier','roc_auc','malignant_recall','accuracy','calibration_intercept','calibration_slope']
summary=raw.groupby(['scenario','model','strategy'])[metrics].agg(['mean','std']).reset_index(); summary.to_csv(f'{OUT}/stage6_summary_mean_sd.csv',index=False)
counts=dec.groupby(['scenario','model','selected_strategy']).size().rename('count').reset_index(); counts.to_csv(f'{OUT}/stage6_selection_counts.csv',index=False)
# Paired seed-level bootstrap intervals for adaptive vs each fixed policy, within scenario/model.
rng=np.random.default_rng(606); B=2000; ci=[]
for (scen,mod),g in raw.groupby(['scenario','model']):
  piv=g.pivot(index='seed',columns='strategy',values=metrics)
  for fixed in STRATEGIES:
    for met in metrics:
      diff=(piv[(met,'Adaptive selector')]-piv[(met,fixed)]).dropna().values
      boot=np.array([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(B)])
      ci.append({'scenario':scen,'model':mod,'adaptive_vs':fixed,'metric':met,'mean_paired_difference':float(diff.mean()),
       'ci95_lower':float(np.quantile(boot,.025)),'ci95_upper':float(np.quantile(boot,.975)),'n_seeds':len(diff)})
pd.DataFrame(ci).to_csv(f'{OUT}/stage6_paired_bootstrap.csv',index=False)
manifest={'stage':'6 - Adaptive strategy selection','dataset':'Wisconsin Diagnostic Breast Cancer (569 rows, 30 numeric features)','seeds':SEEDS,'models':MODELS,'scenarios':SCENARIOS,
 'split':'Stratified 80:20 train/test; training split into 70:30 fit/adaptation; adaptation split 50:50 monitoring/update. Monitoring data select policy; update data fit recalibration/retraining; test remains untouched.',
 'target':{'0':'malignant','1':'benign'},
 'selector_rule':{'retrain':'delta ROC-AUC <= -0.03 OR delta malignant recall <= -0.10 OR delta Brier >= 0.05','recalibrate':'if not retrain and delta Brier >= 0.02 OR abs(calibration intercept) >= 0.50 OR calibration slope < 0.80 OR > 1.20','no_update':'otherwise'},
 'reference':'Same monitoring observations before synthetic shift; observed metrics are computed after shift. This is a controlled paired diagnostic, not a deployable unsupervised detector.',
 'comparison':'Adaptive selector vs fixed no-update, fixed recalibration, fixed retraining. All use identical fit/update/test partitions.',
 'bootstrap':'2,000 paired bootstrap resamples of 10 seed-level differences; intervals reflect split/seed variability only.',
 'limitations':['Only 10 seeds and two model families (Logistic Regression and Random Forest).','Synthetic shifts on a small benchmark dataset; not external clinical validation.','The selector uses labelled monitoring outcomes and a paired unshifted reference; it is not an unsupervised real-world drift detector.','The small monitoring and update partitions make calibration estimates noisy.','The thresholds are prespecified heuristic thresholds, not clinically validated or optimized.','Balanced prevalence condition uses balanced resampling, not a prospective prior-shift design.','Repeated random splits overlap in source observations; bootstrap intervals are not population-level confidence intervals.']}
json.dump(manifest,open(f'{OUT}/stage6_manifest.json','w'),indent=2)
print('rows',len(raw),'decisions',len(dec))
print('\nSelection counts\n',counts.to_string(index=False))
print('\nOverall means\n',raw.groupby('strategy')[metrics].mean().round(3).to_string())
