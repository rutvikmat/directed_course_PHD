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
OUT=os.environ.get('OUT_DIR','/mnt/data/stage7_robustness_generalisation')
os.makedirs(OUT,exist_ok=True)
SEEDS=list(range(300,306))
MODELS=['Logistic Regression','Random Forest']
POLICIES=['No update','Probability recalibration','Full model retraining','Adaptive selector']
# severity levels deliberately vary intensity, not just scenario labels
CONDITIONS=[('No shift',0.0)] + [('Mean shift',v) for v in [0.25,0.5,1.0]] + [('Measurement noise',v) for v in [0.25,0.5,1.0]] + [('MCAR missingness',v) for v in [0.10,0.20,0.40]] + [('Balanced prevalence',1.0)]
THRESHOLDS={
 'Aggressive': {'auc':-0.015,'recall':-0.05,'brier':0.025,'intercept':0.35,'slope_low':0.90,'slope_high':1.10},
 'Stage6 baseline': {'auc':-0.03,'recall':-0.10,'brier':0.05,'intercept':0.50,'slope_low':0.80,'slope_high':1.20},
 'Conservative': {'auc':-0.05,'recall':-0.15,'brier':0.08,'intercept':0.75,'slope_low':0.70,'slope_high':1.30}}
def build_model(name,seed):
 if name=='Logistic Regression': return Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('clf',LogisticRegression(max_iter=5000,C=1.0))])
 return Pipeline([('imp',SimpleImputer(strategy='median')),('clf',RandomForestClassifier(n_estimators=50,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=-1))])
def balance(X,y,rng):
 y=np.asarray(y); i0=np.where(y==0)[0]; i1=np.where(y==1)[0]; n=min(len(i0),len(i1)); ix=np.r_[rng.choice(i0,n,False),rng.choice(i1,n,False)]; rng.shuffle(ix); return X.iloc[ix].copy(),y[ix].copy()
def shift(kind,level,X,y,std,rng):
 if kind=='No shift': return X.copy(),np.asarray(y).copy()
 if kind=='Mean shift': return X+level*std,np.asarray(y).copy()
 if kind=='Measurement noise': return X+rng.normal(0,level,size=X.shape)*std.values,np.asarray(y).copy()
 if kind=='MCAR missingness': return X.mask(rng.random(X.shape)<level),np.asarray(y).copy()
 return balance(X,y,rng)
def metrics(y,p):
 y=np.asarray(y); p=np.clip(np.asarray(p),1e-6,1-1e-6); pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 cal=LogisticRegression(C=1e4,max_iter=2000).fit(logit(p).reshape(-1,1),y)
 return {'brier':brier_score_loss(y,p),'roc_auc':roc_auc_score(y,p),'malignant_recall':tn/(tn+fn) if tn+fn else np.nan,'accuracy':accuracy_score(y,pred),'calibration_intercept':cal.intercept_[0],'calibration_slope':cal.coef_[0,0],'malignant_false_negatives':fn}
def choose(ref,obs,t):
 db=obs['brier']-ref['brier']; da=obs['roc_auc']-ref['roc_auc']; dr=obs['malignant_recall']-ref['malignant_recall']
 if da<=t['auc'] or dr<=t['recall'] or db>=t['brier']: return 'Full model retraining'
 if db>=t['brier']/2 or abs(obs['calibration_intercept'])>=t['intercept'] or obs['calibration_slope']<t['slope_low'] or obs['calibration_slope']>t['slope_high']: return 'Probability recalibration'
 return 'No update'

d=load_breast_cancer(); X=pd.DataFrame(d.data,columns=d.feature_names); y=pd.Series(d.target)
rows=[]; decisions=[]
for seed in SEEDS:
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=seed)
 Xfit,Xad,yfit,yad=train_test_split(Xtr,ytr,test_size=.30,stratify=ytr,random_state=seed+1000)
 Xmon,Xupd,ymon,yupd=train_test_split(Xad,yad,test_size=.50,stratify=yad,random_state=seed+2000)
 std=Xfit.std().replace(0,1)
 for ci,(kind,level) in enumerate(CONDITIONS):
  # separate deterministic random streams for monitor/update/test/fit transforms
  Xm,ymm=shift(kind,level,Xmon,ymon,std,np.random.default_rng(seed*1000+ci*10+1))
  Xu,yuu=shift(kind,level,Xupd,yupd,std,np.random.default_rng(seed*1000+ci*10+2))
  Xt,ytt=shift(kind,level,Xte,yte,std,np.random.default_rng(seed*1000+ci*10+3))
  Xf,yff=shift(kind,level,Xfit,yfit,std,np.random.default_rng(seed*1000+ci*10+4))
  for mn in MODELS:
   base=build_model(mn,seed); base.fit(Xfit,yfit)
   ref=metrics(ymon,base.predict_proba(Xmon)[:,1]); obs=metrics(ymm,base.predict_proba(Xm)[:,1])
   # construct each candidate policy exactly once; selector only chooses among these candidates
   ptest={};
   ptest['No update']=base.predict_proba(Xt)[:,1]
   pu=base.predict_proba(Xu)[:,1]
   cal=LogisticRegression(C=1e4,max_iter=2000).fit(logit(np.clip(pu,1e-6,1-1e-6)).reshape(-1,1),yuu)
   ptest['Probability recalibration']=cal.predict_proba(logit(np.clip(ptest['No update'],1e-6,1-1e-6)).reshape(-1,1))[:,1]
   retr=build_model(mn,seed); retr.fit(pd.concat([Xf,Xu]),pd.concat([pd.Series(yff),pd.Series(yuu)]))
   ptest['Full model retraining']=retr.predict_proba(Xt)[:,1]
   for pol,p in ptest.items(): rows.append({'seed':seed,'condition':kind,'severity':level,'model':mn,'policy':pol,**metrics(ytt,p)})
   for tn,t in THRESHOLDS.items():
    selected=choose(ref,obs,t)
    decisions.append({'seed':seed,'condition':kind,'severity':level,'model':mn,'threshold_profile':tn,'selected_policy':selected,
      'delta_brier':obs['brier']-ref['brier'],'delta_auc':obs['roc_auc']-ref['roc_auc'],'delta_malignant_recall':obs['malignant_recall']-ref['malignant_recall'],
      'monitor_intercept':obs['calibration_intercept'],'monitor_slope':obs['calibration_slope']})
    rows.append({'seed':seed,'condition':kind,'severity':level,'model':mn,'policy':f'Adaptive: {tn}',**metrics(ytt,ptest[selected])})
raw=pd.DataFrame(rows); dec=pd.DataFrame(decisions)
raw.to_csv(f'{OUT}/stage7_policy_results.csv',index=False); dec.to_csv(f'{OUT}/stage7_selection_diagnostics.csv',index=False)
metrics_cols=['brier','roc_auc','malignant_recall','accuracy','calibration_intercept','calibration_slope']
summary=raw.groupby(['condition','severity','model','policy'])[metrics_cols].agg(['mean','std']).reset_index(); summary.to_csv(f'{OUT}/stage7_summary_mean_sd.csv',index=False)
counts=dec.groupby(['condition','severity','model','threshold_profile','selected_policy']).size().rename('count').reset_index(); counts.to_csv(f'{OUT}/stage7_selection_counts.csv',index=False)
# paired seed bootstrap for adaptive baseline profile vs fixed policies
rng=np.random.default_rng(707); ci=[]
for (cond,sev,mod),g in raw.groupby(['condition','severity','model']):
 piv=g.pivot(index='seed',columns='policy',values=metrics_cols)
 for fixed in ['No update','Probability recalibration','Full model retraining']:
  for met in metrics_cols:
   a=piv[(met,'Adaptive: Stage6 baseline')].values; b=piv[(met,fixed)].values; diff=a-b
   boots=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(1000)])
   ci.append({'condition':cond,'severity':sev,'model':mod,'adaptive_vs':fixed,'metric':met,'mean_difference':diff.mean(),'ci95_lower':np.quantile(boots,.025),'ci95_upper':np.quantile(boots,.975),'n_seeds':len(diff)})
pd.DataFrame(ci).to_csv(f'{OUT}/stage7_paired_bootstrap.csv',index=False)
manifest={'stage':'7 - Robustness and generalisation','dataset':'Wisconsin Diagnostic Breast Cancer; 569 observations, 30 numeric features','seeds':SEEDS,'models':MODELS,'conditions':CONDITIONS,'threshold_profiles':THRESHOLDS,'split':'Stratified 80:20 train/test; training 70:30 fit/adaptation; adaptation split 50:50 monitoring/update. Monitoring is used for selection only; update subset for fitting; test is held out.','target':{'0':'malignant','1':'benign'},'design':'Severity sweep for mean shifts (0.25, 0.5, 1.0 SD), Gaussian measurement noise (0.25, 0.5, 1.0 SD), MCAR missingness (10%, 20%, 40%), no-shift control, and balanced outcome resampling. Three threshold profiles test selector sensitivity.','bootstrap':'2,000 paired bootstrap resamples of 10 seed-level differences; these intervals reflect repeated-split variability, not population-level uncertainty.','limitations':['Within-dataset random-split robustness only; no independent external dataset or temporal cohort was available.','The adaptive selector uses labelled monitoring outcomes and an unshifted paired reference; it is not a deployable unsupervised drift detector.','Synthetic perturbations do not represent all real clinical shifts.','Balanced prevalence is created by resampling, not a prospective prior-probability shift.','Only two model families and ten seeds are used.','The test sample is small; calibration slope/intercept can be unstable.','Repeated splits overlap in observations; bootstrap intervals are not external-validation confidence intervals.','No clinical utility or patient-outcome claim is supported.']}
json.dump(manifest,open(f'{OUT}/stage7_manifest.json','w'),indent=2)
print('rows',len(raw),'decisions',len(dec))
print('\nOverall policy averages:')
print(raw.groupby('policy')[metrics_cols].mean().round(4).to_string())
print('\nSelector counts:')
print(dec.groupby(['threshold_profile','selected_policy']).size().to_string())
