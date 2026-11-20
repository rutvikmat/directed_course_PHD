import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from directed_course.Module_2.M2.Dynamical_Structure_Research.src.pipeline import metrics, save_csv, RESULTS

def run(predictions):
    df=predictions.copy()
    df['timestamp']=pd.to_datetime(df.timestamp)
    df['absolute_change']=df.actual.diff().abs()
    threshold=df.absolute_change.quantile(.90)
    df['regime']=np.where(df.absolute_change>=threshold,'Large change','Typical change')
    # Test-period stratification: descriptive, not a causal intervention.
    rows=[]
    for regime,group in df.groupby('regime'):
        for name in ['SeasonalNaive','StateSpace','RandomForest','XGBoost','Hybrid']:
            rows.append({'Regime':regime,'Model':name,'Count':len(group),**metrics(group.actual,group[name])})
    result=pd.DataFrame(rows)
    save_csv(result,'exp4_robustness.csv')
    pivot=result.pivot(index='Model',columns='Regime',values='RMSE')
    ax=pivot.plot(kind='bar',figsize=(10,5),rot=25)
    ax.set_ylabel('RMSE (kW)')
    ax.set_title('Performance across observed demand-change regimes')
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS/'exp4_robustness.png',dpi=160)
    plt.close(ax.figure)
    return result
