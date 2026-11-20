import pandas as pd
import matplotlib.pyplot as plt
from directed_course.Module_2.M2.Dynamical_Structure_Research.src.pipeline import create_models, metrics, save_csv, RESULTS

def run(frame, tr, va, te, base_cols, hybrid_cols, fast=False):
    # Fixed test set; nested chronological training prefixes. State features were
    # fitted once on the full training window: conditional feature ablation,
    # NOT a strict end-to-end low-data test of the state-space component.
    fractions=[.10,.25,.50,.75,1.0]
    rows=[]
    y=frame.target
    for fraction in fractions:
        n=max(200,int(len(tr)*fraction))
        sub=tr[:n]
        for name,cols in [('RandomForest',base_cols),('XGBoost',base_cols),('Hybrid',hybrid_cols)]:
            model=create_models(fast=fast)['XGBoost' if name=='Hybrid' else name]
            model.fit(frame.iloc[sub][cols],y.iloc[sub])
            m=metrics(y.iloc[te],model.predict(frame.iloc[te][cols]))
            rows.append({'TrainingFraction':fraction,'Model':name,'TrainingRows':len(sub),**m})
            print(f'EXP3 {fraction:.0%} {name}: RMSE {m["RMSE"]:.4f}')
    results=pd.DataFrame(rows)
    save_csv(results,'exp3_data_efficiency.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    for name,group in results.groupby('Model'):
        ax.plot(group.TrainingFraction*100,group.RMSE,marker='o',label=name)
    ax.set(xlabel='Training data (%)',ylabel='Test RMSE (kW)',title='Conditional data efficiency (fixed state estimator)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS/'exp3_data_efficiency.png',dpi=160)
    plt.close(fig)
    return results
