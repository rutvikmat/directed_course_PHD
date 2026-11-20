import pandas as pd
import matplotlib.pyplot as plt
from directed_course.Module_2.M2.Dynamical_Structure_Research.src.pipeline import RESULTS, save_csv

def run(frame, te, base_cols, hybrid_cols, fitted):
    for name, cols in [('RandomForest',base_cols),('XGBoost',base_cols),('Hybrid',hybrid_cols)]:
        model = fitted[name]
        importance = pd.DataFrame({'feature':cols,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
        save_csv(importance, f'exp2_{name}_importance.csv')
        top = importance.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9,6))
        ax.barh(top.feature, top.importance)
        ax.set_title(f'{name}: feature importance')
        fig.tight_layout()
        fig.savefig(RESULTS / f'exp2_{name}_importance.png',dpi=160)
        plt.close(fig)
    subset = frame.iloc[te].iloc[:336]
    state_cols = [c for c in frame.columns if c.startswith('state_')]
    if state_cols:
        fig, ax = plt.subplots(figsize=(11,4))
        ax.plot(subset.index,subset[state_cols[0]],label='Filtered local level')
        ax.plot(subset.index,subset.target,label='Observed next-hour power',alpha=.65)
        ax.legend()
        ax.set_title('Dynamical state evolution (illustrative window)')
        fig.tight_layout()
        fig.savefig(RESULTS/'exp2_state_evolution.png',dpi=160)
        plt.close(fig)
    try:
        import shap
        sample = frame.iloc[te][hybrid_cols].sample(n=min(200,len(te)),random_state=42)
        explanation = shap.TreeExplainer(fitted['Hybrid'])(sample)
        shap.summary_plot(explanation.values,sample,show=False,max_display=15)
        plt.tight_layout()
        plt.savefig(RESULTS/'exp2_hybrid_shap.png',dpi=160,bbox_inches='tight')
        plt.close()
    except (ImportError, Exception) as exc:
        print('SHAP plot skipped:',exc)
