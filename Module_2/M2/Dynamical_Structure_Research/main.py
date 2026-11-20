import argparse
import pandas as pd
from directed_course.Module_2.M2.Dynamical_Structure_Research.experiments import exp1_accuracy, exp2_interpretability, exp3_data_efficiency
from directed_course.Module_2.M2.Dynamical_Structure_Research.src.pipeline import build_context, save_csv
from directed_course.Module_2.M2.Dynamical_Structure_Research.experiments import exp4_robustness
 
def main():
    p=argparse.ArgumentParser(description='Dynamical structure research experiments')
    p.add_argument('--max-hours',type=int,default=None,help='Use last N hourly records for faster trial runs')
    p.add_argument('--fast',action='store_true',help='Use smaller tree ensembles')
    p.add_argument('--lstm',action='store_true',help='Train optional PyTorch LSTM')
    p.add_argument('--epochs',type=int,default=12)
    args=p.parse_args()
    frame,tr,va,te,base_cols,hybrid_cols,ssm=build_context(args.max_hours)
    print(f'Rows: {len(frame)} | train={len(tr)} validation={len(va)} test={len(te)}')
    print('Train:',frame.index[tr[0]],'to',frame.index[tr[-1]])
    print('Test:',frame.index[te[0]],'to',frame.index[te[-1]])
    print('\nEXPERIMENT 1 — ACCURACY')
    accuracy,preds,fitted=exp1_accuracy.run(frame,tr,va,te,base_cols,hybrid_cols,args.fast)
    if args.lstm:
        from directed_course.Module_2.M2.Dynamical_Structure_Research.src.lstm import run as run_lstm
        output=run_lstm(frame,tr,va,te,epochs=args.epochs)
        if output:
            pred,m=output
            preds['LSTM']=pred
            save_csv(preds,'exp1_predictions.csv')
            accuracy=pd.concat([accuracy,pd.DataFrame([{'Model':'LSTM',**m}])],ignore_index=True).sort_values('RMSE')
            save_csv(accuracy,'exp1_accuracy.csv')
    print('\nEXPERIMENT 2 — INTERPRETABILITY')
    exp2_interpretability.run(frame,te,base_cols,hybrid_cols,fitted)
    print('\nEXPERIMENT 3 — DATA EFFICIENCY')
    exp3_data_efficiency.run(frame,tr,va,te,base_cols,hybrid_cols,args.fast)
    print('\nEXPERIMENT 4 — ROBUSTNESS')
    exp4_robustness.run(preds)
    print('\nDone. Results are in results/.')

if __name__=='__main__':
    main()
