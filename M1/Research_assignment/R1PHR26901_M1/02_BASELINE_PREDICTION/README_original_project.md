# Wisconsin Breast Cancer ML Practical

## Contents
- `wisconsin_breast_cancer_model_comparison.ipynb`: complete Jupyter Notebook experiment
- `requirements.txt`: Python dependencies
- `outputs/`: generated CSV results (created when the notebook is run)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate    # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open `wisconsin_breast_cancer_model_comparison.ipynb` and run the cells from top to bottom.

## Dataset
The notebook loads the Wisconsin Breast Cancer Diagnostic dataset using
`sklearn.datasets.load_breast_cancer`, so no manual CSV download is needed.

Target encoding:
- 0 = malignant
- 1 = benign

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- SVM (RBF)
- XGBoost

## Methodology
- Stratified 80:20 train/test split
- Stratified 5-fold cross-validation on the training set
- Imputation/scaling inside pipelines to prevent preprocessing leakage
- Held-out test evaluation: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrices, ROC curves, feature importance and error analysis

## Important
This is a coursework/educational experiment, not a clinical diagnostic tool.
The notebook's metrics do not establish clinical validity or readiness.
