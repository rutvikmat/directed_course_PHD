from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data():
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    # scikit-learn encodes malignant=0 and benign=1.
    y = bunch.target.astype(int)
    return X, y, list(X.columns)

def split_data(X, y, test_size=0.20, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def make_logistic_pipeline():
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced",
                                     random_state=42))
    ])

def make_rf_pipeline():
    from sklearn.ensemble import RandomForestClassifier
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
