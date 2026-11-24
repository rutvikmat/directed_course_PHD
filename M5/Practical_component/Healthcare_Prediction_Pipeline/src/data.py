import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def make_synthetic_health_data(n_samples=2400, random_state=42):
    """Create synthetic, non-clinical data with age/sex fields for subgroup exercises."""
    X, y = make_classification(
        n_samples=n_samples, n_features=8, n_informative=6, n_redundant=1,
        n_repeated=0, n_classes=2, weights=[0.68, 0.32],
        class_sep=0.9, flip_y=0.025, random_state=random_state
    )
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(X, columns=[
        "biomarker_a", "biomarker_b", "blood_pressure_index",
        "glucose_index", "body_mass_index", "activity_index",
        "sleep_quality_index", "family_history_score"
    ])
    df["age"] = rng.integers(18, 86, size=n_samples)
    df["sex"] = rng.choice(["female", "male"], size=n_samples)
    # Inject modest missingness to exercise preprocessing.
    for col in ["biomarker_a", "glucose_index", "body_mass_index"]:
        mask = rng.random(n_samples) < 0.04
        df.loc[mask, col] = np.nan
    df["risk_outcome"] = y.astype(int)
    return df

def load_data(path=None, target="risk_outcome", random_state=42):
    if path:
        df = pd.read_csv(path)
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")
        y = df[target]
        if y.isna().any() or set(pd.Series(y).dropna().unique()) - {0, 1}:
            raise ValueError("Target must be complete and binary, encoded as 0 and 1.")
        return df.drop(columns=[target]), y.astype(int), "user_csv"
    df = make_synthetic_health_data(random_state=random_state)
    return df.drop(columns=[target]), df[target], "synthetic_demo"
