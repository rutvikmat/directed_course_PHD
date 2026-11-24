import numpy as np
import pandas as pd

def clipped_noisy_mean_demo(values, clip_norm=1.0, noise_multiplier=0.8, random_state=42):
    """
    Illustrative aggregate perturbation only.
    This is NOT a complete DP mechanism: no adjacency definition, sensitivity proof,
    privacy accountant, composition analysis, or calibrated epsilon/delta guarantee.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("No finite values supplied.")
    clipped = np.clip(x, -clip_norm, clip_norm)
    true_mean = float(clipped.mean())
    rng = np.random.default_rng(random_state)
    # Illustrative noise, deliberately not advertised as a formal DP guarantee.
    noisy_mean = float(true_mean + rng.normal(0, noise_multiplier * clip_norm / len(clipped)))
    return {
        "n_records": int(len(clipped)),
        "clip_bound": float(clip_norm),
        "noise_multiplier_demo_only": float(noise_multiplier),
        "clipped_mean": true_mean,
        "perturbed_mean_demo_only": noisy_mean,
        "formal_epsilon": None,
        "formal_delta": None,
        "privacy_guarantee": "NOT ESTABLISHED - educational perturbation only"
    }

def save_privacy_demo(y_train, output_path):
    result = clipped_noisy_mean_demo(y_train)
    pd.DataFrame([result]).to_csv(output_path, index=False)
    return result
