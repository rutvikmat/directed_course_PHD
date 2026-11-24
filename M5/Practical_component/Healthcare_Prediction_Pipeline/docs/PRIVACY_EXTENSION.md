# Proposed Privacy-Preserving Extension

## Current prototype boundary
The executable `privacy_demo.py` clips values and adds random perturbation to one aggregate mean. It is included only to make the concept tangible. It is **not** a complete differentially private learning algorithm and does not establish a formal privacy guarantee. No epsilon (ε) or delta (δ) is claimed.

## Extension A: Differentially private training
A defensible implementation should:
1. Define the protected unit (one record or one person) and adjacency relation.
2. Bound per-example contribution using gradient clipping.
3. Add calibrated Gaussian noise to clipped gradients.
4. Use a privacy accountant to calculate ε for a chosen δ and the complete training schedule.
5. Report utility against privacy budgets, including multiple seeds.
6. Document threat model, clipping norm, noise multiplier, sampling rate, steps, δ, accountant and library versions.

Possible implementation route: a vetted DP-SGD library such as Opacus for PyTorch, or a maintained differential-privacy library with documented accounting. Do not hand-label a model “differentially private” merely because random noise was added.

## Extension B: Federated learning
Simulate at least 3 clients, partitioning synthetic or approved de-identified records across clients. Each client trains locally; only model updates are transmitted. Compare:
- centralised baseline,
- federated averaging,
- federated averaging with secure aggregation (conceptual or supported implementation).

Federated learning alone does not guarantee privacy. Model updates may leak information. Secure aggregation, clipping, differential privacy, transport security, access controls and governance may be required.

## Evaluation design
Compare predictive utility (ROC-AUC, PR-AUC, Brier score, calibration), subgroup metrics, communication cost, convergence and privacy budget where formally measurable. Include ablations and confidence intervals.

## Threats to discuss
- Membership inference: whether a record was in training data.
- Model inversion: inferring sensitive attributes or representative inputs.
- Gradient leakage: reconstructing training examples from updates.
- Re-identification: linkage of supposedly de-identified records.

## Proposed research question
How does the addition of a formally accounted privacy mechanism affect predictive utility and subgroup performance in a federated healthcare risk-prediction setting?
