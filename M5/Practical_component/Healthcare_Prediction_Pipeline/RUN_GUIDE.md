# Run guide

1. Extract the ZIP.
2. Open a terminal in the extracted project directory.
3. Create and activate a virtual environment.
4. Install dependencies with `pip install -r requirements.txt`.
5. Run `python src/run_pipeline.py`.
6. Inspect the generated `results/` directory.

## Common issues
- If optional SHAP fails to install or run, the pipeline still generates permutation importance and local perturbation explanations.
- If using a CSV, encode the target as 0 and 1. Do not include identifiers, names, phone numbers, addresses or direct identifiers.
- The code assumes a binary classification task. Adapt outcome definition and metrics for survival, regression or multi-class problems.
- For a small dataset, subgroup metrics may be unstable; report denominators and confidence intervals in a formal study.
