# R1PHR26901 — Final Coursework Submission Package

**Project:** Predictive Modelling and Reliability under Distribution Shift using the Wisconsin Diagnostic Breast Cancer dataset  
**Student:** Rutvik Mathapati  
**Programme:** Ph.D. in Computer Science — AI/ML  
**Institution:** Alliance University, Bengaluru  
**Course:** R1PHR26901

## What this package contains

- `01_MAIN_REPORT/`: consolidated final report (DOCX and PDF).
- `02_BASELINE_PREDICTION/`: baseline practical report, notebook, figures and metrics.
- `03_SHIFT_EXPERIMENTS/`: controlled synthetic shift experiments.
- `04_UPDATE_STRATEGY/`: comparison of no update, recalibration and retraining.
- `05_REPEATED_RELIABILITY/`: repeated-seed evaluation and paired bootstrap outputs.
- `06_ADAPTIVE_SELECTION/`: Stage 6 adaptive policy-selection experiment.
- `07_ROBUSTNESS_GENERALISATION/`: Stage 7 severity and threshold sensitivity experiments.
- `08_LITERATURE_REGISTER/`: DOI/bibliographic verification register; this is a verification aid, not a completed systematic review.
- `09_SUBMISSION_CHECKLIST/`: supervisor-facing checklist and submission guidance.

## Recommended submission order

1. Read the consolidated report.
2. Submit the consolidated report as the main coursework document, subject to supervisor/course template requirements.
3. Attach the notebook and scripts as reproducibility supplementary material if permitted.
4. Include the results workbook and CSVs as supplementary evidence.
5. Keep the full ZIP as the complete reproducibility archive.

## Important scope statement

This is a within-dataset, synthetic-shift research practical using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Stages 2–7 do **not** constitute external validation, temporal validation, prospective evaluation, clinical validation, or deployment readiness. All distribution shifts are simulated. The adaptive selector uses labelled monitoring outcomes and a paired unshifted reference; it is not an unsupervised deployment drift detector. The balanced-prevalence condition uses resampling and is not a prospective prior-probability-shift cohort.

The project is appropriate to submit as a preliminary coursework investigation and research prototype, provided the supervisor accepts the scope. Independent external validation is intentionally deferred pending supervisor approval and dataset/access feasibility.

## Reproducibility

Python 3.11 is recommended. Install the baseline requirements using the included `requirements.txt`. Stage scripts may require the packages listed below; versions are not fully pinned in all historical scripts, so use a clean virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn xgboost openpyxl python-docx
```

Run each stage from its own folder so output paths resolve correctly:

```bash
python run_shift_experiments.py
python run_update_strategy_experiments.py
python run_stage5.py
python run_stage6.py
python run_stage7.py
```

Some scripts use fixed output-directory assumptions. Review the first configuration section of each script before rerunning outside its original folder. The baseline notebook can be opened in Jupyter.

## Data and ethics

WDBC is a publicly available, de-identified benchmark dataset. No patient records were collected for this coursework. The results are educational/research findings and must not be used for diagnosis, triage, or treatment.
