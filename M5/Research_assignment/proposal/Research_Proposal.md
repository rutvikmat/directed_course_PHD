# A Privacy-Aware, Explainable and Fair Machine Learning Framework for Healthcare Risk Prediction

**Module:** Module 5 – Data Privacy and Predictive Healthcare AI  
**Assignment:** Short Doctoral Research Proposal  
**Research framework name:** PFE-HR (Privacy, Fairness and Explainability for Healthcare Risk Prediction)

## Abstract

Healthcare prediction systems increasingly use machine learning to identify potential health risks and support clinical decision-making. However, conventional predictive models face challenges involving sensitive patient information, unequal predictive performance across demographic groups, limited interpretability and uncertainty about generalisation. This proposal investigates an integrated framework for healthcare risk prediction that jointly evaluates predictive utility, explainability, subgroup fairness and privacy protection.

The research will establish conventional baselines using Logistic Regression, Random Forest and Gradient Boosting. Explainability methods will be used to examine influential features and individual predictions, while subgroup analysis will assess performance differences across available demographic groups. Privacy-preserving extensions will investigate differentially private optimisation and federated learning with secure aggregation where feasible. Experiments will compare predictive utility, calibration, subgroup outcomes, computational cost and formal privacy parameters. The expected contribution is a reproducible evaluation framework and an empirical analysis of the trade-offs between predictive performance, interpretability, fairness and privacy. The work will not be considered a clinical decision system without appropriate external and prospective validation.

## 1. Introduction and problem statement

Healthcare datasets contain sensitive information, including demographics, medical history, clinical measurements and longitudinal health records. Machine learning can identify patterns in such data and support risk stratification. Yet high predictive performance alone is insufficient for responsible healthcare AI.

Centralised data collection can increase exposure to privacy risks. Aggregate performance can conceal different error rates or calibration across patient subgroups. Complex models may also provide limited insight into the factors associated with their predictions. In addition, retrospective results may not generalise across institutions, populations or time periods.

**Problem statement:** How can machine-learning-based healthcare prediction systems be designed and evaluated to maintain predictive utility while improving explainability, subgroup fairness and privacy protection?

## 2. Research gap

Existing work has developed predictive models, explainability techniques, fairness metrics, differential privacy and federated learning. These areas are often evaluated separately, under different datasets and protocols. This makes it difficult to understand the joint trade-offs that arise when privacy mechanisms affect model utility, subgroup performance and explanation behaviour.

The proposed gap is therefore an **integrated and reproducible evaluation of predictive utility, explanation quality, subgroup fairness and privacy protection within a common healthcare prediction workflow**. The proposal does not assume that these topics have never been combined; a systematic literature review will determine what combinations, methods and evaluation protocols are already established.

| Research limitation to investigate | Proposed research response |
|---|---|
| Accuracy-centric evaluation | Report discrimination, calibration and threshold-based performance |
| Explanations assessed separately | Examine global and local explanations and their stability |
| Aggregate-only reporting | Evaluate performance across sufficiently sized subgroups |
| Centralised training assumptions | Investigate federated training as an alternative architecture |
| Unspecified privacy claims | Use a defined threat model and formal privacy accounting for DP experiments |
| Difficult replication | Publish data provenance, preprocessing, splits, configurations and code where permitted |

## 3. Aim and objectives

**Aim:** To develop and evaluate an explainable, subgroup-aware and privacy-conscious machine learning framework for healthcare risk prediction.

**Objectives:**
1. Review healthcare prediction algorithms and relevant work in explainability, fairness, differential privacy and federated learning.
2. Define a clinically meaningful binary prediction outcome, cohort and prediction horizon.
3. Establish reproducible baseline predictive models.
4. Evaluate global and local explanations and, where possible, explanation stability.
5. Measure predictive performance and calibration across relevant demographic subgroups.
6. Implement and evaluate a privacy-preserving extension using differentially private training and/or federated learning.
7. Quantify privacy–utility–fairness trade-offs and document limitations.

## 4. Proposed algorithmic contribution

The proposed research framework is named **PFE-HR: Privacy, Fairness and Explainability for Healthcare Risk Prediction**. It is an integrated workflow and experimental framework at this stage, not a claim of a newly invented algorithm.

The workflow contains four connected components:

1. **Predictive modelling:** Train Logistic Regression, Random Forest and a gradient-boosting baseline using leakage-aware preprocessing.
2. **Explainability:** Use global feature importance and local explanations. Evaluate explanation stability across resamples or model variants where feasible.
3. **Fairness assessment:** Report subgroup sample sizes, prevalence, discrimination, calibration and error rates. Select fairness measures appropriate to the clinical context rather than assuming one metric is universally correct.
4. **Privacy-preserving learning:** Compare centralised training with a differentially private training approach and/or federated learning. Federated learning alone is not a privacy guarantee; secure aggregation and additional protections may be required.

The potential contribution is a systematic, reproducible evaluation protocol that studies how privacy mechanisms and model choices interact with predictive utility, explanation behaviour and subgroup outcomes. Novelty will be assessed through the literature review and comparative experiments.

## 5. Experimental design

### 5.1 Dataset and cohort

The initial software prototype may use synthetic data for pipeline testing only. The research experiment should use an appropriately licensed public dataset or an institutionally approved healthcare dataset. Candidate sources include the UCI Diabetes 130-US Hospitals dataset, the Pima Indians Diabetes dataset and MIMIC-IV, subject to suitability, access conditions and governance requirements.

The final dataset will be selected based on:
- a clearly defined target outcome and prediction horizon;
- suitable cohort size and relevant predictor variables;
- availability of demographic variables for subgroup analysis;
- data quality, missingness and representativeness;
- licence, access and ethics requirements.

No direct identifiers should be used as model predictors. Patient-level splits must prevent records from the same patient appearing in both training and test sets. Temporal validation and external validation should be used where data permit.

### 5.2 Models and comparison

| Role | Candidate method |
|---|---|
| Interpretable baseline | Logistic Regression |
| Tree ensemble | Random Forest |
| Boosting baseline | XGBoost or HistGradientBoosting |
| Privacy extension | DP-SGD with a documented accountant |
| Distributed extension | Federated Averaging, with secure aggregation where feasible |

### 5.3 Evaluation metrics

| Dimension | Measures |
|---|---|
| Predictive performance | Accuracy, precision, recall, F1-score |
| Discrimination | ROC-AUC and PR-AUC |
| Calibration | Brier score and calibration plots |
| Explainability | SHAP or suitable model-specific explanations; permutation importance; stability |
| Fairness | Subgroup TPR, FPR, PPV, selection rate and calibration; selected parity measures where justified |
| Privacy | Defined threat model, epsilon (ε), delta (δ), privacy accountant and mechanism parameters |
| Efficiency | Training time, communication overhead and convergence |

### 5.4 Experimental stages

- **Experiment 1 – Baselines:** Train and compare conventional models using a prespecified split and validation protocol.
- **Experiment 2 – Explainability:** Generate global and local explanations and assess whether important features and explanations are stable.
- **Experiment 3 – Subgroup analysis:** Evaluate performance, calibration and error rates across justified demographic groups. Report denominators and uncertainty.
- **Experiment 4 – Differential privacy:** Apply a recognised DP-SGD implementation, define the protected unit and adjacency relation, clip per-example gradients, calibrate noise and use a privacy accountant to report ε for a stated δ and training schedule.
- **Experiment 5 – Federated learning:** Partition training data across simulated or approved clients, compare federated and centralised training, and document whether secure aggregation is implemented or only proposed.
- **Experiment 6 – Trade-off analysis:** Compare predictive utility, calibration, subgroup performance, privacy budget and computational cost across configurations.

Model selection must use validation data or nested cross-validation. A locked test set should be used only for final evaluation. Where possible, report confidence intervals and repeated-seed variability.

## 6. Ethical considerations

1. **Data governance:** Use only data permitted by the relevant licence, data-use agreement and institutional policy.
2. **Ethics review:** Obtain Institutional Ethics Committee or other required approval before accessing restricted patient data or conducting research requiring review.
3. **Consent and lawful basis:** Follow the dataset's consent conditions and applicable legal and institutional requirements.
4. **Data minimisation:** Process only variables required for the research question and avoid direct identifiers.
5. **Security:** Apply access controls, encryption, secure storage, retention limits and documented deletion procedures.
6. **Fairness:** Examine representativeness and potential harm across subgroups; do not interpret small or unstable subgroup estimates as definitive.
7. **Transparency:** Document data provenance, missingness, exclusions, model limitations and all relevant experiments.
8. **Human oversight:** Treat outputs as research or decision-support information, not autonomous diagnosis. Clinical use would require separate validation, governance and oversight.
9. **Privacy risk:** De-identification does not eliminate re-identification risk. Federated learning does not by itself prevent leakage from model updates.

## 7. Reproducibility plan

- Maintain version-controlled source code and a clear project structure.
- Record dataset name, version, access date, licence and permitted use.
- Document cohort construction, target definition, preprocessing and missing-data handling.
- Use patient-level and, where appropriate, temporal data splits.
- Fix and record random seeds; report repeated runs where feasible.
- Pin software dependencies and record Python, library and hardware versions.
- Save experiment configurations, logs, metrics, plots and model artefacts.
- Record privacy mechanism, protected unit, clipping norm, noise multiplier, sampling rate, steps, δ and accountant version; report ε only when formally computed.
- Preserve negative and unsuccessful results, not only the best run.
- Share code and non-sensitive artefacts where legally and ethically permitted. Do not redistribute restricted patient data.
- Provide a README with exact execution instructions.

## 8. Expected outcomes

1. A reproducible healthcare prediction benchmark and pipeline.
2. Comparative results for conventional predictive models.
3. Global and local explanation outputs and an assessment of explanation stability.
4. Subgroup performance and fairness audit.
5. A privacy-preserving training demonstration with clearly documented assumptions and privacy accounting, where implemented.
6. An empirical analysis of privacy–utility–fairness trade-offs.
7. A transparent account of limitations and future validation requirements.

## 9. Limitations and future work

Potential limitations include restricted dataset access, incomplete or biased clinical variables, small subgroup sizes, distribution shift and computational overhead. Differential privacy may reduce model utility, and privacy guarantees depend on the threat model, protected unit and correct accounting. Federated learning may expose information through model updates unless suitable protections are implemented.

Future work may investigate longitudinal prediction, multi-institutional validation, adaptive fairness monitoring, secure aggregation, robust explanation methods and prospective clinical evaluation.

## 10. Indicative six-month timeline

| Period | Activity |
|---|---|
| Month 1 | Literature review, research gap refinement and outcome definition |
| Month 2 | Dataset governance, cohort preparation and baseline modelling |
| Month 3 | Explainability and subgroup fairness experiments |
| Month 4 | Differential privacy implementation and accounting |
| Month 5 | Federated learning, ablations and comparative experiments |
| Month 6 | Analysis, reproducibility audit and proposal/report refinement |

## 11. Conclusion

This proposal investigates healthcare risk prediction as a multi-dimensional problem involving predictive utility, explainability, subgroup fairness and privacy. It proposes PFE-HR as an integrated research and evaluation framework, with conventional predictive baselines and privacy-preserving extensions. The study will assess the trade-offs through reproducible experiments and clearly distinguish demonstrated results from proposed work. The framework is intended for research evaluation and is not a clinical diagnostic system.

## References

1. Dwork, C. (2006). Differential Privacy. In *Proceedings of the 33rd International Colloquium on Automata, Languages and Programming (ICALP)*.
2. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. In *Proceedings of AISTATS*.
3. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems*.
4. Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. In *Advances in Neural Information Processing Systems*.
5. Kaissis, G. A., Makowski, M. R., Rückert, D., & Braren, R. F. (2020). Secure, privacy-preserving and federated machine learning in medical imaging. *Nature Machine Intelligence, 2*, 305–311.
6. Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine Learning in Medicine. *New England Journal of Medicine, 380*, 1347–1358.
