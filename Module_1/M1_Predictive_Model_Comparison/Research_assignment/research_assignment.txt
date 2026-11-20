# Research Assignment

## Title
**Formulating a Prediction Problem, Establishing Baselines, Designing an Evaluation Protocol, and Identifying a Potential Methodological Research Gap**

### 1. Introduction

Prediction problems are an important component of machine learning research. They involve developing computational models that learn patterns from historical data and use those patterns to estimate future outcomes.

In healthcare, predictive modelling can support early identification of patterns, personalized forecasting, and data-driven decision-making.

This assignment considers menstrual cycle prediction as a real-world time-series prediction problem, with particular emphasis on irregular and dynamically changing menstrual cycles.

### 2. Formulation of the Prediction Problem

**2.1 Problem Definition**

The objective is to predict the date of the next menstrual cycle onset using an individual's historical menstrual cycle records.

Traditional prediction methods often assume that menstrual cycle patterns remain relatively stable. However, cycle lengths can change over time due to several factors, including stress, sleep patterns, lifestyle changes, and other individual variations.

Therefore, the prediction problem is formulated as:

*Given an individual's historical menstrual cycle observations, can a machine learning model accurately predict the next menstrual cycle onset while adapting to changes in the individual's cycle patterns over time?*

**2.2 Mathematical Formulation**

Let the historical menstrual cycle onset dates for an individual be:

\[
D = \{d_1,d_2,\ldots,d_t\}
\]

The observed cycle lengths are calculated as:

\[
L_i = d_{i+1}-d_i
\]

where \(L_i\) represents the number of days between two consecutive menstrual cycle onset dates.

The prediction objective is to estimate the next cycle length:

\[
\hat{L}_{t+1}=f(L_1,L_2,\ldots,L_t,X_t)
\]

where:

- \(f\) represents the predictive model.
- \(L_1,\ldots,L_t\) represent historical cycle lengths.
- \(X_t\) represents optional contextual variables.
- \(\hat{L}_{t+1}\) represents the predicted next cycle length.

The predicted next menstrual onset date is:

\[
\hat{d}_{t+1}=d_t+\hat{L}_{t+1}
\]

**2.3 Input Features**

Potential input variables include:

1. Historical menstrual onset dates.
2. Previous cycle lengths.
3. Mean and median cycle lengths.
4. Cycle-length variability.
5. Recent cycle trends.
6. Sleep duration.
7. Stress levels.
8. Lifestyle-related observations.

Contextual variables should only be included when reliable, sufficiently complete observations are available.

**2.4 Prediction Output**

The primary output is the predicted date of the next menstrual cycle onset.

The prediction can also provide an estimated uncertainty interval to communicate the possible variation in the predicted date.

### 3. Establishing Suitable Baselines

Baseline models are necessary to determine whether a proposed machine learning approach provides meaningful improvement over simpler prediction techniques.

The following baseline models are proposed.

| Baseline | Method | Purpose |
|---|---|---|
| B1 | Last Cycle Length | Uses the most recent cycle length |
| B2 | Historical Mean | Predicts using the average of previous cycle lengths |
| B3 | Historical Median | Uses the median cycle length |
| B4 | Moving Average | Uses the most recent fixed number of cycles |
| B5 | Linear Regression | Learns relationships between historical cycle features and the next cycle length |
| B6 | Random Forest | Captures nonlinear relationships in historical observations |
| B7 | ARIMA | Models temporal dependencies in cycle-length observations |
| B8 | LSTM | Learns sequential patterns from historical cycle data |

**Proposed model:** An adaptive, regime-aware machine learning framework.

The proposed framework will attempt to identify individual cycle patterns and dynamically adjust its prediction strategy when the observed cycle behaviour changes.

### 4. Designing the Evaluation Protocol

A reliable evaluation protocol is essential to ensure that the predictive performance is measured objectively and without data leakage.

**4.1 Dataset Preparation**

The dataset should contain multiple menstrual cycle observations for each participant.

The following preprocessing activities will be performed:

- Removal of duplicate observations.
- Validation of menstrual onset dates.
- Identification of missing records.
- Calculation of cycle lengths.
- Detection of inconsistent observations.
- Feature extraction and normalization where required.

Missing values should not automatically be interpreted as irregular cycles.

**4.2 Data Splitting Strategy**

A chronological data-splitting strategy will be used rather than random splitting.

The dataset will be divided into:

- Training data: Earlier observations used for model development.
- Validation data: Subsequent observations used for model selection and parameter tuning.
- Testing data: Later, unseen observations used for final performance evaluation.

A participant-level separation will additionally be conducted to assess generalization to previously unseen individuals.

**4.3 Rolling-Origin Evaluation**

A rolling-origin evaluation strategy will be used to simulate real-world prediction.

At each prediction point:

1. Historical observations are used to train or update the model.
2. The model predicts the next cycle length.
3. The actual cycle length is recorded.
4. Prediction error is calculated.
5. The observation becomes available for subsequent predictions.

This process evaluates whether the model can adapt to changing cycle patterns.

**4.4 Evaluation Metrics**

The following metrics will be considered:

**Mean Absolute Error (MAE)**

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

Measures the average absolute prediction error in days.

**Root Mean Squared Error (RMSE)**

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

Penalizes larger prediction errors more heavily.

**Median Absolute Error (MdAE)**

Measures the median absolute prediction error and reduces sensitivity to extreme errors.

**Prediction Accuracy Within Tolerance**

\[
A_k=\frac{1}{n}\sum_{i=1}^{n}
I(|y_i-\hat y_i|\leq k)\times100
\]

This measures the percentage of predictions within a specified tolerance, such as ±3 or ±5 days.

Results will be reported separately for different cycle-pattern groups wherever sufficient data are available.

**4.5 Comparative Evaluation**

All baseline models and the proposed model will be evaluated using identical test observations and prediction horizons.

The comparison will consider:

- Overall prediction error.
- Performance on regular cycles.
- Performance on variable cycles.
- Performance on irregular cycles.
- Adaptation following changes in cycle behaviour.
- Performance across different participants.

Statistical confidence intervals and suitable paired comparisons will be used to assess whether performance differences are meaningful.

### 5. Potential Methodological Research Gap

**5.1 Identified Research Gap**

Many conventional prediction approaches primarily depend on historical cycle averages or assume relatively stable temporal patterns.

However, individual menstrual cycle behaviour may change over time. A model trained on earlier observations may therefore become less suitable when the individual's cycle pattern changes.

A potential methodological gap is the limited evaluation of prediction frameworks that explicitly identify individual cycle regimes and adapt their predictive behaviour when those regimes change.

**5.2 Proposed Research Question**

Can an adaptive, regime-aware machine learning framework improve next menstrual cycle onset prediction compared with conventional statistical and machine learning models, particularly when individual cycle patterns change over time?

**5.3 Proposed Methodological Approach**

The proposed framework will investigate the following components:

1. **Cycle Regime Identification:** Identify patterns such as relatively stable, variable, and highly variable cycle behaviour using historical observations.

2. **Change Detection:** Investigate statistical or machine learning techniques to detect changes in an individual's cycle patterns.

3. **Adaptive Prediction:** Update the predictive model when new observations become available.

4. **Regime-Aware Modelling:** Incorporate identified cycle regimes into the prediction process.

5. **Performance Comparison:** Compare the proposed approach with statistical, traditional machine learning, and sequential deep learning baselines.

Potential techniques include:

- Hidden Markov Models (HMM).
- Change-point detection.
- Random Forest.
- Long Short-Term Memory (LSTM).
- State-space models.
- Online learning methods.

These techniques will be evaluated experimentally to identify the most suitable combination.

### 6. Expected Research Contributions

The proposed study is expected to contribute:

1. A systematic comparative evaluation of menstrual cycle prediction baselines.
2. An approach for identifying individual cycle-pattern changes.
3. An adaptive prediction framework incorporating temporal regime information.
4. A chronological evaluation protocol for changing cycle patterns.
5. An assessment of predictive performance across different cycle-pattern groups.

### 7. Conclusion

This assignment formulates menstrual cycle prediction as a personalized time-series forecasting problem. It establishes statistical, traditional machine learning, and deep learning baselines for comparative evaluation.

A chronological and rolling-origin evaluation protocol is proposed to measure the models' ability to predict future cycle observations without introducing temporal data leakage.

The identified methodological research opportunity is the development and evaluation of adaptive, regime-aware predictive models that respond to changes in individual cycle patterns.

The proposed work aims to investigate whether explicitly modelling temporal regime changes can improve prediction performance, particularly for individuals with variable and irregular menstrual cycles.

### 8. References for Further Reading

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
3. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357–384.
4. Page, E. S. (1954). Continuous Inspection Schemes. *Biometrika*, 41(1/2), 100–115.
5. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts.