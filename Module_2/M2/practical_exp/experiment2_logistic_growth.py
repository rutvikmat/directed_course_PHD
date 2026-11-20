
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================================

TRUE_R = 0.35
TRUE_K = 1000.0
INITIAL_POPULATION = 50.0

DT = 0.1
T_END = 30

time = np.arange(0, T_END + DT, DT)

rng = np.random.default_rng(42)


# ==========================================================
# 2. LOGISTIC DYNAMICAL SYSTEM
# ==========================================================

def logistic_system(t, P, r, K):

    population = P[0]

    growth = r * population * (
        1 - population / K
    )

    return [growth]


# ==========================================================
# 3. SIMULATE POPULATION GROWTH
# ==========================================================

solution = solve_ivp(
    logistic_system,
    [time[0], time[-1]],
    [INITIAL_POPULATION],
    args=(TRUE_R, TRUE_K),
    t_eval=time,
    rtol=1e-9,
    atol=1e-11
)

true_population = solution.y[0]

# Add measurement noise
observed_population = (
    true_population
    + rng.normal(0, 5, len(time))
)

# Population observations must be non-negative
observed_population = np.maximum(
    observed_population, 0
)


# ==========================================================
# 4. CREATE DATASET
# ==========================================================

data = pd.DataFrame({
    "time": time,
    "true_population": true_population,
    "observed_population": observed_population
})

data.to_csv(
    "logistic_population_data.csv",
    index=False
)

print("\nDATASET GENERATED")
print(data.head())


# ==========================================================
# 5. TRAIN-TEST SPLIT
# ==========================================================

split = int(0.70 * len(time))

train_time = time[:split]
test_time = time[split:]

train_data = observed_population[:split]
test_data = observed_population[split:]


# ==========================================================
# 6. PARAMETER ESTIMATION
# Estimate r and K
# ==========================================================

def residuals(parameters):

    r, K = parameters

    predicted_solution = solve_ivp(
        logistic_system,
        [time[0], time[-1]],
        [INITIAL_POPULATION],
        args=(r, K),
        t_eval=train_time,
        rtol=1e-7,
        atol=1e-9
    )

    return (
        predicted_solution.y[0] - train_data
    )


estimate = least_squares(
    residuals,
    x0=[0.25, 900],
    bounds=([0.001, 100], [2.0, 5000])
)

estimated_r, estimated_K = estimate.x

print("\nPARAMETER ESTIMATION")
print("--------------------")
print("Actual growth rate:", TRUE_R)
print("Estimated growth rate:", estimated_r)

print("Actual carrying capacity:", TRUE_K)
print("Estimated carrying capacity:", estimated_K)


# ==========================================================
# 7. EQUILIBRIUM AND STABILITY ANALYSIS
# ==========================================================

equilibrium_points = [0, estimated_K]

def derivative(P):
    return estimated_r * (
        1 - 2 * P / estimated_K
    )

print("\nEQUILIBRIUM STABILITY")
print("---------------------")

for equilibrium in equilibrium_points:

    eigenvalue = derivative(equilibrium)

    if eigenvalue < 0:
        stability = "ASYMPTOTICALLY STABLE"

    elif eigenvalue > 0:
        stability = "UNSTABLE"

    else:
        stability = "INCONCLUSIVE FROM LINEARIZATION"

    print(
        f"Equilibrium = {equilibrium:.2f}, "
        f"Derivative = {eigenvalue:.4f}, "
        f"Stability = {stability}"
    )


# ==========================================================
# 8. DYNAMICAL MODEL FORECAST
# ==========================================================

dynamic_solution = solve_ivp(
    logistic_system,
    [time[0], time[-1]],
    [INITIAL_POPULATION],
    args=(estimated_r, estimated_K),
    t_eval=test_time,
    rtol=1e-7,
    atol=1e-9
)

dynamic_forecast = dynamic_solution.y[0]


# ==========================================================
# 9. RANDOM FOREST MODEL
# Recursive multi-step forecasting
# ==========================================================

X = []
y = []

for i in range(2, len(time)):

    X.append([
        observed_population[i - 1],
        observed_population[i - 2]
    ])

    y.append(observed_population[i])

X = np.array(X)
y = np.array(y)

ml_split = split - 2

X_train = X[:ml_split]
y_train = y[:ml_split]

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)

# Recursive forecasting
ml_forecast = []

previous_1 = observed_population[split - 1]
previous_2 = observed_population[split - 2]

for i in range(split, len(time)):

    features = np.array([[
        previous_1,
        previous_2
    ]])

    prediction = rf.predict(features)[0]

    ml_forecast.append(prediction)

    previous_2 = previous_1
    previous_1 = prediction

ml_forecast = np.array(ml_forecast)


# ==========================================================
# 10. MODEL EVALUATION
# ==========================================================

def evaluate(actual, predicted):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    return mae, rmse


dynamic_mae, dynamic_rmse = evaluate(
    test_data, dynamic_forecast
)

ml_mae, ml_rmse = evaluate(
    test_data, ml_forecast
)

results = pd.DataFrame({
    "Model": [
        "Logistic Dynamical System",
        "Random Forest"
    ],

    "MAE": [
        dynamic_mae,
        ml_mae
    ],

    "RMSE": [
        dynamic_rmse,
        ml_rmse
    ]
})

print("\nFORECAST COMPARISON")
print(results.to_string(index=False))


# ==========================================================
# 11. VISUALIZATION
# ==========================================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 14)
)

# Population growth
axes[0].plot(
    time,
    true_population,
    label="True population"
)

axes[0].scatter(
    time,
    observed_population,
    s=8,
    alpha=0.4,
    label="Noisy observations"
)

axes[0].axhline(
    TRUE_K,
    linestyle="--",
    label="True carrying capacity"
)

axes[0].set_title(
    "Logistic Population Growth"
)

axes[0].set_xlabel("Time (years)")
axes[0].set_ylabel("Population")
axes[0].legend()
axes[0].grid(True)


# Forecast comparison
axes[1].plot(
    test_time,
    test_data,
    label="Observed test data"
)

axes[1].plot(
    test_time,
    dynamic_forecast,
    label="Logistic dynamical forecast"
)

axes[1].plot(
    test_time,
    ml_forecast,
    label="Random Forest forecast"
)

axes[1].set_title(
    "Forecast Comparison"
)

axes[1].set_xlabel("Time (years)")
axes[1].set_ylabel("Population")
axes[1].legend()
axes[1].grid(True)


# Equilibrium stability
P_values = np.linspace(
    0, estimated_K * 1.5, 300
)

growth_rates = (
    estimated_r
    * P_values
    * (1 - P_values / estimated_K)
)

axes[2].plot(
    P_values,
    growth_rates,
    label="Population growth rate"
)

axes[2].axhline(0, linestyle="--")

axes[2].scatter(
    [0, estimated_K],
    [0, 0],
    s=80,
    label="Equilibrium points"
)

axes[2].set_title(
    "Equilibrium and Stability Analysis"
)

axes[2].set_xlabel("Population")
axes[2].set_ylabel("Growth rate")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

plt.savefig(
    "experiment2_results.png",
    dpi=300
)

plt.show()

print("\nEXPERIMENT 2 COMPLETED SUCCESSFULLY")
 