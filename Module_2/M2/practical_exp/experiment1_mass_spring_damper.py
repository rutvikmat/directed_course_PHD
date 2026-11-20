
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================================
# 1. SYSTEM CONFIGURATION
# ==========================================================

MASS = 1.0
TRUE_C = 0.8
TRUE_K = 4.0

DT = 0.02
T_END = 20.0

time = np.arange(0, T_END, DT)

rng = np.random.default_rng(42)


# ==========================================================
# 2. EXTERNAL FORCE
# ==========================================================

def external_force(t):
    return (
        2.0 * np.sin(1.5 * t)
        + 0.5 * np.sin(3.0 * t)
    )


force = external_force(time)


# ==========================================================
# 3. DYNAMICAL SYSTEM
# ==========================================================

def system(t, state, c, k):

    displacement, velocity = state

    acceleration = (
        external_force(t)
        - c * velocity
        - k * displacement
    ) / MASS

    return [velocity, acceleration]


# ==========================================================
# 4. GENERATE SIMULATED OBSERVATIONS
# ==========================================================

true_solution = solve_ivp(
    lambda t, y: system(t, y, TRUE_C, TRUE_K),
    [time[0], time[-1]],
    [0.0, 0.0],
    t_eval=time,
    rtol=1e-9,
    atol=1e-11
)

true_displacement = true_solution.y[0]
true_velocity = true_solution.y[1]

# Add measurement noise
observed_displacement = (
    true_displacement
    + rng.normal(0, 0.02, len(time))
)


# ==========================================================
# 5. CREATE DATASET
# ==========================================================

data = pd.DataFrame({
    "time": time,
    "force": force,
    "true_displacement": true_displacement,
    "observed_displacement": observed_displacement,
    "true_velocity": true_velocity
})

data.to_csv("mass_spring_data.csv", index=False)

print("\nDataset generated successfully!")
print(data.head())


# ==========================================================
# 6. PARAMETER ESTIMATION
# Estimate damping coefficient and spring stiffness
# ==========================================================

split = int(0.70 * len(time))

train_time = time[:split]
train_observations = observed_displacement[:split]


def estimation_residual(parameters):

    c, k = parameters

    solution = solve_ivp(
        lambda t, y: system(t, y, c, k),
        [time[0], time[-1]],
        [0.0, 0.0],
        t_eval=train_time,
        rtol=1e-7,
        atol=1e-9
    )

    return (
        solution.y[0] - train_observations
    )


result = least_squares(
    estimation_residual,
    x0=[1.0, 3.0],
    bounds=([0.01, 0.01], [10.0, 20.0])
)

estimated_c, estimated_k = result.x

print("\nPARAMETER ESTIMATION")
print("--------------------")
print("True damping coefficient:", TRUE_C)
print("Estimated damping coefficient:", estimated_c)
print("True spring stiffness:", TRUE_K)
print("Estimated spring stiffness:", estimated_k)


# ==========================================================
# 7. STABILITY ANALYSIS
# ==========================================================

A = np.array([
    [0.0, 1.0],
    [-estimated_k / MASS, -estimated_c / MASS]
])

eigenvalues = np.linalg.eigvals(A)

print("\nSTABILITY ANALYSIS")
print("------------------")
print("System matrix A:\n", A)
print("Eigenvalues:", eigenvalues)

if np.all(np.real(eigenvalues) < 0):
    print("System is ASYMPTOTICALLY STABLE")
elif np.any(np.real(eigenvalues) > 0):
    print("System is UNSTABLE")
else:
    print("Further stability analysis is required")


# ==========================================================
# 8. DYNAMICAL MODEL FORECAST
# Forecast the held-out period using estimated parameters
# ==========================================================

initial_displacement = observed_displacement[split - 1]

initial_velocity = (
    observed_displacement[split - 1]
    - observed_displacement[split - 2]
) / DT

forecast_time = time[split:]

dynamic_solution = solve_ivp(
    lambda t, y: system(
        t, y, estimated_c, estimated_k
    ),
    [time[split - 1], time[-1]],
    [initial_displacement, initial_velocity],
    t_eval=forecast_time,
    rtol=1e-7,
    atol=1e-9
)

dynamic_forecast = dynamic_solution.y[0]


# ==========================================================
# 9. MACHINE LEARNING MODEL
# Random Forest one-step-ahead prediction
# ==========================================================

X = []
y = []

for i in range(2, len(time)):

    X.append([
        observed_displacement[i - 1],
        observed_displacement[i - 2],
        force[i],
        force[i - 1]
    ])

    y.append(observed_displacement[i])

X = np.array(X)
y = np.array(y)

# Align indices with original time array
ml_split = split - 2

X_train = X[:ml_split]
X_test = X[ml_split:]

y_train = y[:ml_split]
y_test = y[ml_split:]

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

ml_forecast = rf.predict(X_test)


# ==========================================================
# 10. EVALUATION
# ==========================================================

actual = observed_displacement[split:]

dynamic_mae = mean_absolute_error(
    actual, dynamic_forecast
)

dynamic_rmse = np.sqrt(
    mean_squared_error(actual, dynamic_forecast)
)

ml_mae = mean_absolute_error(
    y_test, ml_forecast
)

ml_rmse = np.sqrt(
    mean_squared_error(y_test, ml_forecast)
)

results = pd.DataFrame({
    "Model": [
        "Estimated Dynamical System",
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

print("\nMODEL PERFORMANCE COMPARISON")
print(results.to_string(index=False))


# ==========================================================
# 11. VISUALIZATION
# ==========================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Original system
axes[0].plot(
    time, true_displacement,
    label="True displacement"
)

axes[0].scatter(
    time, observed_displacement,
    s=5, alpha=0.3,
    label="Noisy observations"
)

axes[0].set_title("Simulated Mass-Spring-Damper System")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Displacement")
axes[0].legend()
axes[0].grid(True)

# Forecast comparison
axes[1].plot(
    forecast_time, actual,
    label="Observed test data"
)

axes[1].plot(
    forecast_time, dynamic_forecast,
    label="Dynamical system forecast"
)

axes[1].plot(
    forecast_time, ml_forecast,
    label="Random Forest forecast",
    alpha=0.8
)

axes[1].set_title("Forecast Comparison")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Displacement")
axes[1].legend()
axes[1].grid(True)

# Eigenvalue plot
axes[2].scatter(
    np.real(eigenvalues),
    np.imag(eigenvalues),
    s=100
)

axes[2].axvline(0, linestyle="--")
axes[2].axhline(0, linestyle="--")

axes[2].set_title("Eigenvalue Stability Analysis")
axes[2].set_xlabel("Real Part")
axes[2].set_ylabel("Imaginary Part")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("experiment1_results.png", dpi=300)
plt.show()

print("\nExperiment completed successfully!")
 