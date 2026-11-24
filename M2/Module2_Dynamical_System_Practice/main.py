import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

# 1. Simulate a continuous-time logistic dynamical system
r_true, K_true, x0, noise_sd = 0.20, 100.0, 10.0, 1.5
time = np.arange(0, 101, 1)

def logistic_rhs(t, x, r, K):
    return r * x * (1 - x / K)

sol = solve_ivp(logistic_rhs, (time[0], time[-1]), [x0],
                args=(r_true, K_true), t_eval=time,
                rtol=1e-9, atol=1e-11)
true_x = sol.y[0]
observed_x = np.maximum(true_x + np.random.normal(0, noise_sd, len(time)), 0.01)

df = pd.DataFrame({"time": time, "true_state": true_x,
                   "observed_state": observed_x})
df.to_csv("outputs/simulated_data.csv", index=False)

# 2. Chronological train/test split
split = int(0.70 * len(time))
train_t, test_t = time[:split], time[split:]
train_y, test_y = observed_x[:split], observed_x[split:]

# 3. Identify logistic parameters from training observations
# The logistic closed-form solution is fitted using elapsed time from the first observation.
def logistic_curve(t, r, K):
    initial = train_y[0]
    return K / (1 + ((K - initial) / initial) * np.exp(-r * t))

params, _ = curve_fit(logistic_curve, train_t - train_t[0], train_y,
                      p0=[0.15, 90.0], bounds=([0.001, 1.0], [2.0, 1000.0]),
                      maxfev=20000)
r_hat, K_hat = params

# Forecast from the final observed training state, over the test horizon
forecast_t = np.arange(0, len(test_t), dtype=float)
x_start = train_y[-1]
def forecast_curve(elapsed, r, K):
    return K / (1 + ((K - x_start) / x_start) * np.exp(-r * elapsed))
dyn_forecast = forecast_curve(forecast_t, r_hat, K_hat)

# 4. Random Forest one-step model with five lagged observations; recursive test forecast
lags = 5
def make_lagged(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.asarray(X), np.asarray(y)

X_train, y_train = make_lagged(train_y, lags)
rf = RandomForestRegressor(n_estimators=300, max_depth=8,
                           min_samples_leaf=2, random_state=SEED)
rf.fit(X_train, y_train)

history = list(train_y)
rf_forecast = []
for _ in range(len(test_t)):
    pred = rf.predict(np.asarray(history[-lags:]).reshape(1, -1))[0]
    rf_forecast.append(pred)
    history.append(pred)
rf_forecast = np.asarray(rf_forecast)

# 5. Evaluation
def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

results = pd.DataFrame({
    "Identified logistic system": metrics(test_y, dyn_forecast),
    "Random Forest": metrics(test_y, rf_forecast)
}).T
results.to_csv("outputs/evaluation_metrics.csv")

# 6. Equilibrium and local stability
# f(x) = r*x*(1-x/K); f'(x) = r*(1-2*x/K)
equilibria = [0.0, K_hat]
stability_rows = []
for x_star in equilibria:
    eigenvalue = r_hat * (1 - 2*x_star/K_hat)
    status = "Unstable" if eigenvalue > 0 else "Asymptotically stable" if eigenvalue < 0 else "Inconclusive"
    stability_rows.append({"equilibrium": x_star, "jacobian_eigenvalue": eigenvalue, "local_stability": status})
pd.DataFrame(stability_rows).to_csv("outputs/stability_analysis.csv", index=False)

# 7. Forecast plot
plt.figure(figsize=(10, 5.5))
plt.plot(time, true_x, label="True simulated system", linewidth=2)
plt.scatter(time, observed_x, label="Noisy observations", s=13)
plt.axvline(train_t[-1], linestyle="--", label="Train/test boundary")
plt.plot(test_t, dyn_forecast, label="Identified logistic forecast", linewidth=2)
plt.plot(test_t, rf_forecast, label="Random Forest forecast", linewidth=2)
plt.xlabel("Time step")
plt.ylabel("System state")
plt.title("Forecast comparison")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("outputs/forecast_comparison.png", dpi=200)
plt.close()

# 8. Stability plot
x_grid = np.linspace(0, K_hat*1.35, 300)
dx = logistic_rhs(0, x_grid, r_hat, K_hat)
plt.figure(figsize=(8, 5))
plt.plot(x_grid, dx, label="dx/dt")
plt.axhline(0, linewidth=0.8)
for x_star in equilibria:
    plt.axvline(x_star, linestyle="--", alpha=0.7)
plt.xlabel("State x")
plt.ylabel("Rate of change dx/dt")
plt.title("Logistic system: equilibrium and stability")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("outputs/stability_analysis.png", dpi=200)
plt.close()

print(f"Estimated r: {r_hat:.4f}")
print(f"Estimated K: {K_hat:.4f}")
print(results.round(4).to_string())
