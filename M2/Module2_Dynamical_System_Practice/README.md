# Module 2 Practical: Dynamical System Identification

## Aim
Estimate a dynamical system from simulated observations, examine its local stability, and compare its forecasts with a conventional machine-learning model.

## Models
- Dynamical model: continuous-time logistic growth system.
- Conventional ML model: Random Forest regression using five lagged observations.

## Run
```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python main.py
```

The script writes the simulated dataset, forecast metrics, stability results and plots to `outputs/`.

## Reproducibility
The random seed is fixed at 42. The data are simulated; they are not measurements from a real population.
