"""
train.py
Small helpers for fitting, hyperparameter tuning, timing, and saving models
and metrics so every model notebook follows the same routine.
"""

import os
import json
import time
import joblib
from sklearn.model_selection import GridSearchCV


def fit_and_time(model, X, y):
    # Fit a model and return it together with the training time in seconds.
    t0 = time.time()
    model.fit(X, y)
    return model, time.time() - t0


def tune(estimator, param_grid, X, y, cv=3, scoring="accuracy"):
    # Grid search with cross-validation. Returns the best model, the best
    # parameters, and the total search time in seconds.
    gs = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                      n_jobs=-1, verbose=1)
    t0 = time.time()
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, time.time() - t0


def inference_time_per_sample(model, X, n=500):
    # Average prediction time per sample, useful for the cost comparison.
    Xs = X[:n]
    t0 = time.time()
    model.predict(Xs)
    return (time.time() - t0) / max(len(Xs), 1)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics, path, extra=None):
    # Save a metrics dictionary as JSON. The confusion matrix is converted to a
    # plain list. Extra fields such as training time can be added through extra.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for k, v in metrics.items():
        out[k] = v.tolist() if hasattr(v, "tolist") else v
    if extra:
        out.update(extra)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
