"""Forecast evaluation for Poisson count forecasts.

All scores are averaged over the evaluation period and negatively oriented (lower is
better). The predictive distribution at each time point is ``Poisson(lambda_t)``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from .models import BaseCountModel

__all__ = [
    "train_test_split",
    "log_score",
    "dawid_sebastiani_score",
    "ranked_probability_score",
    "mae",
    "rmse",
    "pearson_residuals",
    "randomized_pit",
    "forecast_metrics",
    "compare_models",
]


def train_test_split(y, X=None, test_size: int | float = 0.2):
    """Time-ordered split. ``test_size`` is a count (int) or a fraction (float in (0, 1))."""
    n = len(y)
    n_test = int(round(test_size * n)) if isinstance(test_size, float) else int(test_size)
    if not 0 < n_test < n:
        raise ValueError(f"test_size gives {n_test} test points for n={n}.")
    cut = n - n_test

    def _slice(a):
        if a is None:
            return None
        return a.iloc[:cut] if hasattr(a, "iloc") else np.asarray(a)[:cut]

    def _slice_te(a):
        if a is None:
            return None
        return a.iloc[cut:] if hasattr(a, "iloc") else np.asarray(a)[cut:]

    return _slice(y), _slice_te(y), _slice(X), _slice_te(X)


def _arrays(y, lam):
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    if y.shape != lam.shape:
        raise ValueError("y and lam must have the same shape.")
    return y, lam


def log_score(y, lam) -> float:
    """Mean negative Poisson log predictive density (exact, including log y!)."""
    y, lam = _arrays(y, lam)
    return float(-np.mean(stats.poisson.logpmf(y, lam)))


def dawid_sebastiani_score(y, lam) -> float:
    """Mean Dawid-Sebastiani score ``((y - mu)/sigma)^2 + 2 log sigma`` with mu = sigma^2 = lam."""
    y, lam = _arrays(y, lam)
    return float(np.mean((y - lam) ** 2 / lam + np.log(lam)))


def ranked_probability_score(y, lam, kmax: int | None = None) -> float:
    """Mean ranked probability score ``sum_k (F(k) - 1{y <= k})^2``."""
    y, lam = _arrays(y, lam)
    if kmax is None:
        kmax = int(max(y.max(), stats.poisson.ppf(1 - 1e-10, lam.max()))) + 1
    k = np.arange(kmax + 1)
    F = stats.poisson.cdf(k[None, :], lam[:, None])
    ind = (y[:, None] <= k[None, :]).astype(float)
    return float(np.mean(np.sum((F - ind) ** 2, axis=1)))


def mae(y, lam) -> float:
    y, lam = _arrays(y, lam)
    return float(np.mean(np.abs(y - lam)))


def rmse(y, lam) -> float:
    y, lam = _arrays(y, lam)
    return float(np.sqrt(np.mean((y - lam) ** 2)))


def pearson_residuals(y, lam) -> np.ndarray:
    y, lam = _arrays(y, lam)
    return (y - lam) / np.sqrt(lam)


def randomized_pit(y, lam, seed=None) -> np.ndarray:
    """Randomised PIT values; uniform on (0, 1) when the forecasts are calibrated."""
    y, lam = _arrays(y, lam)
    rng = np.random.default_rng(seed)
    lo = stats.poisson.cdf(y - 1, lam)
    hi = stats.poisson.cdf(y, lam)
    return lo + rng.uniform(size=y.shape) * (hi - lo)


def forecast_metrics(y, lam) -> dict[str, float]:
    return {
        "log_score": log_score(y, lam),
        "dss": dawid_sebastiani_score(y, lam),
        "rps": ranked_probability_score(y, lam),
        "mae": mae(y, lam),
        "rmse": rmse(y, lam),
    }


def compare_models(
    specs: Mapping[str, tuple[BaseCountModel, object]],
    y,
    test_size: int | float = 0.2,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit each model on the training block and score rolling one-step forecasts on the test block.

    ``specs`` maps a display name to ``(model, X)`` where ``X`` is the full-length covariate
    matrix for that model (or ``None``). Returns a metrics table sorted by log score and a
    dict of test-period intensities.

    Every model is scored on the same test observations. In-sample log-likelihoods and AIC
    are deliberately left out: each model conditions on a different number of initial
    observations (``max(p, q)``, 1 or ``lags``), so they are sums over different samples
    and not comparable across models.
    """
    rows, preds = [], {}
    for name, (model, X) in specs.items():
        y_tr, y_te, X_tr, X_te = train_test_split(y, X, test_size)
        model.fit(y_tr, X_tr)
        lam = model.predict(y_te, X_te)
        preds[name] = lam
        rows.append(
            {
                "model": name,
                "n_params": model.params_.size,
                **forecast_metrics(y_te, lam),
            }
        )
    table = pd.DataFrame(rows).set_index("model").sort_values("log_score")
    return table, preds
