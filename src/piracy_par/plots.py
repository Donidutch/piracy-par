"""Forecast and diagnostic plots.

Functions build :class:`matplotlib.figure.Figure` objects directly (no ``pyplot`` state),
so they work headless under any backend. Save with ``fig.savefig(path)``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .evaluation import pearson_residuals, randomized_pit

__all__ = ["plot_one_step", "plot_forecast", "plot_diagnostics"]

# Validated categorical order (blue, orange, aqua, yellow); observed data in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def _index(n, index):
    return np.arange(n) if index is None else pd.Index(index)


def plot_one_step(
    y_train,
    y_test,
    predictions: Mapping[str, np.ndarray],
    *,
    index=None,
    title: str = "One-step-ahead forecasts on the test period",
    max_models: int = 3,
) -> Figure:
    """Observed counts with each model's rolling one-step intensities on the test block."""
    y_all = np.r_[np.asarray(y_train, float), np.asarray(y_test, float)]
    x = _index(len(y_all), index)
    cut = len(y_train)
    fig = Figure(figsize=(10, 4.2), facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot()
    _style(ax)
    ax.plot(x, y_all, color=INK_2, linewidth=1.0, marker="o", markersize=2.5, label="observed")
    ax.axvline(x[cut], color=INK_2, linewidth=1, linestyle="--")
    ax.text(x[cut], ax.get_ylim()[1], " test", color=INK_2, va="top", fontsize=9)
    for i, (name, lam) in enumerate(list(predictions.items())[:max_models]):
        ax.plot(x[cut:], lam, color=SERIES[i], linewidth=2, label=name)
    ax.set_ylabel("count per week", color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return fig


def plot_forecast(
    y,
    forecast: pd.DataFrame,
    *,
    index=None,
    future_index=None,
    history: int = 104,
    title: str = "Multi-step forecast",
) -> Figure:
    """Recent history plus the forecast mean and simulated prediction interval."""
    y = np.asarray(y, float)
    x = _index(len(y), index)
    h = len(forecast)
    if future_index is None:
        if isinstance(x, pd.DatetimeIndex) and len(x) > 1:
            step = x[-1] - x[-2]
            future_index = pd.DatetimeIndex([x[-1] + step * k for k in range(1, h + 1)])
        else:
            future_index = np.arange(len(y), len(y) + h)
    start = max(0, len(y) - history)
    fig = Figure(figsize=(10, 4.2), facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot()
    _style(ax)
    ax.plot(
        x[start:],
        y[start:],
        color=INK_2,
        linewidth=1.0,
        marker="o",
        markersize=2.5,
        label="observed",
    )
    ax.fill_between(
        future_index,
        forecast["lower"],
        forecast["upper"],
        color=SERIES[0],
        alpha=0.18,
        linewidth=0,
        label="prediction interval",
    )
    ax.plot(future_index, forecast["mean"], color=SERIES[0], linewidth=2, label="forecast mean")
    ax.set_ylabel("count per week", color=INK)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return fig


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    denom = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[: len(x) - k]) / denom for k in range(nlags + 1)])


def plot_diagnostics(
    y, lam, *, title: str = "Residual diagnostics", nlags: int = 20, seed=0
) -> Figure:
    """Pearson residuals over time, their ACF, and a randomised PIT histogram."""
    r = pearson_residuals(y, lam)
    pit = randomized_pit(y, lam, seed=seed)
    nlags = min(nlags, len(r) - 1)
    fig = Figure(figsize=(11, 3.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3)
    for ax in axes:
        _style(ax)
    axes[0].plot(r, color=SERIES[0], linewidth=1.2)
    axes[0].axhline(0, color=INK_2, linewidth=1)
    axes[0].set_title("Pearson residuals", color=INK, loc="left", fontsize=10)
    acf = _acf(r, nlags)
    band = 1.96 / np.sqrt(len(r))
    axes[1].vlines(np.arange(1, nlags + 1), 0, acf[1:], color=SERIES[0], linewidth=2)
    axes[1].axhspan(-band, band, color=GRID, alpha=0.7, linewidth=0)
    axes[1].axhline(0, color=INK_2, linewidth=1)
    axes[1].set_title("ACF of Pearson residuals", color=INK, loc="left", fontsize=10)
    axes[2].hist(pit, bins=10, range=(0, 1), color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    axes[2].axhline(len(pit) / 10, color=INK_2, linewidth=1, linestyle="--")
    axes[2].set_title("Randomised PIT", color=INK, loc="left", fontsize=10)
    fig.suptitle(title, color=INK, fontsize=11)
    return fig
