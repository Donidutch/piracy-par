import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import stats

from piracy_par import PAR, LogLinearPAR, PoissonGLM
from piracy_par.evaluation import (
    compare_models,
    dawid_sebastiani_score,
    forecast_metrics,
    log_score,
    randomized_pit,
    ranked_probability_score,
    train_test_split,
)
from piracy_par.plots import plot_diagnostics, plot_forecast, plot_one_step

Y = np.array([0, 1, 3, 2, 5, 0, 1])
LAM = np.array([0.5, 1.2, 2.0, 2.5, 3.0, 0.8, 1.1])


def test_log_score_matches_scipy():
    assert log_score(Y, LAM) == pytest.approx(-stats.poisson.logpmf(Y, LAM).mean())


def test_dss():
    expected = np.mean((Y - LAM) ** 2 / LAM + np.log(LAM))
    assert dawid_sebastiani_score(Y, LAM) == pytest.approx(expected)


def test_rps_brute_force():
    total = 0.0
    for y, lam in zip(Y, LAM, strict=True):
        total += sum((stats.poisson.cdf(k, lam) - (y <= k)) ** 2 for k in range(200))
    assert ranked_probability_score(Y, LAM) == pytest.approx(total / len(Y), rel=1e-8)


def test_pit_in_unit_interval():
    pit = randomized_pit(Y, LAM, seed=0)
    assert np.all((pit >= 0) & (pit <= 1))


def test_metrics_keys():
    assert set(forecast_metrics(Y, LAM)) == {"log_score", "dss", "rps", "mae", "rmse"}


def test_train_test_split_is_time_ordered():
    y = pd.Series(np.arange(10))
    tr, te, Xtr, Xte = train_test_split(y, None, test_size=3)
    assert list(tr) == list(range(7)) and list(te) == [7, 8, 9]
    assert Xtr is None and Xte is None
    tr, te, *_ = train_test_split(np.arange(10), test_size=0.2)
    assert len(te) == 2
    with pytest.raises(ValueError):
        train_test_split(np.arange(10), test_size=10)


def test_compare_models_and_plots(tmp_path):
    y = PAR(1, 1).simulate(160, params=[0.8, 0.3, 0.4], seed=4)
    specs = {
        "naive mean": (PoissonGLM(), None),
        "PAR(1,1)": (PAR(1, 1), None),
        "log-linear": (LogLinearPAR(), None),
    }
    table, preds = compare_models(specs, y, test_size=30)
    assert set(table.index) == set(specs)
    assert table["log_score"].is_monotonic_increasing
    assert all(len(v) == 30 for v in preds.values())

    figs = [
        plot_one_step(y[:130], y[130:], preds),
        plot_forecast(y, specs["PAR(1,1)"][0].forecast(10, n_sims=200, seed=0),
                      index=pd.date_range("2023-01-02", periods=160, freq="W-MON")),
        plot_diagnostics(y[130:], preds["PAR(1,1)"]),
    ]
    for i, fig in enumerate(figs):
        assert isinstance(fig, Figure)
        out = tmp_path / f"fig{i}.png"
        fig.savefig(out)
        assert out.stat().st_size > 1000
