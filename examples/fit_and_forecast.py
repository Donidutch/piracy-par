"""Fit the count models to weekly data, compare one-step forecasts, and save plots.

By default this uses the bundled SYNTHETIC sample (see piracy_par.data.sample). To use your
own weekly counts, e.g. the output of ``piracy-par-scrape weekly``::

    uv run python examples/fit_and_forecast.py --data weekly_counts.csv \
        --date-col week_end --count-col count

Seasonal covariates are derived from the dates (annual sin/cos terms; shifted to be
non-negative for the linear PARX).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from piracy_par import PAR, PARX, LogLinearPAR, LogLinearPARX, PoissonGLM
from piracy_par.data import load_sample, load_weekly_counts, seasonal_features
from piracy_par.evaluation import compare_models, train_test_split
from piracy_par.plots import plot_diagnostics, plot_forecast, plot_one_step


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="CSV with a date column and a count column")
    ap.add_argument("--date-col", default="week_end")
    ap.add_argument("--count-col", default="count")
    ap.add_argument("--test-size", type=int, default=30, help="test weeks (default 30)")
    ap.add_argument("--horizon", type=int, default=12, help="multi-step forecast horizon")
    ap.add_argument("--out", default="outputs", help="directory for plots and tables")
    args = ap.parse_args(argv)

    if args.data:
        y = load_weekly_counts(args.data, date_col=args.date_col, count_col=args.count_col)
        source = args.data
    else:
        y = load_sample()
        source = "bundled SYNTHETIC sample (not real IMB data)"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X = seasonal_features(y.index)
    X_pos = seasonal_features(y.index, nonnegative=True)
    print(f"Data: {source}")
    print(f"{len(y)} weeks, {y.index[0]:%Y-%m-%d} to {y.index[-1]:%Y-%m-%d}, "
          f"mean {y.mean():.2f}, variance {y.var():.2f}\n")

    specs = {
        "Naive mean (Poisson)": (PoissonGLM(lags=0), None),
        "Poisson GLM, 1 lag + season": (PoissonGLM(lags=1), X),
        "PAR(1,1)": (PAR(1, 1), None),
        "PARX(1,1) + season": (PARX(1, 1), X_pos),
        "Log-linear PAR": (LogLinearPAR(), None),
        "Log-linear PARX + season": (LogLinearPARX(), X),
    }
    table, preds = compare_models(specs, y, test_size=args.test_size)
    n_train = len(y) - args.test_size
    print(f"Rolling one-step forecasts, train {n_train} weeks / test {args.test_size} weeks "
          "(lower is better for every score):")
    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(table)
    table.to_csv(out / "metrics.csv", float_format="%.4f")

    for name in ("PAR(1,1)", "Log-linear PARX + season"):
        print(f"\n--- {name} (fitted on the training block) ---")
        print(specs[name][0].summary())

    y_tr, y_te, *_ = train_test_split(y, None, args.test_size)
    best = list(table.index[:3])
    fig = plot_one_step(y_tr, y_te, {k: preds[k] for k in best}, index=y.index,
                        title="One-step-ahead forecasts, three best models by log score")
    fig.savefig(out / "one_step_forecasts.png", dpi=150)

    best_name = table.index[0]
    best_model = specs[best_name][0]
    fig = plot_diagnostics(y_te.to_numpy(), preds[best_name],
                           title=f"Test-period diagnostics: {best_name}")
    fig.savefig(out / "diagnostics.png", dpi=150)

    # refit the log-linear PARX on all data and forecast beyond the sample
    final = LogLinearPARX().fit(y, X)
    future = pd.DatetimeIndex([y.index[-1] + pd.Timedelta(weeks=k)
                               for k in range(1, args.horizon + 1)])
    fc = final.forecast(args.horizon, seasonal_features(future), n_sims=4000, seed=1)
    fc.insert(0, "week_end", future.strftime("%Y-%m-%d"))
    fc.to_csv(out / "forecast.csv", index=False, float_format="%.3f")
    fig = plot_forecast(y.to_numpy(), fc, index=y.index, future_index=future,
                        title=f"{args.horizon}-week forecast, log-linear PARX (90% interval)")
    fig.savefig(out / "multi_step_forecast.png", dpi=150)
    print(f"\n{args.horizon}-week forecast from the full-sample log-linear PARX:")
    print(fc.round(2).to_string(index=False))
    print(f"\nBest model by test log score: {best_name} "
          f"({type(best_model).__name__}, {np.size(best_model.params_)} parameters)")
    print(f"Saved metrics.csv, forecast.csv and 3 plots to {out.resolve()}")


if __name__ == "__main__":
    main()
