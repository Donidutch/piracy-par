# piracy-par

Poisson autoregressions for weekly counts of maritime piracy incidents, plus a scraper for
the ICC-CCS / IMB live piracy map.

> Draft README; the results section and figures are filled in at the documentation step.

## What is here

| Module | Contents |
|---|---|
| `piracy_par.models` | `PAR(p, q)` / `PARX` (linear INGARCH), `LogLinearPAR` / `LogLinearPARX` (Fokianos & Tjøstheim 2011), `PoissonGLM` baseline |
| `piracy_par.evaluation` | log score, Dawid-Sebastiani, ranked probability score, MAE/RMSE, randomised PIT, time-ordered split, model comparison |
| `piracy_par.plots` | one-step forecast, multi-step fan and residual diagnostic figures (matplotlib `Figure`, headless) |
| `piracy_par.data.imb_parser` | browser-free parser for map entries, place repair, incident/vessel type heuristics |
| `piracy_par.data.imb_scraper` | Selenium fetcher and the `piracy-par-scrape` CLI (`fetch`, `parse`, `weekly`) |
| `piracy_par.data.sample` | synthetic weekly sample generator and the bundled `synthetic_weekly_counts.csv` |

## Models

All models assume `y_t | F_{t-1} ~ Poisson(lambda_t)`.

- **PAR(p, q) / PARX**: `lambda_t = omega + sum_i alpha_i y_{t-i} + sum_j beta_j lambda_{t-j} + gamma' x_t`,
  with `omega > 0`, `alpha, beta, gamma >= 0`, non-negative `x_t` and `sum(alpha) + sum(beta) < 1`.
- **Log-linear PAR(X)**: `log lambda_t = d + a log lambda_{t-1} + b log(1 + y_{t-1}) + c' x_t`,
  constrained to `|a| + |b| < 1` (a conservative sufficient condition for stability; negative
  coefficients are allowed). Starting values come from a Poisson GLM of `y_t` on
  `[1, log(1 + y_{t-1}), x_t]`.
- **PoissonGLM(lags)**: `log lambda_t = const + sum_k phi_k log(1 + y_{t-k}) + beta' x_t`;
  `lags=0` without covariates is the naive constant-mean forecast.

Estimation is exact conditional maximum likelihood (the `log y!` term included), conditioning
on the first `max(p, q)` (linear), 1 (log-linear) or `lags` (GLM) observations with pre-sample
intensities set to the training mean. The linear and log-linear models use SLSQP with analytic
gradients from the score recursion; standard errors come from a numerical Hessian of the
analytic gradient and are not valid for parameters on a constraint boundary.

Every model has `fit(y, X)`, `loglik(params)`, `score(params)`, `fitted_values()`,
`predict(y_new, X_new)` (rolling one-step), `forecast(h, X_future)` (exact mean for the
linear model, simulated otherwise, plus interval), `simulate(n)`, `params`, `coef_table()`
and `summary()`.

## Data

IMB narrations and coordinates are copyrighted by ICC Commercial Crime Services and the site
grants no redistribution licence, so **no scraped data is included**. The bundled
`synthetic_weekly_counts.csv` is simulated from a log-linear PARX (156 weeks, mean about 2.3)
and is labelled as synthetic in the file itself.

To build a real series yourself (check the ICC-CCS terms first; the scraper waits at least
2 s between pages and identifies itself with its own user agent):

```bash
uv sync --extra scrape            # Selenium; needs Chrome/Chromium
uv run piracy-par-scrape fetch --years 2023 2024 2025 --out scrape_out
uv run piracy-par-scrape weekly --incidents scrape_out/all_incidents.csv --out weekly_counts.csv
uv run python examples/fit_and_forecast.py --data weekly_counts.csv
```

Weeks run Tuesday to Monday and are labelled by the closing Monday (`W-MON`).

## Quick start

```bash
uv sync
uv run pytest
uv run python examples/fit_and_forecast.py --out outputs
```

## Limitations

- Results on the bundled data are on synthetic counts; results on real data are not
  reproducible from this repository.
- The scraper's table ids and page layout were observed in 2025 and may change.
- Incident and vessel types are keyword heuristics on the narration text.

## References

- Ferland, Latour & Oraichi (2006). Integer-valued GARCH process. *J. Time Series Analysis*.
- Fokianos, Rahbek & Tjøstheim (2009). Poisson autoregression. *JASA*.
- Fokianos & Tjøstheim (2011). Log-linear Poisson autoregression. *J. Multivariate Analysis*.
- Agosto, Cavaliere, Kristensen & Rahbek (2016). Modeling corporate defaults: Poisson
  autoregressions with exogenous covariates (PARX). *J. Empirical Finance*.
