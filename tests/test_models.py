import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from piracy_par import PAR, PARX, LogLinearPAR, LogLinearPARX, PoissonGLM
from piracy_par.data.weekly import seasonal_features


def _exog(n, nonnegative=False):
    return seasonal_features(
        pd.date_range("2010-01-04", periods=n, freq="W-MON"), nonnegative=nonnegative
    )


# --------------------------------------------------------------------------------------
# independent reference recursions (plain loops, written separately from the package)
# --------------------------------------------------------------------------------------
def ref_linear(theta, y, X, p, q):
    m = max(p, q)
    k = 0 if X is None else X.shape[1]
    omega, alpha, beta = theta[0], theta[1 : 1 + p], theta[1 + p : 1 + p + q]
    gamma = theta[1 + p + q : 1 + p + q + k]
    lam = np.full(len(y), y.mean())
    for t in range(m, len(y)):
        v = omega
        for i in range(1, p + 1):
            v += alpha[i - 1] * y[t - i]
        for j in range(1, q + 1):
            v += beta[j - 1] * lam[t - j]
        if X is not None:
            v += X[t] @ gamma
        lam[t] = v
    return lam, m


def ref_loglinear(theta, y, X):
    d, a, b = theta[:3]
    nu = np.empty(len(y))
    nu[0] = np.log(y.mean())
    for t in range(1, len(y)):
        nu[t] = d + a * nu[t - 1] + b * np.log1p(y[t - 1])
        if X is not None:
            nu[t] += X[t] @ theta[3:]
    return np.exp(nu), 1


def ref_glm(theta, y, X, lags):
    lam = np.full(len(y), y.mean())
    for t in range(lags, len(y)):
        eta = theta[0] + sum(theta[k] * np.log1p(y[t - k]) for k in range(1, lags + 1))
        if X is not None:
            eta += X[t] @ theta[1 + lags :]
        lam[t] = np.exp(eta)
    return lam, lags


CASES = [
    (
        "par11",
        lambda: PAR(1, 1),
        [0.8, 0.3, 0.4],
        None,
        lambda th, y, X: ref_linear(th, y, X, 1, 1),
    ),
    (
        "par21x",
        lambda: PARX(2, 1),
        [0.5, 0.2, 0.1, 0.3, 0.4, 0.2, 0.1, 0.3],
        "nonneg",
        lambda th, y, X: ref_linear(th, y, X, 2, 1),
    ),
    (
        "par12",
        lambda: PAR(1, 2),
        [0.6, 0.25, 0.3, 0.2],
        None,
        lambda th, y, X: ref_linear(th, y, X, 1, 2),
    ),
    ("loglin", lambda: LogLinearPAR(), [0.3, 0.4, 0.3], None, ref_loglinear),
    ("loglinx", lambda: LogLinearPARX(), [0.2, 0.3, 0.35, 0.2, -0.1], "signed", ref_loglinear),
    (
        "glm2x",
        lambda: PoissonGLM(lags=2),
        [0.3, 0.3, 0.1, 0.2, -0.1],
        "signed",
        lambda th, y, X: ref_glm(th, y, X, 2),
    ),
]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    y = rng.poisson(2.5, size=200).astype(float)
    return y, _exog(200).to_numpy(), _exog(200, nonnegative=True).to_numpy()


def _pick_X(kind, data):
    return {None: None, "signed": data[1], "nonneg": data[2]}[kind]


@pytest.mark.parametrize("name,make,theta,xkind,ref", CASES, ids=[c[0] for c in CASES])
def test_loglik_matches_scipy(name, make, theta, xkind, ref, data):
    y, X = data[0], _pick_X(xkind, data)
    model = make().fit(y, X)
    theta = np.asarray(theta, float)
    lam, m = ref(theta, y, X)
    expected = stats.poisson.logpmf(y[m:], lam[m:]).sum()
    assert model.loglik(theta) == pytest.approx(expected, rel=1e-10, abs=1e-8)
    # fitted intensities agree with the reference recursion at the MLE as well
    lam_hat, _ = ref(model.params_, y, X)
    np.testing.assert_allclose(model.fitted_values(), lam_hat, rtol=1e-9)


@pytest.mark.parametrize("name,make,theta,xkind,ref", CASES, ids=[c[0] for c in CASES])
def test_analytic_gradient(name, make, theta, xkind, ref, data):
    y, X = data[0], _pick_X(xkind, data)
    model = make().fit(y, X)
    theta = np.asarray(theta, float)
    num = optimize.approx_fprime(theta, model.loglik, 1e-6)
    np.testing.assert_allclose(model.score(theta), num, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("make", [lambda: PAR(1, 1), lambda: LogLinearPAR()])
def test_mle_satisfies_kkt(make, data):
    """Score ~ 0 for free parameters; parameters on a bound or an active constraint are
    flagged. (On these i.i.d. data the log-linear fit runs to a = 0.9999: with the
    pre-sample intensity fixed at the sample mean, a -> 1 mimics a constant intensity.)"""
    model = make().fit(data[0])
    free = ~model.coef_table()["at_bound"].to_numpy()
    assert free.any()
    assert np.max(np.abs(model.score()[free])) < 1e-2


def test_par_parameter_recovery():
    true = np.array([0.6, 0.35, 0.45])
    y = PAR(1, 1).simulate(3000, params=true, seed=7)
    m = PAR(1, 1).fit(y)
    assert m.converged_
    err = np.abs(m.params_ - true)
    assert np.all(err < 4 * m.bse_), (m.params_, m.bse_)
    assert np.all(err < np.array([0.3, 0.06, 0.1]))


def test_parx_parameter_recovery():
    n = 3000
    X = _exog(n, nonnegative=True)
    true = np.array([0.4, 0.3, 0.4, 0.3, 0.15, 0.2, 0.25])
    y = PARX(1, 1).simulate(n, params=true, X=X, seed=11)
    m = PARX(1, 1).fit(y, X)
    assert np.all(np.abs(m.params_ - true) < 4 * m.bse_), (m.params_, m.bse_)


def test_loglinear_parameter_recovery():
    n = 3000
    X = _exog(n)
    true = np.array([0.2, 0.4, 0.3, 0.15, -0.1])
    y = LogLinearPARX().simulate(n, params=true, X=X, seed=3)
    m = LogLinearPARX().fit(y, X)
    assert np.all(np.abs(m.params_ - true) < 4 * m.bse_), (m.params_, m.bse_)
    assert list(m.params.index) == ["d", "a", "b", "c_season_sin", "c_season_cos"]


def test_loglinear_negative_coefficient_recovery():
    true = np.array([0.9, -0.3, 0.4])
    y = LogLinearPAR().simulate(3000, params=true, seed=5)
    m = LogLinearPAR().fit(y)
    assert m.params_[1] < 0
    assert np.all(np.abs(m.params_ - true) < 4 * m.bse_)


def test_glm_matches_generic_optimiser(data):
    y, X = data[0], data[1]
    m = PoissonGLM(lags=1).fit(y, X)
    res = optimize.minimize(lambda th: -m.loglik(th), np.zeros(4), method="BFGS")
    np.testing.assert_allclose(m.params_, res.x, atol=1e-4)
    naive = PoissonGLM().fit(y)
    assert np.exp(naive.params_[0]) == pytest.approx(y.mean())


# --------------------------------------------------------------------------------------
# prediction / forecasting / simulation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "make,xkind",
    [
        (lambda: PAR(2, 1), None),
        (lambda: PARX(1, 1), "nonneg"),
        (lambda: LogLinearPAR(), None),
        (lambda: LogLinearPARX(), "signed"),
        (lambda: PoissonGLM(lags=2), "signed"),
    ],
)
def test_forecast_shapes(make, xkind, data):
    y, X = data[0], _pick_X(xkind, data)
    tr, te = slice(0, 170), slice(170, 200)
    m = make().fit(y[tr], None if X is None else X[tr])
    lam = m.predict(y[te], None if X is None else X[te])
    assert lam.shape == (30,) and np.all(lam > 0)
    fc = m.forecast(8, None if X is None else X[te][:8], n_sims=300, seed=0)
    assert list(fc.columns) == ["step", "mean", "lower", "upper"]
    assert len(fc) == 8
    assert np.all(fc["lower"] <= fc["upper"]) and np.all(fc["mean"] > 0)
    sim = m.simulate(50, X=None if X is None else X[:50], seed=1)
    assert sim.shape == (50,) and sim.dtype.kind == "i"
    assert m.fitted_values().shape == (170,)


def test_predict_matches_refit_recursion(data):
    """One-step predictions on the test block equal the in-sample recursion on the full series."""
    y = data[0]
    m = PAR(1, 1).fit(y[:150])
    lam_full = np.empty(len(y))
    lam_full[0] = y[:150].mean()  # same pre-sample value as the training fit
    for t in range(1, len(y)):
        lam_full[t] = m.params_[0] + m.params_[1] * y[t - 1] + m.params_[2] * lam_full[t - 1]
    np.testing.assert_allclose(m.predict(y[150:]), lam_full[150:], rtol=1e-10)


def test_linear_mean_forecast_converges_to_unconditional_mean():
    y = PAR(1, 1).simulate(500, params=[0.6, 0.3, 0.4], seed=2)
    m = PAR(1, 1).fit(y)
    fc = m.forecast(400, n_sims=50, seed=0)
    target = m.params_[0] / (1 - m.persistence())
    assert fc["mean"].iloc[-1] == pytest.approx(target, rel=1e-6)


def test_simulate_is_reproducible():
    a = LogLinearPAR().simulate(100, params=[0.3, 0.4, 0.3], seed=9)
    b = LogLinearPAR().simulate(100, params=[0.3, 0.4, 0.3], seed=9)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------------------
# constraints and input validation
# --------------------------------------------------------------------------------------
def test_linear_stationarity_enforced():
    rng = np.random.default_rng(0)
    # strongly trending counts would want sum(alpha) + sum(beta) >= 1
    y = rng.poisson(np.linspace(1, 30, 300))
    m = PAR(1, 1).fit(y)
    assert m.persistence() < 1.0
    assert m.params_[0] > 0 and np.all(m.params_[1:] >= 0)
    assert np.all(m.fitted_values() > 0)


def test_loglinear_stability_enforced():
    rng = np.random.default_rng(1)
    y = rng.poisson(np.exp(np.linspace(0, 3, 300)))
    m = LogLinearPAR().fit(y)
    a, b = m.params_[1:3]
    assert max(abs(a), abs(b), abs(a + b)) < 1.0
    assert m.persistence() < 1.0
    unconstrained = LogLinearPAR(stationarity=False).fit(y)
    assert abs(unconstrained.params_[1]) < 1.0
    assert unconstrained.loglik_ >= m.loglik_ - 1e-6


def test_parx_rejects_negative_covariates(data):
    with pytest.raises(ValueError, match="non-negative"):
        PARX(1, 1).fit(data[0], data[1])


def test_parx_requires_covariates(data):
    with pytest.raises(ValueError, match="requires covariates"):
        PARX(1, 1).fit(data[0])


def test_predict_without_covariates_raises(data):
    y, X = data[0], data[1]
    m = LogLinearPARX().fit(y[:150], X[:150])
    with pytest.raises(ValueError, match="pass X"):
        m.predict(y[150:])
    m2 = LogLinearPAR().fit(y[:150])
    with pytest.raises(ValueError, match="without covariates"):
        m2.predict(y[150:], X[150:])


@pytest.mark.parametrize("bad", [[1, 2, -1, 3] * 10, [1.5, 2, 3, 4] * 10, [1, np.nan, 2, 3] * 10])
def test_rejects_non_count_data(bad):
    with pytest.raises(ValueError):
        PAR().fit(bad)


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        PAR().predict([1, 2, 3])


def test_summary_contains_parameters(data):
    m = PAR(1, 1).fit(data[0])
    s = m.summary()
    for name in ("omega", "alpha1", "beta1", "loglik", "aic"):
        assert name in s
    assert set(m.coef_table().columns) == {"estimate", "std_err", "z", "at_bound"}


def test_recovery_at_large_count_scale():
    """omega lives on the count scale; the optimiser rescales it, so large counts still fit."""
    true = np.array([5000.0, 0.3, 0.5])
    y = PAR(1, 1).simulate(1500, params=true, seed=8)
    m = PAR(1, 1).fit(y)
    np.testing.assert_allclose(m.params_[1:], true[1:], atol=0.05)
    assert m.params_[0] == pytest.approx(true[0], rel=0.25)


# --------------------------------------------------------------------------------------
# regression tests for issues found in review
# --------------------------------------------------------------------------------------
def _seasonal_par_series(coef, n=3000, seed=0):
    """Linear PAR(1,1) with a signed sinusoidal term (kept positive by construction)."""
    S = _exog(n).to_numpy()
    rng = np.random.default_rng(seed)
    lam, ys = 2.0, []
    for t in range(n):
        lam = 1.0 + 0.3 * (ys[-1] if ys else 2) + 0.3 * lam + S[t] @ coef
        ys.append(rng.poisson(lam))
    return np.array(ys, dtype=float)


def test_parx_seasonal_basis_handles_any_phase():
    """A seasonal peak outside the first quadrant (here -sin, peak near day 270) is picked up
    by the rectified basis; the old 1 + sin / 1 + cos basis left gamma at its bound."""
    y = _seasonal_par_series([-0.8, 0.0])
    X = _exog(len(y), nonnegative=True)
    assert list(X.columns) == [
        "season_sin_pos",
        "season_sin_neg",
        "season_cos_pos",
        "season_cos_neg",
    ]
    assert (X.to_numpy() >= 0).all()
    m = PARX(1, 1).fit(y, X)
    gamma = m.params.filter(like="gamma")
    assert gamma["gamma_season_sin_neg"] > 0.5
    assert m.loglik_ > PAR(1, 1).fit(y).loglik_ + 100  # seasonality is captured


def test_loglinear_opposite_signs_not_pinned():
    """a = 0.8, b = -0.5 is stationary (max(|a|, |b|, |a+b|) < 1) though |a| + |b| > 1;
    the default constraint must not bias the fit towards |a| + |b| = 1."""
    true = np.array([0.6, 0.8, -0.5])
    y = LogLinearPAR(stationarity=False).simulate(2000, params=true, seed=2)
    m = LogLinearPAR().fit(y)
    assert abs(m.params_[1]) + abs(m.params_[2]) > 1.0
    assert np.all(np.abs(m.params_ - true) < 4 * m.bse_)
    assert not m.coef_table()["at_bound"].any()


def test_at_bound_flags_active_stationarity_constraint():
    rng = np.random.default_rng(1)
    y = rng.poisson(np.exp(np.linspace(0, 3, 300)))
    m = LogLinearPAR().fit(y)
    hi = 1 - 1e-4
    assert m.persistence() == pytest.approx(hi, abs=1e-6)
    flags = m.coef_table()["at_bound"]
    assert flags["a"] and flags["b"] and not flags["d"]


def test_glm_separation_not_reported_as_converged():
    rng = np.random.default_rng(1)
    x = (np.arange(200) % 2).astype(float)
    y = np.where(x == 1, 0, rng.poisson(3, 200))
    with pytest.warns(RuntimeWarning, match="MLE does not exist"):
        m = PoissonGLM().fit(y, x[:, None])
    assert not m.converged_
    ok = PoissonGLM(lags=1).fit(rng.poisson(2.0, 200))
    assert ok.converged_


def test_hessian_stays_inside_bounds():
    """At omega on its lower bound the central stencil would leave the feasible region
    (negative intensities, penalty value); one-sided differences keep the Hessian finite."""
    from piracy_par.models import _numeric_hessian

    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40), rng.poisson(3, 60), np.zeros(40), rng.poisson(3, 60)].astype(float)
    m = PAR(1, 1).fit(y)
    theta = np.array([1e-8, 0.5, 0.45])
    bounds, _ = m._bounds_and_constraints(0)

    def nll_grad(t):
        return m._nll_grad(t, m._y, None, m._ybar)

    h = 1e-5
    below = theta - np.array([h, 0.0, 0.0])
    assert nll_grad(below)[0] == 1e12  # infeasible: the central stencil hits the penalty
    H = _numeric_hessian(lambda t: nll_grad(t)[1], theta, bounds)
    forward = (nll_grad(theta + np.array([h, 0.0, 0.0]))[1] - nll_grad(theta)[1]) / h
    assert H[0, 0] == pytest.approx(forward[0])
    assert H[0, 0] > 0 and np.all(np.isfinite(H))


def test_sparse_data_reaches_better_optimum():
    """Very sparse counts: the likelihood is flat with local maxima. The extra starts must
    reach at least the values found by an independent multistart search in review."""
    rng = np.random.default_rng(7)
    n = 400
    t = np.arange(n)
    # reproduce the review's random stream (several unrelated draws come first)
    rng.poisson(np.linspace(1, 30, n))
    rng.poisson(np.exp(np.cumsum(rng.normal(0, 0.1, n)) + 1))
    rng.poisson(3, n)
    rng.poisson(1e4 + 5e3 * np.sin(t / 10))
    y = rng.poisson(0.1, n).astype(float)
    Xn = np.c_[1 + np.sin(2 * np.pi * t / 52), 1 + np.cos(2 * np.pi * t / 52)]
    assert PARX(1, 1).fit(y, Xn).loglik_ >= -144.031
    assert LogLinearPAR().fit(y).loglik_ >= -144.341
    assert LogLinearPARX().fit(y, Xn - 1).loglik_ >= -143.528
