"""Weekly aggregation of incident records and helpers for weekly count series.

Week convention: ``freq="W-MON"`` puts each incident in the week that ends on the next
Monday (Tuesday-to-Monday weeks, right-closed), labelled by that Monday (``week_end``).
Weeks without incidents are kept with a count of zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["aggregate_weekly", "seasonal_features", "load_weekly_counts", "DEDUP_COLUMNS"]

DEDUP_COLUMNS = ("date", "time", "lat_token", "lon_token", "place", "narration")


def _parse_dates(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    missing = out.isna() & s.notna()
    if missing.any():
        out[missing] = pd.to_datetime(s[missing], dayfirst=True, errors="coerce")
    return out


def aggregate_weekly(
    incidents: pd.DataFrame,
    *,
    freq: str = "W-MON",
    by_type: bool = False,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Count incidents per week.

    Dates are parsed day-first (``DD.MM.YYYY`` as on the IMB map); unparseable dates are
    dropped, and duplicate incidents (same date, time, coordinates, place and narration)
    are counted once. ``start``/``end`` extend the range with zero weeks.
    Returns columns ``week_end, count`` (plus ``count_<incident_type>`` if ``by_type``).
    """
    df = incidents.copy()
    df["_date"] = _parse_dates(df["date"].astype("string"))
    df = df.dropna(subset=["_date"])
    subset = [c for c in DEDUP_COLUMNS if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset)
    df = df.set_index("_date").sort_index()

    counts = df.resample(freq).size().rename("count")
    if start is not None or end is not None:
        lo = pd.Timestamp(start) if start is not None else counts.index.min()
        hi = pd.Timestamp(end) if end is not None else counts.index.max()
        full = pd.date_range(lo, hi, freq=freq)
        if len(counts):
            full = full.union(counts.index)
        counts = counts.reindex(full, fill_value=0)
    out = counts.to_frame()
    if by_type and "incident_type" in df.columns:
        typed = df.groupby("incident_type").resample(freq).size().unstack(0, fill_value=0)
        typed = typed.reindex(out.index, fill_value=0)
        typed.columns = [f"count_{str(c).lower().replace(' ', '_')}" for c in typed.columns]
        out = out.join(typed)
    out.index.name = "week_end"
    return out.reset_index().astype({c: int for c in out.columns})


def seasonal_features(dates, *, nonnegative: bool = False, harmonics: int = 1) -> pd.DataFrame:
    """Annual Fourier terms ``sin/cos(2 pi k * (day_of_year - 1) / 365.25)``.

    With ``nonnegative=False`` (default) there are two signed columns per harmonic, for the
    log-linear models and the GLM, whose covariate coefficients are unrestricted.

    ``nonnegative=True`` gives four half-wave rectified columns per harmonic,
    ``max(sin, 0)``, ``max(-sin, 0)``, ``max(cos, 0)`` and ``max(-cos, 0)``, for the linear
    PARX, which needs non-negative covariates and coefficients. A shifted pair
    ``1 + sin``/``1 + cos`` with non-negative coefficients could only place the seasonal peak
    in the first quadrant of the year; the rectified basis can put it anywhere, and unlike
    ``1 +/- sin``/``1 +/- cos`` it is not collinear with the intercept. With non-negative
    coefficients it can only add bumps to the baseline, so it approximates a sinusoid rather
    than reproducing one exactly.
    """
    idx = pd.DatetimeIndex(dates)
    angle = 2.0 * np.pi * (idx.day_of_year.to_numpy() - 1) / 365.25
    cols = {}
    for k in range(1, harmonics + 1):
        suffix = "" if harmonics == 1 else str(k)
        s, c = np.sin(k * angle), np.cos(k * angle)
        if nonnegative:
            cols[f"season_sin{suffix}_pos"] = np.maximum(s, 0.0)
            cols[f"season_sin{suffix}_neg"] = np.maximum(-s, 0.0)
            cols[f"season_cos{suffix}_pos"] = np.maximum(c, 0.0)
            cols[f"season_cos{suffix}_neg"] = np.maximum(-c, 0.0)
        else:
            cols[f"season_sin{suffix}"] = s
            cols[f"season_cos{suffix}"] = c
    return pd.DataFrame(cols, index=idx)


def load_weekly_counts(path, *, date_col: str = "week_end", count_col: str = "count") -> pd.Series:
    """Read a weekly count CSV into an integer Series indexed by date.

    Lines starting with ``#`` are treated as comments.
    """
    df = pd.read_csv(Path(path), comment="#")
    for col in (date_col, count_col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not in {list(df.columns)}")
    s = pd.Series(
        df[count_col].to_numpy(), index=pd.to_datetime(df[date_col]), name=count_col
    ).sort_index()
    if s.isna().any():
        raise ValueError(f"{count_col!r} has missing values.")
    if not np.all(s.to_numpy() == np.round(s.to_numpy())):
        raise ValueError(f"{count_col!r} is not integer-valued; Poisson models need counts.")
    return s.astype(int)
