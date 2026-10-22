"""Data helpers: IMB map parsing/scraping, weekly aggregation, synthetic sample.

Submodules are imported lazily so ``python -m piracy_par.data.<module>`` runs cleanly.
"""

from importlib import import_module

_EXPORTS = {
    "aggregate_weekly": "weekly",
    "load_weekly_counts": "weekly",
    "seasonal_features": "weekly",
    "extract_details_from_html": "imb_parser",
    "parse_incident": "imb_parser",
    "parse_lines": "imb_parser",
    "load_sample": "sample",
    "make_synthetic_weekly": "sample",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
