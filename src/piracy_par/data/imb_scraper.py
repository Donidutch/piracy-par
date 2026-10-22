"""Scrape the ICC-CCS / IMB live piracy map and build weekly counts.

The map pages render their incident table with JavaScript (WP Google Maps + DataTables), so
fetching needs a real browser: install the optional extra ``piracy-par[scrape]`` (Selenium)
and Chrome/Chromium. Parsing and aggregation are browser-free.

The scraped text is copyrighted by ICC Commercial Crime Services. Check the site's terms
before scraping, keep the delay, and do not redistribute the output.

Usage::

    piracy-par-scrape fetch --years 2023 2024 2025 --out scrape_out
    piracy-par-scrape parse --raw scrape_out/raw_data_*.csv --out scrape_out
    piracy-par-scrape weekly --incidents scrape_out/all_incidents.csv --out weekly_counts.csv

Importing this module does not import Selenium or touch the network.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import pandas as pd

from .imb_parser import extract_details_from_html, parse_lines
from .weekly import DEDUP_COLUMNS, aggregate_weekly

log = logging.getLogger(__name__)

__all__ = ["year_url", "DEFAULT_TABLE_IDS", "USER_AGENT", "scrape", "parse_raw_files", "main"]

CURRENT_YEAR_URL = "https://icc-ccs.org/map/"
YEAR_URL_TEMPLATE = "https://icc-ccs.org/piracy-map-{year}/"
#: Table ids observed on the pages in 2025. ``None`` means auto-detect ``wpgmza_table_*``.
DEFAULT_TABLE_IDS: dict[int, str | None] = {
    2023: "wpgmza_table_4",
    2024: "wpgmza_table_3",
    2025: "wpgmza_table_19",
}
CURRENT_YEAR = 2025
USER_AGENT = "piracy-par-research-scraper/0.1 (+academic count time-series project)"
MIN_DELAY = 2.0


def year_url(year: int, current_year: int = CURRENT_YEAR) -> str:
    return CURRENT_YEAR_URL if year >= current_year else YEAR_URL_TEMPLATE.format(year=year)


# --------------------------------------------------------------------------------------
# browser side (Selenium imported lazily)
# --------------------------------------------------------------------------------------
def make_driver(*, headless: bool = True, user_agent: str = USER_AGENT, page_timeout: int = 60):
    try:
        from selenium import webdriver
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Fetching needs Selenium: install with `uv sync --extra scrape` "
            "or `pip install piracy-par[scrape]`."
        ) from exc
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument(f"--user-agent={user_agent}")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(page_timeout)
    return driver


def fetch_page_html(driver, url: str, table_id: str | None, *, timeout: float = 12.0) -> str:
    """Load ``url``, show all table rows, and return the rendered page source."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select, WebDriverWait

    driver.get(url)
    if table_id is None:
        css = 'table[id^="wpgmza_table_"]'
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
        table_id = driver.find_element(By.CSS_SELECTOR, css).get_attribute("id")
    else:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, table_id)))
    try:
        dropdown = Select(driver.find_element(By.NAME, f"{table_id}_length"))
        try:
            dropdown.select_by_value("-1")
        except Exception:
            dropdown.select_by_visible_text("All")
    except Exception as exc:  # the length selector is missing on some pages
        log.info("No 'show all' selector for %s (%s); using rendered rows", table_id, exc)
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, f"#{table_id} tbody td"))
    )
    return driver.page_source


def robots_allowed(url: str, user_agent: str = USER_AGENT) -> bool:
    parts = urlsplit(url)
    rp = robotparser.RobotFileParser(f"{parts.scheme}://{parts.netloc}/robots.txt")
    try:
        rp.read()
    except OSError as exc:
        log.warning("Could not read robots.txt (%s); proceeding cautiously", exc)
        return True
    return rp.can_fetch(user_agent, url)


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def _write_outputs(frames: list[pd.DataFrame], failed: list[str], out: Path) -> pd.DataFrame:
    all_inc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_inc.empty:
        all_inc = all_inc.drop_duplicates(subset=list(DEDUP_COLUMNS))
        all_inc.to_csv(out / "all_incidents.csv", index=False)
        places = pd.Series(all_inc["place"].dropna().unique(), name="place")
        places.to_csv(out / "places_list.csv", index=False)
    if failed:
        pd.DataFrame({"raw_text": failed}).to_csv(out / "failed_lines.csv", index=False)
    return all_inc


def scrape(
    years: Iterable[int],
    out_dir,
    *,
    table_ids: dict[int, str | None] | None = None,
    delay: float = MIN_DELAY,
    retries: int = 3,
    raw_only: bool = False,
    headless: bool = True,
    driver_factory: Callable = make_driver,
    fetch: Callable = fetch_page_html,
    robots_check: Callable[[str], bool] = robots_allowed,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch each year's map page, save raw details text, and parse it.

    A year that fails after ``retries`` attempts is logged and skipped. Writes
    ``raw_data_{year}.csv`` and, unless ``raw_only``, ``incidents_{year}.csv``,
    ``all_incidents.csv``, ``places_list.csv`` and ``failed_lines.csv``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = {**DEFAULT_TABLE_IDS, **(table_ids or {})}
    if delay < MIN_DELAY:
        log.warning("delay %.1fs raised to the %.1fs minimum", delay, MIN_DELAY)
        delay = MIN_DELAY

    frames, failed_all = [], []
    driver = None
    try:
        for i, year in enumerate(years):
            url = year_url(year)
            if not robots_check(url):
                log.error("robots.txt disallows %s; skipping %s", url, year)
                continue
            if i > 0:
                sleep(delay)
            html = None
            for attempt in range(1, retries + 1):
                try:
                    if driver is None:
                        driver = driver_factory(headless=headless)
                    html = fetch(driver, url, ids.get(year))
                    break
                except Exception as exc:  # network/browser errors: back off and retry
                    log.warning("%s attempt %d/%d failed: %s", year, attempt, retries, exc)
                    if attempt < retries:
                        sleep(delay * 2**attempt)
            if html is None:
                log.error("giving up on %s", year)
                continue
            lines = extract_details_from_html(html, ids.get(year))
            pd.DataFrame({"raw_text": lines}).to_csv(out / f"raw_data_{year}.csv", index=False)
            log.info("%s: %d rows saved", year, len(lines))
            if raw_only:
                continue
            df, failed = parse_lines(lines, year)
            df.to_csv(out / f"incidents_{year}.csv", index=False)
            frames.append(df)
            failed_all.extend(failed)
    finally:
        if driver is not None:
            driver.quit()
    return _write_outputs(frames, failed_all, out) if not raw_only else pd.DataFrame()


def parse_raw_files(paths: Iterable, out_dir) -> pd.DataFrame:
    """Offline re-parse of saved ``raw_data_{year}.csv`` files (no network)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, failed_all = [], []
    for p in map(Path, paths):
        lines = pd.read_csv(p)["raw_text"].dropna().astype(str).tolist()
        stem = p.stem.removeprefix("raw_data_")
        year = int(stem) if stem.isdigit() else None
        df, failed = parse_lines(lines, year)
        if year is not None:
            df.to_csv(out / f"incidents_{year}.csv", index=False)
        frames.append(df)
        failed_all.extend(failed)
    return _write_outputs(frames, failed_all, out)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_table_ids(items: list[str]) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    for item in items or []:
        year, _, tid = item.partition("=")
        out[int(year)] = tid or None
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="piracy-par-scrape", description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="fetch map pages with a headless browser and parse them")
    f.add_argument("--years", type=int, nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--table-id", action="append", metavar="YEAR=ID",
                   help="override a year's table id (empty ID = auto-detect)")
    f.add_argument("--delay", type=float, default=MIN_DELAY,
                   help=f"seconds between page loads (minimum {MIN_DELAY})")
    f.add_argument("--retries", type=int, default=3)
    f.add_argument("--no-headless", action="store_true")
    f.add_argument("--raw-only", action="store_true", help="save raw text, skip parsing")

    p = sub.add_parser("parse", help="parse saved raw_data_{year}.csv files offline")
    p.add_argument("--raw", nargs="+", required=True)
    p.add_argument("--out", required=True)

    w = sub.add_parser("weekly", help="aggregate parsed incidents to weekly counts")
    w.add_argument("--incidents", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--freq", default="W-MON")
    w.add_argument("--by-type", action="store_true")
    w.add_argument("--start")
    w.add_argument("--end")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "fetch":
        df = scrape(args.years, args.out, table_ids=_parse_table_ids(args.table_id),
                    delay=args.delay, retries=args.retries, raw_only=args.raw_only,
                    headless=not args.no_headless)
        print(f"{len(df)} incidents written to {args.out}")
    elif args.cmd == "parse":
        df = parse_raw_files(args.raw, args.out)
        print(f"{len(df)} incidents written to {args.out}")
    elif args.cmd == "weekly":
        inc = pd.read_csv(args.incidents, dtype={"date": str})
        weekly = aggregate_weekly(inc, freq=args.freq, by_type=args.by_type,
                                  start=args.start, end=args.end)
        weekly.to_csv(args.out, index=False, date_format="%Y-%m-%d")
        print(f"{len(weekly)} weeks written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
