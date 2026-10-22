"""Scraper tests. All incident strings here are invented; no network is used."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from piracy_par.data import imb_scraper
from piracy_par.data.imb_parser import (
    INCIDENT_COLUMNS,
    classify_incident_type,
    dms_to_decimal,
    extract_details_from_html,
    infer_vessel_type,
    normalize_line,
    parse_incident,
    parse_lines,
    place_is_ok,
    split_place_if_glued,
)
from piracy_par.data.weekly import aggregate_weekly

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "token,expected",
    [
        ("01:21N", 1 + 21 / 60),
        ("104:21E", 104 + 21 / 60),
        ("06:26.1S", -(6 + 26.1 / 60)),
        ("003:22.9W", -(3 + 22.9 / 60)),
        ("12°30.5N", 12 + 30.5 / 60),
        ("95:00N", None),  # latitude out of range
        ("181:00E", None),  # longitude out of range
        ("10:75N", None),  # minutes >= 60
        ("garbage", None),
        ("", None),
    ],
)
def test_dms_to_decimal(token, expected):
    out = dms_to_decimal(token)
    if expected is None:
        assert out is None
    else:
        assert out == pytest.approx(expected)


def test_normalize_fixes_known_typos():
    assert normalize_line("07:07.2024: 0100 UTC: Posn: 01:10N – 103:50E, X Port. Y.").startswith(
        "07.07.2024"
    )
    s = normalize_line("01.01.2024: Posn: 17.02.7N – 082:20.1E, Kakinada Anchorage. Z.")
    assert "17:02.7N" in s
    s = normalize_line("01.01.2024: Posn: 01:21N - 104:21EE, Singapore Straits. Z.")
    assert "104:21E," in s
    # a comma between lat and lon becomes a dash; hyphens elsewhere are kept
    s = normalize_line("01.01.2024: Posn: 01:21N, 104:21E, Singapore Straits. Ro-Ro vessel.")
    assert "01:21N – 104:21E" in s and "Ro-Ro" in s


def test_parse_pattern_a():
    inc = parse_incident(
        "03.02.2025: 2310 UTC: Posn: 01:21N – 104:21E, Singapore Straits. "
        "Four robbers boarded a bulk carrier underway and escaped."
    )
    assert inc.parse_source == "A"
    assert (inc.date, inc.time) == ("03.02.2025", "2310 UTC")
    assert inc.place == "Singapore Straits"
    assert inc.narration.startswith("Four robbers")
    assert inc.lat_decimal == pytest.approx(1.35) and inc.lon_decimal == pytest.approx(104.35)


def test_parse_glued_place_and_narration():
    # no full stop between the place and the narration
    inc = parse_incident(
        "11.04.2024: 0300 LT: Posn: 01:15N – 104:05E, Singapore Straits Two robbers boarded a tug."
    )
    assert inc.place == "Singapore Straits"
    assert inc.narration.startswith("Two robbers boarded a tug")


def test_parse_patterns_b_and_c():
    b = parse_incident("02.05.2024: Night Hours: Posn: Off Some Coast. Skiff approached a vessel.")
    assert b.parse_source == "B" and b.place == "Off Some Coast" and b.lat_token == ""
    c = parse_incident("02.05.2024: 0200 UTC: Some Harbour, Somewhere. Robbers stole stores.")
    assert c.parse_source == "C" and c.place == "Some Harbour, Somewhere"
    assert parse_incident("not an incident") is None


def test_split_place_is_case_sensitive():
    assert split_place_if_glued("Port of the Isles") == ("Port of the Isles", "")
    assert split_place_if_glued("Lagos Anchorage Robbers boarded") == (
        "Lagos Anchorage",
        "Robbers boarded",
    )


def test_place_qc_and_enrichment():
    assert place_is_ok("Lagos Anchorage")
    assert not place_is_ok("12:30")
    assert not place_is_ok("Lagos")  # valid place, but flagged for manual QC
    assert classify_incident_type("Pirates hijacked the tanker") == "Hijacked"
    assert classify_incident_type("Shots were fired at the vessel") == "Fired upon"
    assert classify_incident_type("Robbers boarded the vessel") == "Boarding"
    assert classify_incident_type("A skiff approached the ship") == "Suspicious"
    assert classify_incident_type("") == "Unknown"
    assert infer_vessel_type("an anchored product tanker") == ("tanker", "inferred_from_text")
    assert infer_vessel_type("nothing here") == ("Unknown", "none")


def test_extract_details_header_first():
    html = (FIX / "imb_map_snippet.html").read_text()
    lines = extract_details_from_html(html)
    assert len(lines) == 3
    assert lines[0].startswith("03.02.2025") and "\n" in lines[0]  # <br> kept as newline
    assert "ship's stores" in lines[1]
    with pytest.raises(ValueError):
        extract_details_from_html(html, table_id="wpgmza_table_404")


def test_extract_details_third_cell_fallback():
    html = (FIX / "imb_map_noheader.html").read_text()
    lines = extract_details_from_html(html, table_id="wpgmza_table_7")
    assert [ln[:10] for ln in lines] == ["05.03.2024", "09.03.2024"]


def test_parse_lines_from_fixture():
    lines = extract_details_from_html((FIX / "imb_map_snippet.html").read_text())
    df, failed = parse_lines(lines, 2025)
    assert list(df.columns) == INCIDENT_COLUMNS
    assert len(df) == 2 and failed == ["not an incident line"]
    assert df.loc[1, "place"] == "Lagos Anchorage, Nigeria"
    assert df.loc[1, "vessel_type_inferred"] == "tanker"
    assert (df["year"] == 2025).all()


def test_place_rename_and_salvage():
    lines = extract_details_from_html((FIX / "imb_map_noheader.html").read_text())
    df, _ = parse_lines(lines, 2024)
    assert df.loc[0, "place"] == "Chattogram Anchorage"
    assert df.loc[1, "place"].startswith("Around 30nm east of")


def test_scrape_with_fake_browser(tmp_path, caplog):
    pages = {
        imb_scraper.year_url(2024): (FIX / "imb_map_noheader.html").read_text(),
        imb_scraper.year_url(2025): (FIX / "imb_map_snippet.html").read_text(),
    }
    calls, sleeps, quit_called = [], [], []

    class FakeDriver:
        def quit(self):
            quit_called.append(True)

    def fake_fetch(driver, url, table_id):
        calls.append(url)
        if url not in pages:
            raise ConnectionError("simulated outage")
        return pages[url]

    df = imb_scraper.scrape(
        [2023, 2024, 2025],
        tmp_path,
        table_ids={2024: "wpgmza_table_7", 2025: None},
        delay=0.0,  # raised to the minimum, but sleep is faked
        retries=2,
        driver_factory=lambda headless: FakeDriver(),
        fetch=fake_fetch,
        robots_check=lambda url: True,
        sleep=sleeps.append,
    )
    # 2023 failed twice and was skipped; the other years were parsed
    assert calls.count(imb_scraper.year_url(2023)) == 2
    assert set(df["year"]) == {2024, 2025} and len(df) == 4
    assert all(s >= imb_scraper.MIN_DELAY for s in sleeps)
    assert quit_called == [True]
    for name in (
        "raw_data_2024.csv",
        "incidents_2025.csv",
        "all_incidents.csv",
        "places_list.csv",
        "failed_lines.csv",
    ):
        assert (tmp_path / name).exists()

    # offline re-parse and weekly aggregation through the CLI entry point
    out2 = tmp_path / "reparsed"
    assert (
        imb_scraper.main(
            ["parse", "--raw", str(tmp_path / "raw_data_2025.csv"), "--out", str(out2)]
        )
        == 0
    )
    weekly_csv = tmp_path / "weekly.csv"
    assert (
        imb_scraper.main(
            [
                "weekly",
                "--incidents",
                str(tmp_path / "all_incidents.csv"),
                "--out",
                str(weekly_csv),
                "--by-type",
            ]
        )
        == 0
    )
    weekly = pd.read_csv(weekly_csv)
    assert weekly["count"].sum() == 4
    assert "count_boarding" in weekly.columns


def test_robots_disallow_skips_year(tmp_path):
    with pytest.raises(imb_scraper.ScrapeError):
        imb_scraper.scrape(
            [2025],
            tmp_path,
            robots_check=lambda url: False,
            driver_factory=lambda headless: pytest.fail("no driver needed"),
            sleep=lambda s: None,
        )


def test_fetch_fails_fast_without_selenium(tmp_path, monkeypatch):
    """Missing Selenium: no robots.txt request, no retries, nonzero exit code."""
    monkeypatch.setattr(imb_scraper.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        imb_scraper, "robots_allowed", lambda url: pytest.fail("robots.txt must not be fetched")
    )
    monkeypatch.setattr(imb_scraper.time, "sleep", lambda s: pytest.fail("no backoff"))
    assert imb_scraper.main(["fetch", "--years", "2025", "--out", str(tmp_path)]) == 1


def test_fetch_exit_code_when_every_year_fails(tmp_path, monkeypatch):
    def failing_scrape(*args, **kwargs):
        raise imb_scraper.ScrapeError("none of the requested years could be fetched")

    monkeypatch.setattr(imb_scraper, "scrape", failing_scrape)
    assert imb_scraper.main(["fetch", "--years", "2025", "--out", str(tmp_path)]) == 1


def test_weekly_aggregation_convention():
    inc = pd.DataFrame(
        {
            "date": ["31.12.2024", "01.01.2025", "06.01.2025", "07.01.2025", "07.01.2025", "bad"],
            "time": ["", "", "", "", "", ""],
            "narration": ["a", "b", "c", "d", "d", "e"],
        }
    )
    w = aggregate_weekly(inc, start="2024-12-30", end="2025-01-27")
    # Tue-Mon weeks labelled by the closing Monday; duplicate (07.01, d) counted once
    assert list(w["week_end"].dt.strftime("%Y-%m-%d")) == [
        "2024-12-30",
        "2025-01-06",
        "2025-01-13",
        "2025-01-20",
        "2025-01-27",
    ]
    assert list(w["count"]) == [0, 3, 1, 0, 0]


def test_import_does_not_load_selenium():
    code = "import piracy_par.data.imb_scraper, sys; print('selenium' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
