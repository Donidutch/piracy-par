"""Browser-free parsing of IMB Piracy Reporting Centre map entries.

Each entry on the ICC-CCS live piracy map has a free-text "details" cell shaped like::

    DD.MM.YYYY: [HHMM UTC|LT | Night Hours ...:] [Posn:] LAT - LON, Place. Narration...

This module turns those strings into structured records, repairs the place field and adds
heuristic incident-type and vessel-type labels. It also extracts the details column from a
saved page (HTML) without a browser. Nothing here touches the network.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser

import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "Incident",
    "normalize_line",
    "dms_to_decimal",
    "parse_incident",
    "parse_lines",
    "split_place_if_glued",
    "clean_place_value",
    "place_is_ok",
    "salvage_place",
    "classify_incident_type",
    "infer_vessel_type",
    "extract_details_from_html",
    "INCIDENT_COLUMNS",
]

INCIDENT_COLUMNS = [
    "date", "time", "lat_token", "lon_token", "place", "narration", "lat_decimal",
    "lon_decimal", "raw_details", "parse_source", "year", "incident_type",
    "vessel_type_inferred", "vessel_type_source", "place_ok",
]


@dataclass
class Incident:
    date: str
    time: str
    lat_token: str
    lon_token: str
    place: str
    narration: str
    lat_decimal: float | None = None
    lon_decimal: float | None = None
    raw_details: str | None = None
    parse_source: str | None = None


# --------------------------------------------------------------------------------------
# normalisation and patterns
# --------------------------------------------------------------------------------------
MOJIBAKE_FIXES = (
    ("â€“", "–"),  # en dash
    ("â€™", "’"),
    ("â€œ", "“"),
    ("â€\x9d", "”"),
    ("â€˜", "‘"),
)

LATLON_TOKEN = r"(?:\d{1,3}[:°\s]\d{1,2}(?:\.\d*)?\s*[NSEW])"
DASH_OR_COMMA = r"(?:\s*[–—\-]\s*|,\s*)"
TIME_TOKEN = (
    r"(?:(?:\d{3,4}\s*(?:UTC|LT)?)|"
    r"(?:Night\s+Hours|Daylight\s+Hours|Early\s+Morning|Morning|Afternoon|Evening|Dusk|Dawn))"
)  # spaces are written as \s+ because the patterns below are compiled with re.VERBOSE


def normalize_line(line: str) -> str:
    """Clean a raw details string so the patterns can match it."""
    s = (line or "").strip()
    s = re.sub(r"\s*\n+\s*", " ", s)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    for bad, good in MOJIBAKE_FIXES:
        s = s.replace(bad, good)
    s = re.sub(r"\b(Posn|Position|Pos)\s*:\s*", r"\1: ", s, flags=re.IGNORECASE)
    # date typo "07:07.2024" -> "07.07.2024"
    s = re.sub(r"^(\s*\d{2}):(\d{2}\.\d{4}\b)", r"\1.\2", s)
    # "17.02.7N" -> "17:02.7N"
    s = re.sub(r"\b(\d{1,3})\.(\d{1,2}\.\d+)\s*([NSEW])\b", r"\1:\2\3", s)
    # doubled hemisphere letter "104:21EE" -> "104:21E"
    s = re.sub(r"\b(\d{1,3}[:°\s]\d{1,2}(?:\.\d*)?\s*)([NSEW])\2\b", r"\1\2", s)
    # lat/lon separator (hyphen, em dash or comma) -> " – "; other hyphens are left alone
    s = re.sub(rf"({LATLON_TOKEN})\s*(?:[–—\-]|,)\s*(?={LATLON_TOKEN})", r"\1 – ", s)
    return re.sub(r"\s+", " ", s).strip()


PATTERN_A = re.compile(
    rf"""^\s*(?P<date>\d{{2}}\.\d{{2}}\.\d{{4}})\s*:\s*
    (?:(?P<time>{TIME_TOKEN})\s*:?)?
    \s*(?:Posn|Position|Pos)\s*:\s*
    (?P<lat>{LATLON_TOKEN}){DASH_OR_COMMA}(?P<lon>{LATLON_TOKEN})
    \s*[,.]\s*(?P<place>[^.]+)\.\s*(?P<narration>.*?)\s*$""",
    re.VERBOSE | re.IGNORECASE,
)
PATTERN_B = re.compile(
    rf"""^\s*(?P<date>\d{{2}}\.\d{{2}}\.\d{{4}})\s*:\s*
    (?:(?P<time>{TIME_TOKEN})\s*:?)?
    \s*(?:Posn|Position|Pos)\s*:\s*
    (?P<place>[^.]+)\.\s*(?P<narration>.*?)\s*$""",
    re.VERBOSE | re.IGNORECASE,
)
PATTERN_C = re.compile(
    rf"""^\s*(?P<date>\d{{2}}\.\d{{2}}\.\d{{4}})\s*:\s*
    (?:(?P<time>{TIME_TOKEN})\s*:?\s*)?
    (?P<place>[^.]+)\.\s*(?P<narration>.*?)\s*$""",
    re.VERBOSE | re.IGNORECASE,
)
PATTERNS = (("A", PATTERN_A), ("B", PATTERN_B), ("C", PATTERN_C))

_DMS_RE = re.compile(r"^(\d{1,3})[:\s](\d{1,2}(?:\.\d*)?)\s*([NSEW])$")


def dms_to_decimal(token: str) -> float | None:
    """Convert ``DD:MM.m[NSEW]`` (degrees and decimal minutes) to signed decimal degrees.

    Returns ``None`` when the token is malformed or out of range.
    """
    if not token:
        return None
    t = token.strip().upper().replace("°", ":")
    m = _DMS_RE.match(t)
    if not m:
        return None
    degrees = int(m.group(1))
    minutes = float(m.group(2).rstrip(".") or 0)
    hemi = m.group(3)
    if minutes >= 60:
        return None
    value = degrees + minutes / 60.0
    limit = 90.0 if hemi in "NS" else 180.0
    if value > limit:
        return None
    return -value if hemi in "SW" else value


# --------------------------------------------------------------------------------------
# place / narration helpers
# --------------------------------------------------------------------------------------
# Narration usually starts with a capitalised word from this list. The search is
# case-sensitive so lowercase words inside a place name ("Port of the ...") do not split it.
NARRATION_START_RE = re.compile(
    r"\b("
    r"Duty|Alert|Alarm|Armed|Crew|Master|Persons?|Robbers?|Perpetrators?|Individuals?|"
    r"The|Upon|On|After|Reports?|VTIS|VTS|Coast\s+Guard|Authorities|Police|Navy|SSAS|"
    r"Immediately|Suddenly|Noticed|Board(?:ed|ing)?|Approach(?:ed|ing)?|Fired|Shot|"
    r"Raised|Search|Mustered|Escap(?:e|ed|ing)|Tied|Threaten(?:ed|ing)|"
    r"Cargo|Stores?|Properties|Stole|Theft|Hijack(?:ed)?|"
    r"Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten"
    r")\b"
)
COORDS_PAIR_RE = re.compile(
    r"\b\d{1,3}[:\s]\d{1,2}(?:\.\d+)?\s*[NSEW]\s*[–\-]\s*"
    r"\d{1,3}[:\s]\d{1,2}(?:\.\d+)?\s*[NSEW]\s*,?\s*",
    re.I,
)
TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*$")
POSN_RE = re.compile(r"\b(Posn|Position|UTC|LT)\b", re.I)
GEO_WORD = re.compile(
    r"(Anchorage|Port|Harbour|Harbor|Jetty|Terminal|Roads|Straits?|Gulf|Bay|Coast|Sea|"
    r"Ocean|Channel|River)",
    re.I,
)
DIST_PHRASE = re.compile(
    r"(?i)\b(?:around|about|approximately)\s*\d{1,4}(?:\.\d+)?\s*(?:nm|nautical miles?)\b.*\bof\b"
)
DIST_FULL = re.compile(
    r"(?i)\b(?:around|about|approximately)\s*\d{1,4}(?:\.\d+)?\s*"
    r"(?:nm|nautical miles?)\s*"
    r"(?:(?:NNE|ENE|ESE|SSE|SSW|WSW|WNW|NNW|NE|SE|SW|NW|N|E|S|W)\b|"
    r"northeast|northwest|southeast|southwest|north|south|east|west)?"
    r"\s*of\s*[^.,]+(?:,\s*[A-Z][^.,]+)?"
)
LEAD_TIME_ART = re.compile(
    r"^[\s–—-]*\d{3,4}\s*(?:UTC|LT)?\s*:?\s*(?:Posn|Position)?\s*:?\s*", re.I
)
PLACE_RENAMES = {"Chittagong": "Chattogram"}


def strip_leading_coords(text: str) -> str:
    return COORDS_PAIR_RE.sub("", text, count=1).strip(" ,") if isinstance(text, str) else text


def split_place_if_glued(place: str) -> tuple[str, str]:
    """Split ``"Singapore Straits Two robbers ..."`` into place and narration head."""
    s = strip_leading_coords((place or "").strip(" '\""))
    if not s:
        return "", ""
    dot = s.find(". ")
    if dot != -1:
        return s[:dot].strip().rstrip("."), s[dot + 2 :].strip()
    m = NARRATION_START_RE.search(s)
    if m and m.start() > 0:
        return s[: m.start()].rstrip(" ,."), s[m.start() :].strip()
    return s, ""


def place_is_ok(place: str) -> bool:
    """QC check: does the value look like a maritime place? (Used as a flag, not a filter.)"""
    if not isinstance(place, str) or not place.strip():
        return False
    s = place.strip()
    if TIME_RE.match(s) or POSN_RE.search(s) or len(s) < 3:
        return False
    return bool(GEO_WORD.search(s) or "Singapore Straits" in s or DIST_PHRASE.search(s))


def extract_after_coords(raw_line: str) -> str:
    if not isinstance(raw_line, str):
        return ""
    m = re.search(
        r"\b\d{1,3}[:\s]\d{1,2}(?:\.\d+)?\s*[NSEW]\s*[–\-]\s*"
        r"\d{1,3}[:\s]\d{1,2}(?:\.\d+)?\s*[NSEW]\s*,\s*",
        raw_line,
    )
    if not m:
        return ""
    return raw_line[m.end() :].split(".", 1)[0].strip(" ,")


def clean_place_value(place: str, narration: str = "") -> str:
    s = (place or "").strip(" '\"")
    s = LEAD_TIME_ART.sub("", s).strip(" ,")
    s = strip_leading_coords(s)
    if TIME_RE.match(s):
        s = ""
    s = re.sub(r"\s+", " ", s).strip(" ,")
    s = s.replace("Singapore, Straits", "Singapore Straits")
    for word in ("Anchorage", "Port", "Terminal", "Jetty", "Straits", "Harbour"):
        s = re.sub(rf"\b{word}\b", word, s, flags=re.I)
    s, _ = split_place_if_glued(s)
    if not place_is_ok(s) and "singapore straits" in (narration or "").lower():
        return "Singapore Straits"
    return s


def salvage_place(place: str, narration: str = "", raw: str = "") -> str:
    """Best-effort repair of the place field; falls back to the cleaned original."""
    p = clean_place_value(place, narration)
    n, raw = narration or "", raw or ""
    if place_is_ok(p):
        return p
    for src in (raw, n):
        m = DIST_FULL.search(src)
        if m:
            cand, _ = split_place_if_glued(m.group(0).strip(" ,."))
            if place_is_ok(cand):
                return cand
    for src in (raw, n):
        cand = extract_after_coords(normalize_line(src))
        if cand:
            cand = clean_place_value(split_place_if_glued(cand)[0], n or raw)
            if place_is_ok(cand):
                return cand
    if re.search(r"(?i)\baround\s*\d{1,4}\b", p) and re.search(r"(?i)\b(?:nm|nautical miles?)\b",
                                                              n + raw):
        p2 = split_place_if_glued(re.sub(r"(?i)\baround\s*(\d{1,4})\b", r"Around \1nm", p))[0]
        if place_is_ok(p2):
            return p2
    if "singapore straits" in (n + raw).lower():
        return "Singapore Straits"
    head, _ = split_place_if_glued(p)
    return head or p


# --------------------------------------------------------------------------------------
# enrichment (heuristic labels from the narration)
# --------------------------------------------------------------------------------------
INCIDENT_PATTERNS = {
    "Hijacked": [r"\bhijack(?:ed)?\b", r"\bcommandeer(?:ed)?\b"],
    "Fired upon": [r"\bfired\b", r"\bshots?\b", r"\bopened\s+fire\b"],
    "Attempted": [r"\battempt(?:ed)?\b"],
    "Boarding": [r"\bboard(?:ed|ing)?\b", r"\brobbers?\b", r"\bunauthori[sz]ed\s+persons?\b",
                 r"\btheft\b", r"\bstole\b", r"\bsteal(?:ing)?\b"],
    "Suspicious": [r"\bsuspicious\b", r"\bapproached\b"],
}
INCIDENT_REGEX = {k: re.compile("|".join(v), re.I) for k, v in INCIDENT_PATTERNS.items()}

VESSEL_PATTERNS = {
    "tanker": [r"\b(?:product|oil|chemical|gas|lpg|lng)\s+tanker\b", r"\btanker\b"],
    "bulk carrier": [r"\bbulk\s+carrier\b"],
    "container": [r"\bcontainer\s+(?:ship|vessel)\b"],
    "general cargo": [r"\bgeneral\s+cargo(?:\s+ship)?\b", r"\bcargo\s+ship\b"],
    "fishing vessel": [r"\bfishing\s+vessel\b", r"\btrawler\b"],
    "yacht/pleasure": [r"\b(?:super)?yacht\b", r"\bpleasure\s+craft\b"],
    "tug/barge": [r"\btug\b", r"\bbarge\b"],
    "passenger": [r"\bpassenger\s+(?:ship|vessel)\b", r"\bferry\b"],
    "offshore support": [r"\boffshore\s+support\b", r"\bsupply\s+vessel\b", r"\bOSV\b"],
}
VESSEL_REGEX = {k: re.compile("|".join(v), re.I) for k, v in VESSEL_PATTERNS.items()}


def classify_incident_type(text: str) -> str:
    """First match in priority order Hijacked > Fired upon > Attempted > Boarding > Suspicious."""
    if not isinstance(text, str) or not text:
        return "Unknown"
    for label, rx in INCIDENT_REGEX.items():
        if rx.search(text):
            return label
    return "Unknown"


def infer_vessel_type(text: str) -> tuple[str, str]:
    """Vessel category guessed from free text (the map has no vessel column)."""
    if isinstance(text, str):
        for label, rx in VESSEL_REGEX.items():
            if rx.search(text):
                return label, "inferred_from_text"
    return "Unknown", "none"


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------
def parse_incident(line: str) -> Incident | None:
    """Parse one details string; ``None`` if no pattern matches."""
    s = normalize_line(line)
    for name, pat in PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        g = m.groupdict()
        place, glue = split_place_if_glued((g.get("place") or "").strip())
        narration = (g.get("narration") or "").strip()
        if glue:
            sep = " " if glue.endswith(".") else ". "
            narration = (glue + sep + narration).strip()
        lat_tok = (g.get("lat") or "").strip()
        lon_tok = (g.get("lon") or "").strip()
        return Incident(
            date=g["date"].strip(),
            time=(g.get("time") or "").strip(),
            lat_token=lat_tok,
            lon_token=lon_tok,
            place=place,
            narration=narration,
            lat_decimal=dms_to_decimal(lat_tok) if lat_tok else None,
            lon_decimal=dms_to_decimal(lon_tok) if lon_tok else None,
            raw_details=line,
            parse_source=name,
        )
    return None


def parse_lines(
    lines: list[str], year: int | None = None, *, renames: dict | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Parse, repair places and enrich. Returns ``(incidents, failed_lines)``."""
    renames = PLACE_RENAMES if renames is None else renames
    records, failed = [], []
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            continue
        inc = parse_incident(line)
        if inc is None:
            failed.append(line)
        else:
            records.append(asdict(inc))
    if failed:
        log.warning("%d of %d lines did not parse", len(failed), len(failed) + len(records))
    df = pd.DataFrame.from_records(records, columns=[f for f in Incident.__dataclass_fields__])
    if df.empty:
        return pd.DataFrame(columns=INCIDENT_COLUMNS), failed
    df["place"] = [
        salvage_place(p, n, r) for p, n, r in zip(df["place"], df["narration"], df["raw_details"],
                                                  strict=True)
    ]
    for old, new in renames.items():
        df["place"] = df["place"].str.replace(old, new, regex=False)
    df["year"] = year if year is not None else df["date"].str[-4:].astype(int)
    df["incident_type"] = df["narration"].map(classify_incident_type)
    vt = [infer_vessel_type(f"{n} {r}") for n, r in zip(df["narration"], df["raw_details"],
                                                         strict=True)]
    df["vessel_type_inferred"] = [v[0] for v in vt]
    df["vessel_type_source"] = [v[1] for v in vt]
    df["place_ok"] = df["place"].map(place_is_ok)
    return df[INCIDENT_COLUMNS], failed


# --------------------------------------------------------------------------------------
# HTML table extraction (works on a saved page or Selenium's page_source)
# --------------------------------------------------------------------------------------
class _TableCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self._depth = 0
        self._section: str | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_tag: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self.tables.append({"id": dict(attrs).get("id") or "", "head": [], "rows": []})
        elif self._depth != 1:
            return
        elif tag in ("thead", "tbody", "tfoot"):
            self._section = tag
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell, self._cell_tag = [], tag
        elif tag == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth -= 1
            return
        if self._depth != 1:
            return
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"[ \t\r\f\v]+", " ", "".join(self._cell)).strip()
            if self._row is not None:
                self._row.append((self._cell_tag, text))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            table = self.tables[-1]
            if self._section == "thead" or all(t == "th" for t, _ in self._row):
                table["head"] = [txt for _, txt in self._row]
            else:
                table["rows"].append([txt for _, txt in self._row])
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


DETAIL_HEADERS = ("details", "narration", "incident", "description")


def extract_details_from_html(html: str, table_id: str | None = None) -> list[str]:
    """Return the incident details strings from the map's data table.

    The table is chosen by ``table_id``, else the first ``wpgmza_table_*`` table, else the
    first table. The details column is found by header name; if no header matches, every
    third cell (index 2, 5, 8, ...) is used, as on the 2023-2025 pages.
    """
    parser = _TableCollector()
    parser.feed(html)
    tables = parser.tables
    if not tables:
        return []
    table = None
    if table_id:
        table = next((t for t in tables if t["id"] == table_id), None)
        if table is None:
            raise ValueError(f"No table with id {table_id!r} in page.")
    if table is None:
        table = next((t for t in tables if t["id"].startswith("wpgmza_table_")), tables[0])
    header = [h.strip().lower() for h in table["head"]]
    idx = next((header.index(c) for c in DETAIL_HEADERS if c in header), None)
    if idx is not None:
        return [r[idx] for r in table["rows"] if len(r) > idx and r[idx]]
    cells = [c for r in table["rows"] for c in r]
    return [cells[i] for i in range(2, len(cells), 3) if cells[i]]
