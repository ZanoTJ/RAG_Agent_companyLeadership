"""Entity resolution: turn per-source leader records into unique people.

Stage 1 (deterministic): names are normalized (accents, punctuation, honorifics,
credentials, middle initials removed) and records with the same normalized
first + last name are merged.

Stage 2 (LLM-assisted, only for ambiguous clusters): clusters with similar
surnames AND either related first names (nickname/prefix: "Dan"/"Daniel") or an
overlapping, specific role ("Jen-Hsun"/"Jensen" Huang, both "founder ... CEO")
are sent to a resolver together with their titles, evidence quotes, and sources.
A shared surname or a shared title alone never makes a candidate. They merge only if
the resolver confirms they are the same person. With no resolver, or on any
error, they stay separate: a similar name alone never merges two people.

Each resulting person keeps every alias and supporting source. Outdated roles are
not stored: a role is dropped when a credible source says it has ended (and that
source is at least as new as the claim); people with only former roles are
excluded. The current title is chosen by source authority first, then recency, and
is flagged "may be outdated" unless backed by a live company page or a source dated
within STALE_AFTER_DAYS. No status is stored; freshness is computed from dates.

This module makes no network calls; the resolver is injected (see rag.py).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Callable
from urllib.parse import urlparse

# ---------------------------------------------------------------- names

HONORIFICS = {"dr", "mr", "mrs", "ms", "mx", "prof", "sir", "dame"}
CREDENTIALS = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "mba", "cpa", "cfa", "esq",
               "jd", "cfp", "pe", "pmp", "frm"}

# Common English nicknames whose short form is NOT a prefix of the full name.
# (Prefix forms like Dan/Daniel, Vlad/Vladimir, Chris/Christopher are handled by
# a generic rule.) Matches here only make a pair a *candidate* for LLM review.
_NICKNAMES = {
    "robert": {"bob", "bobby", "rob", "robbie", "bert"},
    "william": {"bill", "billy", "will", "willy", "liam"},
    "james": {"jim", "jimmy", "jamie"},
    "michael": {"mike", "mikey", "mick"},
    "richard": {"rick", "ricky", "rich", "dick"},
    "thomas": {"tom", "tommy"},
    "anthony": {"tony"},
    "edward": {"ed", "eddie", "ted", "ned"},
    "john": {"jack", "johnny", "jon"},
    "katherine": {"kate", "katie", "kathy", "kat"},
    "catherine": {"cathy", "kate", "katie", "cat"},
    "elizabeth": {"liz", "beth", "betsy", "eliza", "lisa"},
    "margaret": {"peggy", "maggie", "meg"},
    "alexander": {"sasha", "alex", "xander"},
    "alexandra": {"sasha", "alex", "lexi"},
    "stephen": {"steve", "stevie"},
    "jeffrey": {"jeff"},
    "geoffrey": {"jeff", "geoff"},
    "andrew": {"andy", "drew"},
    "nicholas": {"nick", "nicky"},
    "patrick": {"pat", "paddy"},
    "patricia": {"pat", "patty", "trish"},
    "susan": {"sue", "susie"},
    "joseph": {"joe", "joey"},
    "lawrence": {"larry"},
    "henry": {"hank", "harry"},
    "charles": {"chuck", "charlie"},
    "david": {"dave"},
    "rebecca": {"becky", "becca"},
    "nathaniel": {"nate", "nat"},
    "nathan": {"nate"},
    "zachary": {"zach", "zack"},
    "jennifer": {"jen", "jenny"},
    "deborah": {"deb", "debbie"},
    "theodore": {"ted", "teddy", "theo"},
    "francis": {"frank"},
    "frederick": {"fred", "freddie"},
    "gregory": {"greg"},
    "samuel": {"sam"},
    "benjamin": {"ben"},
    "matthew": {"matt"},
}
_NICK_GROUPS: dict[str, set[str]] = {}
for _full, _nicks in _NICKNAMES.items():
    for _n in _nicks | {_full}:
        _NICK_GROUPS.setdefault(_n, set()).update(_nicks | {_full})


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def display_name(name: str) -> str:
    """'Stephanie Guild, CFA' -> 'Stephanie Guild' (keeps original spelling/case)."""
    name = re.sub(r"\s+", " ", name).strip()
    parts = [p.strip() for p in name.split(",")]
    while len(parts) > 1 and _ascii(parts[-1]).lower().replace(".", "") in CREDENTIALS:
        parts.pop()
    return ", ".join(parts)


@dataclass(frozen=True)
class NameParts:
    first: str
    last: str
    nicknames: tuple[str, ...]   # from parentheses, e.g. "Mariano-Florentino (Tino) Cuellar"

    @property
    def key(self) -> str:
        return f"{self.first} {self.last}"


def parse_name(name: str) -> NameParts | None:
    """Normalize a person's name. Returns None if there's no first AND last name."""
    nicks = tuple(_ascii(n).lower().strip() for n in re.findall(r"\((.*?)\)", name))
    name = re.sub(r"\(.*?\)", " ", display_name(name))
    name = _ascii(name).lower().replace("'", "")
    name = re.sub(r"[^a-z\s-]", " ", name).replace("-", " ")
    tokens = [t for t in name.split()
              if t not in HONORIFICS and t not in CREDENTIALS and len(t) > 1]
    if len(tokens) < 2:
        return None
    return NameParts(tokens[0], tokens[-1], nicks)


def first_names_related(a: str, b: str) -> bool:
    """Same, nickname of, or prefix of (>= 3 letters) - e.g. dan/daniel, bob/robert."""
    if a == b:
        return True
    if b in _NICK_GROUPS.get(a, ()):
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 3 and long_.startswith(short)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def surnames_similar(a: str, b: str) -> bool:
    """Equal, or a likely spelling variant: 1 edit (len >= 4) / 2 edits (len >= 8)."""
    if a == b:
        return True
    n = min(len(a), len(b))
    d = _edit_distance(a, b)
    return (n >= 4 and d <= 1) or (n >= 8 and d <= 2)


# Words too common in titles to indicate the same role on their own.
_GENERIC_ROLE_WORDS = {
    "and", "of", "the", "for", "to", "at", "in", "a", "an", "co",
    "director", "directors", "board", "member", "independent", "non", "executive",
    "senior", "sr", "vice", "president", "vp", "svp", "evp", "chief", "officer", "head",
    "lead", "manager", "general", "group", "deputy", "associate", "assistant", "office",
}
_CSUITE = re.compile(r"\bchief (executive|financial|operating|technology|marketing|"
                     r"information|legal|people|product|revenue|security|"
                     r"accounting|investment|scientific|science) officer\b")


def role_tokens(title: str) -> set[str]:
    """Distinctive words of a title: 'Founder, President and Chief Executive Officer'
    -> {'founder', 'ceo'}; 'Director' -> set()."""
    t = _ascii(title).lower().replace("co-founder", "founder").replace("cofounder", "founder")
    t = _CSUITE.sub(lambda m: f"c{m[1][0]}o {m[1]}", t)  # keep the function word too
    return {w for w in re.findall(r"[a-z]+", t) if w not in _GENERIC_ROLE_WORDS and len(w) > 1}


def roles_overlap(titles_a, titles_b) -> bool:
    ta = set().union(*(role_tokens(t) for t in titles_a)) if titles_a else set()
    tb = set().union(*(role_tokens(t) for t in titles_b)) if titles_b else set()
    return bool(ta & tb)


def is_candidate_pair(a: NameParts, b: NameParts, titles_a=(), titles_b=()) -> bool:
    """Similar enough to deserve an evidence-based check; never merged on this alone.

    Requires similar surnames AND (related first names OR an overlapping specific
    role). Same surname with unrelated roles, or the same title with a different
    surname, is never a candidate.
    """
    if a.key == b.key:
        return False  # already merged deterministically
    if not surnames_similar(a.last, b.last):
        return False
    firsts = first_names_related(a.first, b.first) or a.first in b.nicknames \
        or b.first in a.nicknames
    return firsts or roles_overlap(titles_a, titles_b)


# ---------------------------------------------------------------- sources & dates

_ARTICLE_PATH = re.compile(
    r"/(blog|news|newsroom|press|press-releases?|articles?|posts?|stories|insights|"
    r"announcements?|media)(/|$)", re.I)
TIER_LABEL = {1: "Official / filing", 2: "News / reference / profile"}
RESOLUTION_VERSION = 3  # bump when resolution logic changes; stored data is re-resolved
OFFICIAL_TYPES = {"official", "regulatory"}


def registrable_domain(url_or_host: str) -> str:
    """'https://investors.acme.co.uk/x' -> 'acme.co.uk'; 'www.acme.com' -> 'acme.com'."""
    host = urlparse(url_or_host).netloc if "//" in url_or_host else url_or_host
    labels = host.lower().split(":")[0].strip(".").split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in {
            "co", "com", "org", "net", "ac", "gov", "edu"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _is_gov(url: str) -> bool:
    return urlparse(url).netloc.lower().split(":")[0].endswith(".gov")


def source_tier(url: str, company_domain: str, source_type: str = "") -> int:
    """1 = company's own domain (any subdomain), government filing, or a source the
    evaluator classified official/regulatory; 2 = other accepted public sources."""
    if (registrable_domain(url) == registrable_domain(company_domain) or _is_gov(url)
            or source_type in OFFICIAL_TYPES):
        return 1
    return 2


def name_authority(url: str, company_domain: str, source_type: str = "") -> int:
    """For choosing a display name: the company's own site (how it refers to its
    people) > filings (often legal names) > other public sources."""
    if registrable_domain(url) == registrable_domain(company_domain):
        return 0
    return source_tier(url, company_domain, source_type)


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_date(s: str | None) -> tuple[int, int, int]:
    """Lenient: '2025-04-25', 'April 2025', 'Apr 25, 2025', '2025' -> sortable tuple.
    Returns (0, 0, 0) when no year is found."""
    if not s:
        return (0, 0, 0)
    s = s.strip().lower()
    m = re.search(r"\b(19|20)(\d{2})-(\d{1,2})(?:-(\d{1,2}))?", s)
    if m:
        return (int(m[1] + m[2]), int(m[3]), int(m[4] or 0))
    y = re.search(r"\b(19|20)\d{2}\b", s)
    if not y:
        return (0, 0, 0)
    mon = next((n for name, n in _MONTHS.items() if re.search(rf"\b{name}", s)), 0)
    day = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b(?!\d)", s.replace(y[0], ""))
    return (int(y[0]), mon, int(day[1]) if mon and day else 0)


def format_date(t: tuple[int, int, int]) -> str:
    y, m, d = t
    return "" if not y else f"{y}" + (f"-{m:02d}" if m else "") + (f"-{d:02d}" if d else "")


# ---------------------------------------------------------------- resolution

SENIORITY_RANK = {"executive": 0, "board": 1, "vp": 2, "department_head": 3, "other": 4}
STALE_AFTER_DAYS = 730  # evidence older than this can't confirm a role is still current

# resolver(cluster_a, cluster_b) -> (same_person, reason). Each cluster is a list
# of records (name, title, as_of, evidence, source_url).
Resolver = Callable[[list[dict], list[dict]], tuple[bool, str]]


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _ascii(t).lower()).strip()


def _record_date(r: dict, fetched: tuple[int, int, int]) -> tuple[int, int, int]:
    """Effective date of a record's evidence.

    A live (non-article) page on the company's own site describes the present, so a
    role it lists is evidence as of the fetch date - even if it mentions a start date
    ("on our board since 2021"). Otherwise use the date the source states, if any."""
    live_company_page = (registrable_domain(r["source_url"]) == r["_company"]
                         and not _ARTICLE_PATH.search(urlparse(r["source_url"]).path))
    if live_company_page and r.get("status") != "former":
        return fetched
    return parse_date(r.get("as_of"))


def is_recent(d: tuple[int, int, int], fetched: tuple[int, int, int]) -> bool:
    if d == (0, 0, 0):
        return False
    then = date(d[0], d[1] or 1, d[2] or 1)
    return (date(*fetched) - then).days <= STALE_AFTER_DAYS


def may_be_outdated(as_of: str, collected_on: date | None) -> bool:
    """A person's current title is shown "(may be outdated)" unless its supporting
    evidence is a live company page or a source dated within STALE_AFTER_DAYS of
    collection. `as_of` is the stored effective evidence date."""
    fetched = (collected_on or date.today()).timetuple()[:3]
    return not is_recent(parse_date(as_of), fetched)


def _supersedes(former: dict, claim: dict) -> bool:
    """A 'former' record (e.g. "previously served as Co-CEO") cancels a current claim
    of the same role when it is at least as new as that claim."""
    fa, ca = role_tokens(former["title"]), role_tokens(claim["title"])
    same_role = (_norm_title(former["title"]) == _norm_title(claim["title"])
                 or (fa and fa <= ca))
    return bool(same_role) and former["date"] != (0, 0, 0) and former["date"] >= claim["date"]


def resolve_people(records: list[dict], company_domain: str, company_name: str = "",
                   resolver: Resolver | None = None, fetched_on: date | None = None,
                   log: Callable[[str], None] = lambda _m: None,
                   stats: dict | None = None) -> list[dict]:
    """Group per-source records into unique people (see module docstring).

    Records may carry `status` ("current"/"former"/"unknown", from extraction; used
    only here, never stored), `source_type` (from source evaluation), `person_id`
    (existing DB id - the merged person keeps the lowest), `aliases`, and
    `extra_sources` (supporting URLs without a title). People whose only roles are
    former are excluded. `stats` gets checked/merged/failed/excluded_former counts."""
    stats = stats if stats is not None else {}
    stats.update(checked=0, merged=0, failed=0, excluded_former=0)
    fetched = (fetched_on or date.today()).timetuple()[:3]
    company = registrable_domain(company_domain)
    at_company = re.compile(rf"\s+(at|@)\s+{re.escape(company_name)}\.?$", re.I) \
        if company_name else None

    # --- Stage 1: clean records and group by exact normalized name.
    clusters: dict[str, list[dict]] = {}
    parts_of: dict[str, NameParts] = {}
    for rec in records:
        parts = parse_name(rec.get("name") or "")
        if not parts or not rec.get("source_url"):
            continue  # unverifiable (first name / initials only) or unsourced
        if rec.get("source_type") == "people_data_aggregator":
            continue  # never evidence (defensive; such sources are filtered earlier)
        r = dict(rec)
        r["name"] = display_name(r["name"])
        r["title"] = (r.get("title") or "").strip()
        if at_company:
            r["title"] = at_company.sub("", r["title"])  # "CTO at Acme" -> "CTO"
        r["status"] = r.get("status") or "unknown"
        r["_company"] = company
        r["tier"] = source_tier(r["source_url"], company_domain, r.get("source_type", ""))
        r["authority"] = name_authority(r["source_url"], company_domain, r.get("source_type", ""))
        r["date"] = _record_date(r, fetched)
        clusters.setdefault(parts.key, []).append(r)
        parts_of.setdefault(parts.key, parts)

    # --- Stage 2: evidence-based check for similar-but-different names.
    parent = {k: k for k in clusters}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    keys = sorted(clusters)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if find(a) == find(b) or not is_candidate_pair(
                    parts_of[a], parts_of[b],
                    [r["title"] for r in clusters[a]], [r["title"] for r in clusters[b]]):
                continue
            label = f"{clusters[a][0]['name']} / {clusters[b][0]['name']}"
            if resolver is None:
                log(f"Possible duplicate kept separate (no resolver): {label}")
                continue
            stats["checked"] += 1
            try:
                same, reason = resolver(clusters[a], clusters[b])
            except Exception as e:  # conservative: never merge on failure
                stats["failed"] += 1
                log(f"Entity check failed for {label}: {e} - kept separate")
                continue
            log(f"{'Merged' if same else 'Kept separate'}: {label} ({reason})")
            if same:
                stats["merged"] += 1
                parent[find(b)] = find(a)

    merged: dict[str, list[dict]] = {}
    for k, recs in clusters.items():
        merged.setdefault(find(k), []).extend(recs)
    people = []
    for recs in merged.values():
        person = _consolidate(recs, fetched)
        if person is None:
            stats["excluded_former"] += 1
            log(f"Excluded (only former roles): {recs[0]['name']}")
        else:
            people.append(person)
    return people


def _consolidate(recs: list[dict], fetched: tuple[int, int, int]) -> dict | None:
    """Build one person from all their records, or None if every role is former."""
    formers = [r for r in recs if r["status"] == "former"]
    # Outdated roles are dropped, not stored: explicit "former" records, and current
    # claims of a role that an equally new or newer source says has ended.
    active = [r for r in recs if r["status"] != "former"
              and not any(_supersedes(f, r) for f in formers)]
    if not active:
        return None

    def rank(r: dict):
        return (r["tier"], tuple(-x for x in r["date"]),
                0 if r["status"] == "current" else 1,
                SENIORITY_RANK.get(r.get("seniority"), 4), -len(r["title"]))

    active.sort(key=rank)
    best = active[0]

    # Canonical spelling: the name the company itself uses, then filings, then other
    # sources; within a level, the most frequently used spelling. Spellings from
    # dropped (former) records are still kept as aliases.
    spellings: dict[str, list[dict]] = {}
    for r in recs:
        spellings.setdefault(r["name"], []).append(r)

    def name_rank(n: str):
        best_auth = min(x["authority"] for x in spellings[n])
        return (best_auth, -sum(x["authority"] == best_auth for x in spellings[n]),
                -len(spellings[n]), n)
    canonical = min(spellings, key=name_rank)
    all_names = set(spellings).union(*(set(r.get("aliases") or []) for r in recs))

    roles, seen = [], set()
    for r in active:
        k = (_norm_title(r["title"]), r["source_url"])
        if k in seen:
            continue
        seen.add(k)
        roles.append({
            # only dates actually stated by the source (not the fetch-date fallback)
            "name": r["name"], "title": r["title"],
            "as_of": format_date(parse_date(r.get("as_of"))),
            "source_url": r["source_url"], "source_tier": r["tier"],
            "source_type": r.get("source_type") or "",
            "evidence": r.get("evidence") or "", "is_current_title": r is best,
        })

    return {
        "id": min((r["person_id"] for r in recs if r.get("person_id")), default=None),
        "name": canonical,
        "aliases": sorted(n for n in all_names if n != canonical),
        "title": best["title"],
        "as_of": format_date(best["date"]),  # effective evidence date (internal)
        "may_be_outdated": not is_recent(best["date"], fetched),
        "department": best.get("department") or next(
            (r["department"] for r in active if r.get("department")), ""),
        "seniority": best.get("seniority") or "other",
        "source_url": best["source_url"],
        "source_urls": list(dict.fromkeys(
            [r["source_url"] for r in active] + [u for r in active
                                                 for u in r.get("extra_sources") or []])),
        "evidence": best.get("evidence") or "",
        "roles": roles,
    }
