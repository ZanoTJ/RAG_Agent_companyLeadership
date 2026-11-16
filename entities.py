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

Each resulting person keeps every alias and supporting source. Only leadership
roles count (C-level, VP/SVP/EVP, head of a department - see leadership_level);
board-only members and other titles are excluded. Outdated roles are not stored:
a role is dropped when a credible source says it has ended or a newer source names
someone else in the same one-person role (e.g. CTO); people left without a current
leadership role are excluded, and so are people missing from the company's current
roster pages whose evidence is old. Freshness is computed from dates.

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
RESOLUTION_VERSION = 4  # bump when resolution logic changes; stored data is re-resolved
OFFICIAL_TYPES = {"official", "regulatory"}

# ---------------------------------------------------------------- leadership level

LEVELS = ("executive", "vp", "department_head")
LEVEL_RANK = {"executive": 0, "vp": 1, "department_head": 2}
_C_ABBREV = re.compile(r"\bc(e|f|o|t|m|i|is|s|r|p|l|hr|c|a|d|b)o\b")
_PLAIN_VP = re.compile(r"\b(vice president|vp)\b")
_SENIOR_VP = re.compile(r"\b(svp|evp)\b")
_HEAD = re.compile(r"\bhead\b")
_DIRECTOR = re.compile(r"\b(director|general manager|gm)\b")
_BOARD_WORDS = re.compile(r"\b(board|independent|non executive|lead director|chair\w*|trustee)\b")
_NOT_CSUITE = re.compile(r"\bchief of staff\b|\beditor in chief\b|\bcommander in chief\b")
_NON_JOB = re.compile(r"\b(founder|board|director|chair\w*|investor|advisor|adviser|"
                      r"partner|member|trustee|owner)\b")


def _level_text(title: str) -> str:
    t = _ascii(title or "").lower().replace("-", " ")
    t = re.sub(r"\b(senior|sr\.?) (vice president|vp)\b", "svp", t)
    t = re.sub(r"\bexecutive (vice president|vp)\b", "evp", t)
    return re.sub(r"\s+", " ", t)


def leadership_level(title: str, label: str = "") -> str | None:
    """Deterministic level of a title: 'executive' (C-level or President), 'vp'
    (Vice President, SVP, EVP), 'department_head', or None (not a leadership role:
    board-only, founder-only, Chief of Staff, ...).

    A plain "Vice President and Chief X Officer" is VP rank. Director-level titles
    (Director, Senior Director, General Manager) qualify as department heads only
    when extraction found that the person leads a department or function
    (`label == "department_head"`); "Head of ..." always does."""
    t = _level_text(title)
    if not t:
        return None
    if _PLAIN_VP.search(t):
        return "vp"
    csuite = re.sub(_NOT_CSUITE, " ", t)
    if re.search(r"\bchief\b", csuite) or _C_ABBREV.search(csuite) or \
            re.search(r"\bpresident\b", csuite):
        return "executive"
    if _SENIOR_VP.search(t):
        return "vp"
    if _HEAD.search(t):
        return "department_head"
    if _DIRECTOR.search(t) and not _BOARD_WORDS.search(t) and label == "department_head":
        return "department_head"
    return None


def _is_other_job(title: str) -> bool:
    """A non-leadership job title ("Solution Consultant"), as opposed to a board,
    founder, or director-level title that can coexist with a leadership role."""
    t = _level_text(title)
    return bool(t) and leadership_level(title) is None and not _NON_JOB.search(t)


# One-person roles: a newer source naming someone else in the role supersedes the claim.
_SINGLE_ROLE_LONG = {
    "chief executive officer": "ceo", "chief financial officer": "cfo",
    "chief operating officer": "coo", "chief technology officer": "cto",
    "chief marketing officer": "cmo", "chief information officer": "cio",
    "chief information security officer": "ciso", "chief legal officer": "clo",
    "general counsel": "general counsel",
}
_SINGLE_ROLES = set(_SINGLE_ROLE_LONG.values())
_TITLE_FILLER = {"founder", "co founder", "cofounder", "chairman", "chair", "chairwoman",
                 "chairperson", "president", "svp", "evp", "chairman of the board", ""}


def single_holder_roles(title: str) -> set[str]:
    """Company-wide one-person roles in a title: 'Co-founder / CTO' -> {'cto'}.
    Scoped titles ('Crypto CFO', 'COO, Robinhood Financial') and Co-CEOs give none."""
    t = _level_text(title)
    for long_form, short in _SINGLE_ROLE_LONG.items():
        t = t.replace(long_form, short)
    parts = [p.strip() for p in re.split(r",|&|/|;|\+|\band\b", t)]
    if any(p not in _SINGLE_ROLES and p not in _TITLE_FILLER for p in parts):
        return set()
    return {p for p in parts if p in _SINGLE_ROLES}


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
    """1 = company's own domain (any subdomain) or a government filing - decided by
    the URL only, never by an LLM label; 2 = other accepted public sources."""
    if registrable_domain(url) == registrable_domain(company_domain) or _is_gov(url):
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

STALE_AFTER_DAYS = 730  # evidence older than this can't confirm a role is still current

# resolver(cluster_a, cluster_b) -> (same_person, reason). Each cluster is a list
# of records (name, title, as_of, evidence, source_url).
Resolver = Callable[[list[dict], list[dict]], tuple[bool, str]]


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _ascii(t).lower()).strip()


def _record_date(r: dict, fetched: tuple[int, int, int]) -> tuple[int, int, int]:
    """Effective date of a record's evidence.

    Evidence of the present counts as of the fetch date, even without a stated date:
    - a live (non-article) page on the company's own site ("on our board since 2021");
    - a person's own public profile (the only person on that page) that presents the
      role as current, e.g. a LinkedIn position "to present".
    Otherwise use the date the source states, if any."""
    if r.get("status") == "former":
        return parse_date(r.get("as_of"))
    if _is_live_company_page(r) or (r.get("_own_profile") and r.get("status") == "current"):
        return fetched
    return parse_date(r.get("as_of"))


def _scope_of(title: str, other: str, company: str) -> bool:
    """True if `other` is `title` plus a scope other than the company itself:
    'President and COO' -> 'President and COO of TradePMR' (not '... of Robinhood')."""
    base, longer = _norm_title(title), _norm_title(other)
    if not base or not longer.startswith(base + " "):
        return False
    extra = set(longer[len(base):].split()) - {"of", "at", "by", "the", "for", "inc", "llc"}
    if extra & _TITLE_WORDS:
        return False  # more roles ("... Co-Founder and President"), not a scope
    return bool(extra - {company.split(".")[0]})


_TITLE_WORDS = {"and", "founder", "co", "cofounder", "president", "chairman", "chair", "ceo",
                "officer", "chief", "director", "vp", "svp", "evp", "head", "secretary",
                "treasurer", "manager", "partner", "member", "board"}


def _is_own_profile(r: dict, parts: NameParts) -> bool:
    """The person's own public profile (e.g. linkedin.com/in/jane-doe, janedoe.com),
    not a company page or a post: the URL itself carries the person's name."""
    if r.get("source_type") != "professional_profile":
        return False
    u = urlparse(r["source_url"])
    if _ARTICLE_PATH.search(u.path):
        return False
    compact = re.sub(r"[^a-z]", "", (u.netloc + u.path).lower())
    last = parts.last.replace(" ", "")
    return last in compact and (parts.first in compact or len(last) >= 5)


def _is_live_company_page(r: dict) -> bool:
    return (registrable_domain(r["source_url"]) == r["_company"]
            and not _ARTICLE_PATH.search(urlparse(r["source_url"]).path))

def _supersedes(former: dict, claim: dict) -> bool:
    """A 'former' record (e.g. "previously served as Co-CEO") cancels a current claim
    of the same role when it is at least as new as that claim."""
    fa, ca = role_tokens(former["title"]), role_tokens(claim["title"])
    same_role = (_norm_title(former["title"]) == _norm_title(claim["title"])
                 or (fa and fa <= ca)
                 # "Chief Technology Officer" ended -> a "CTO" claim ended too
                 or single_holder_roles(former["title"]) & single_holder_roles(claim["title"]))
    return bool(same_role) and former["date"] != (0, 0, 0) and former["date"] >= claim["date"]


def resolve_people(records: list[dict], company_domain: str, company_name: str = "",
                   resolver: Resolver | None = None, fetched_on: date | None = None,
                   log: Callable[[str], None] = lambda _m: None,
                   stats: dict | None = None) -> list[dict]:
    """Group per-source records into unique people (see module docstring).

    Records may carry `status` ("current"/"former"/"unknown", from extraction; used
    only here, never stored), `source_type` (from source evaluation), `person_id`
    (existing DB id - the merged person keeps the lowest), `aliases`, and
    `extra_sources` (supporting URLs without a title). `stats` gets
    checked/merged/failed/excluded_former counts and `excluded`: one
    {name, title, reason} entry per excluded person."""
    stats = stats if stats is not None else {}
    stats.update(checked=0, merged=0, failed=0, excluded_former=0, excluded=[])
    fetched = (fetched_on or date.today()).timetuple()[:3]
    company = registrable_domain(company_domain)
    at_company = re.compile(rf"\s+(at|@)\s+{re.escape(company_name)}\.?$", re.I) \
        if company_name else None

    # A live company page naming ROSTER_MIN+ people is a current roster.
    names_on: dict[str, set[str]] = {}
    for rec in records:
        parts = parse_name(rec.get("name") or "")
        if parts and rec.get("source_url"):
            names_on.setdefault(rec["source_url"], set()).add(parts.key)
    has_roster = any(len(names) >= ROSTER_MIN and _is_live_company_page(
        {"source_url": u, "_company": company}) for u, names in names_on.items())

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
        r["_own_profile"] = _is_own_profile(r, parts)
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

    def exclude(recs: list[dict], reason: str) -> None:
        if reason == REASON_FORMER:
            stats["excluded_former"] += 1
        stats["excluded"].append({"name": recs[0]["name"], "title": recs[0]["title"],
                                  "reason": reason})
        log(f"Excluded {recs[0]['name']}: {reason}")

    built: list[tuple[list[dict], dict]] = []
    for recs in merged.values():
        person, reason = _consolidate(recs, fetched)
        if person is None:
            exclude(recs, reason)
        elif has_roster and not person["_tier1"] and _is_stale(person["_newest"], fetched):
            exclude(recs, REASON_NOT_ON_ROSTER)
        else:
            built.append((recs, person))

    # --- Stage 3: one-person roles (CEO, CTO, ...) claimed by different people.
    holders: dict[str, list[int]] = {}
    for i, (_, p) in enumerate(built):
        for role in p["_single_roles"]:
            holders.setdefault(role, []).append(i)
    dropped: set[int] = set()
    for role, idx in holders.items():
        idx = [i for i in idx if i not in dropped]
        if len(idx) < 2:
            continue
        idx.sort(key=lambda i: built[i][1]["_single_roles"][role], reverse=True)
        top, second = (built[i][1]["_single_roles"][role] for i in idx[:2])
        decided = top > second and (second != (0, 0, 0))
        label = role.upper() if role != "general counsel" else "General Counsel"
        if not decided:
            names = ", ".join(built[i][1]["name"] for i in idx)
            for i in idx:
                built[i][1]["conflict"] = f"Sources disagree on who is {label}: {names}"
            log(f"Unresolved: several people listed as {label} ({names})")
            continue
        winner = built[idx[0]][1]
        for i in idx[1:]:
            recs = [r for r in built[i][0] if role not in single_holder_roles(r["title"])]
            person, reason = _consolidate(recs, fetched) if recs else (None, "")
            if person is None:
                exclude(built[i][0], f"{label} role superseded: {winner['name']} is listed "
                                     f"as {label} in newer evidence")
                dropped.add(i)
            else:
                built[i] = (recs, person)
                log(f"{built[i][1]['name']}: {label} claim superseded by {winner['name']}")

    people = []
    for i, (_, p) in enumerate(built):
        if i not in dropped:
            people.append({k: v for k, v in p.items() if not k.startswith("_")})
    return people


ROSTER_MIN = 3  # people listed on a live company page for it to count as a roster
REASON_FORMER = "only former roles"
REASON_NO_LEADERSHIP = "no qualifying leadership title (board-only, founder-only, or other role)"
REASON_NOT_ON_ROSTER = ("not on the company's current leadership pages, no official source, "
                        f"and no evidence from the last {STALE_AFTER_DAYS} days")


def _is_stale(d: tuple[int, int, int], fetched: tuple[int, int, int]) -> bool:
    """Undated, or older than STALE_AFTER_DAYS before the fetch date."""
    if d == (0, 0, 0):
        return True
    newest = date(d[0], min(d[1], 12) or 1, min(d[2], 28) or 1)
    return (date(*fetched) - newest).days > STALE_AFTER_DAYS


def _consolidate(recs: list[dict], fetched: tuple[int, int, int]) -> tuple[dict | None, str]:
    """Build one person from all their records, or (None, reason) if they hold no
    current leadership role."""
    formers = [r for r in recs if r["status"] == "former"]
    # Outdated roles are dropped, not stored: explicit "former" records, and current
    # claims of a role that an equally new or newer source says has ended.
    active = [r for r in recs if r["status"] != "former"
              and not any(_supersedes(f, r) for f in formers)]
    if not active:
        return None, REASON_FORMER
    for r in active:
        r["level"] = leadership_level(r["title"], r.get("seniority") or "")
    qualifying = [r for r in active if r["level"]]
    if not qualifying:
        return None, REASON_NO_LEADERSHIP

    def rank(r: dict):
        return (r["tier"], tuple(-x for x in r["date"]),
                0 if r["status"] == "current" else 1,
                LEVEL_RANK[r["level"]], -len(r["title"]))

    qualifying.sort(key=rank)
    best = qualifying[0]
    # The same title with an added scope ("President and COO" vs. "President and COO of
    # TradePMR by Robinhood") is the same role stated more precisely: show the precise
    # one, dated by the newer of the two sources.
    scoped = [r for r in qualifying if _scope_of(best["title"], r["title"], r["_company"])]
    date_ = best["date"]
    if scoped:
        precise = min(scoped, key=rank)
        date_ = max(best["date"], precise["date"])
        best = precise
    active.sort(key=lambda r: r is not best)
    # Profile fields: from tier-1 sources first, then the newest.
    by_trust = sorted(active, key=lambda r: (r["tier"], tuple(-x for x in r["date"])))
    profile = {f: next((r[f] for r in by_trust if r.get(f)), "")
               for f in ("location", "profile_url", "bio")}

    # A current non-leadership job title in an equally new or newer source
    # (e.g. profile headline "Solution Consultant" vs. "Head of ...") is a discrepancy.
    conflict = ""
    contra = next((r for r in active if r["status"] == "current" and r["date"] >= best["date"]
                   and _is_other_job(r["title"])), None)
    if contra:
        conflict = (f'Sources disagree: also listed as "{contra["title"]}" '
                    f'({contra["source_url"]})')

    single: dict[str, tuple[int, int, int]] = {}
    for r in qualifying:
        if any(_scope_of(r["title"], o["title"], o["_company"]) for o in qualifying):
            continue  # a scoped version of this title exists: not a company-wide role
        for role in single_holder_roles(r["title"]):
            single[role] = max(single.get(role, (0, 0, 0)), r["date"])

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
    # Evidence that a role ended is kept (hidden) so re-checking stored data later
    # still cancels the older claims; it is never shown as a role.
    ended = [{"name": r["name"], "title": r["title"],
              "as_of": format_date(parse_date(r.get("as_of"))),
              "source_url": r["source_url"], "source_tier": r["tier"],
              "source_type": r.get("source_type") or "", "evidence": r.get("evidence") or ""}
             for r in formers if r["date"] != (0, 0, 0)]

    return {
        "id": min((r["person_id"] for r in recs if r.get("person_id")), default=None),
        "name": canonical,
        "aliases": sorted(n for n in all_names if n != canonical),
        "title": best["title"],
        "as_of": format_date(date_),  # effective evidence date (internal)
        "conflict": conflict,
        "department": best.get("department") or next(
            (r["department"] for r in active if r.get("department")), ""),
        "seniority": best["level"],
        "source_url": best["source_url"],
        "source_urls": list(dict.fromkeys(
            [r["source_url"] for r in active] + [u for r in active
                                                 for u in r.get("extra_sources") or []])),
        "evidence": best.get("evidence") or "",
        **profile,
        "roles": roles,
        "ended": ended,
        # internal, used by resolve_people and stripped from the result
        "_date": date_,
        "_newest": max(r["date"] for r in active),
        "_tier1": any(r["tier"] == 1 for r in active),
        "_single_roles": single,
    }, ""
