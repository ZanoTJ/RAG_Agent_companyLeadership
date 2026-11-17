"""SQLite storage: companies, leaders (one row per unique person), the supporting
evidence for each person's role, source passages, and a web-search cache."""

import calendar
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from entities import parse_date

DB_PATH = str(Path(__file__).parent / "leadership.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL UNIQUE,
    scraped_at  TEXT               -- set only when a collection is stored successfully
);

CREATE TABLE IF NOT EXISTS leaders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,   -- canonical person ID (internal)
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    title       TEXT,
    department  TEXT,
    seniority   TEXT,          -- executive | vp | department_head (from the title)
    source_url  TEXT NOT NULL,  -- primary (best) source
    source_urls TEXT,           -- all supporting sources, newline-separated
    evidence    TEXT,           -- verbatim quote from the source showing the role
    aliases     TEXT,           -- JSON list of other spellings of this person's name
    as_of       TEXT,           -- effective date of the title's evidence (internal)
    conflict    TEXT,           -- unresolved disagreement between sources, if any
    location    TEXT,           -- professional base (never a home address)
    profile_url TEXT,           -- public profile, e.g. LinkedIn
    bio         TEXT,           -- one short sentence from the sources
    outdated    INTEGER DEFAULT 0,  -- bool: as_of is more than 1 year before today
    UNIQUE (company_id, name, title)
);

-- Supporting evidence: each source that reports a current role for a person.
CREATE TABLE IF NOT EXISTS role_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    leader_id        INTEGER NOT NULL REFERENCES leaders(id) ON DELETE CASCADE,
    name             TEXT,      -- the person's name as written by this source
    title            TEXT,
    as_of            TEXT,      -- date stated by the source, if any (internal)
    source_url       TEXT NOT NULL,
    source_tier      INTEGER,   -- 1 official/filing, 2 other accepted public source
    source_type      TEXT,      -- official | regulatory | news | reference | professional_profile
    evidence         TEXT,
    is_current_title INTEGER DEFAULT 0
);

-- Accepted sources only, stored as relevant passages rather than whole pages.
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    url         TEXT NOT NULL,
    content     TEXT,
    fetched_at  TEXT,
    source_type TEXT,           -- result of source evaluation
    published   TEXT,           -- publication date, when known
    origin      TEXT,           -- collect | web (answer-time web search)
    UNIQUE (company_id, url)
);

-- Answer-time web searches, so repeated questions don't repeat identical searches.
CREATE TABLE IF NOT EXISTS search_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    cache_key   TEXT NOT NULL,
    query       TEXT,
    result_count INTEGER,
    created_at  TEXT,
    UNIQUE (company_id, cache_key)
);
"""

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


OUTDATED_AFTER_DAYS = 365


def is_outdated(as_of: str | None, today: date | None = None) -> bool:
    """True when `as_of` ('2025', '2025-04' or '2025-04-25') is more than a year
    before `today`. Undated evidence can't be compared, so it is not flagged.
    A partial date is read as the latest day it could mean ('2025' -> 2025-12-31),
    so a person is never flagged on a guess."""
    y, m, d = parse_date(as_of)
    if not y:
        return False
    m = min(m, 12) or 12
    last_day = calendar.monthrange(y, m)[1]
    as_of_date = date(y, m, min(d, last_day) or last_day)
    return ((today or date.today()) - as_of_date).days > OUTDATED_AFTER_DAYS


@contextmanager
def get_conn(db_path: str | None = None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create the tables if they don't exist. Safe to rerun."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        # Existing databases predate the `outdated` column: add it in place.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(leaders)")}
        if "outdated" not in cols:
            conn.execute("ALTER TABLE leaders ADD COLUMN outdated INTEGER DEFAULT 0")
        _refresh_outdated(conn)


def _refresh_outdated(conn, company_id: int | None = None) -> None:
    """Recompute `outdated` against today's date, since it changes as time passes."""
    sql = "SELECT id, as_of, outdated FROM leaders"
    rows = conn.execute(sql + (" WHERE company_id = ?" if company_id is not None else ""),
                        () if company_id is None else (company_id,)).fetchall()
    fresh = [(int(is_outdated(r["as_of"])), r["id"], r["outdated"] or 0) for r in rows]
    changed = [(new, rid) for new, rid, old in fresh if new != old]
    conn.executemany("UPDATE leaders SET outdated = ? WHERE id = ?", changed)


# ---------------------------------------------------------------- companies

def save_collection(name: str, url: str, people: list[dict], sources: list[dict],
                    db_path: str | None = None) -> int:
    """Replace a company's people, sources and search cache with a new collection in
    ONE transaction: if anything fails, the previous data stays untouched.
    `scraped_at` is set only here, i.e. only when the write succeeds."""
    with get_conn(db_path) as conn:
        conn.execute("""INSERT INTO companies (name, url) VALUES (?, ?)
                        ON CONFLICT(url) DO UPDATE SET name = excluded.name""", (name, url))
        cid = conn.execute("SELECT id FROM companies WHERE url = ?", (url,)).fetchone()["id"]
        for table in ("leaders", "sources", "search_cache"):
            conn.execute(f"DELETE FROM {table} WHERE company_id = ?", (cid,))
        _insert_people(conn, cid, people)
        conn.executemany(_SOURCE_SQL, [(cid, s["url"], s["content"], _now(), s.get("source_type", ""),
                                        s.get("published") or "", "collect") for s in sources])
        conn.execute("UPDATE companies SET scraped_at = ? WHERE id = ?", (_now(), cid))
    return cid


def get_company(company_id: int, db_path: str | None = None) -> dict | None:
    with get_conn(db_path) as conn:
        r = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
        return dict(r) if r else None


def collected_on(company: dict | None) -> date | None:
    if company and company.get("scraped_at"):
        return datetime.fromisoformat(company["scraped_at"]).date()
    return None


def list_companies(db_path: str | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT c.*, COUNT(l.id) AS leader_count
               FROM companies c LEFT JOIN leaders l ON l.company_id = c.id
               GROUP BY c.id ORDER BY c.scraped_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- sources

_SOURCE_SQL = """INSERT INTO sources (company_id, url, content, fetched_at, source_type,
                                      published, origin)
                 VALUES (?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(company_id, url) DO UPDATE SET
                     content = excluded.content, fetched_at = excluded.fetched_at,
                     source_type = excluded.source_type, published = excluded.published"""


def save_source(company_id: int, url: str, content: str, source_type: str = "",
                published: str = "", origin: str = "collect",
                db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(_SOURCE_SQL, (company_id, url, content, _now(), source_type,
                                   published or "", origin))


def get_sources(company_id: int, db_path: str | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT url, content, source_type, published, origin, fetched_at
               FROM sources WHERE company_id = ?""", (company_id,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- search cache

def get_cached_search(company_id: int, cache_key: str, max_age_days: int = 7,
                      db_path: str | None = None) -> dict | None:
    with get_conn(db_path) as conn:
        r = conn.execute("SELECT * FROM search_cache WHERE company_id = ? AND cache_key = ?",
                         (company_id, cache_key)).fetchone()
    if not r:
        return None
    age = datetime.now(timezone.utc) - datetime.fromisoformat(r["created_at"])
    return dict(r) if age.days < max_age_days else None


def save_search(company_id: int, cache_key: str, query: str, result_count: int,
                db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO search_cache (company_id, cache_key, query, result_count, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(company_id, cache_key) DO UPDATE SET query = excluded.query,
                   result_count = excluded.result_count, created_at = excluded.created_at""",
            (company_id, cache_key, query, result_count, _now()))


# ---------------------------------------------------------------- people

def _insert_people(conn, company_id: int, people: list[dict]) -> int:
    """Store resolved people (from entities.resolve_people) and their supporting
    evidence. Returns the number stored."""
    stored = 0
    for p in people:
        if not p.get("name") or not p.get("source_url"):
            continue
        cur = conn.execute(
            """INSERT OR IGNORE INTO leaders
               (company_id, name, title, department, seniority, source_url,
                source_urls, evidence, aliases, as_of, conflict, location, profile_url, bio,
                outdated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (company_id, p["name"].strip(), (p.get("title") or "").strip(),
             (p.get("department") or "").strip(), p.get("seniority") or "other",
             p["source_url"], "\n".join(p.get("source_urls") or [p["source_url"]]),
             (p.get("evidence") or "").strip(), json.dumps(p.get("aliases") or []),
             p.get("as_of") or "", p.get("conflict") or "", p.get("location") or "",
             p.get("profile_url") or "", p.get("bio") or "",
             int(is_outdated(p.get("as_of")))),
        )
        if not cur.rowcount:
            continue
        stored += 1
        conn.executemany(
            """INSERT INTO role_history (leader_id, name, title, as_of, source_url,
                                         source_tier, source_type, evidence,
                                         is_current_title)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(cur.lastrowid, r.get("name") or p["name"], r["title"], r.get("as_of") or "",
              r["source_url"], r.get("source_tier"), r.get("source_type") or "",
              r.get("evidence") or "", int(bool(r.get("is_current_title"))))
             for r in p.get("roles", [])],
        )
    return stored


def get_leaders(company_id: int, db_path: str | None = None) -> list[dict]:
    """One row per unique person, by seniority. `person_id` is internal (citations,
    counting)."""
    order = """CASE seniority WHEN 'executive' THEN 0 WHEN 'vp' THEN 1
               WHEN 'department_head' THEN 2 ELSE 3 END"""
    with get_conn(db_path) as conn:
        _refresh_outdated(conn, company_id)
        rows = conn.execute(
            f"""SELECT id AS person_id, name, aliases, title, department, seniority,
                       source_url, COALESCE(source_urls, source_url) AS source_urls,
                       evidence, as_of, COALESCE(outdated, 0) AS outdated,
                       COALESCE(conflict, '') AS conflict,
                       COALESCE(location, '') AS location,
                       COALESCE(profile_url, '') AS profile_url, COALESCE(bio, '') AS bio
                FROM leaders WHERE company_id = ? ORDER BY {order}, name""",
            (company_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["aliases"] = json.loads(d["aliases"]) if d["aliases"] else []
        d["outdated"] = bool(d["outdated"])
        out.append(d)
    return out


def get_role_history(company_id: int, db_path: str | None = None) -> list[dict]:
    """Supporting evidence for every person, most authoritative first."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT l.id AS person_id, l.name, r.name AS name_in_source, r.title, r.as_of,
                      r.source_url, r.source_tier, r.source_type, r.evidence,
                      r.is_current_title
               FROM role_history r JOIN leaders l ON l.id = r.leader_id
               WHERE l.company_id = ?
               ORDER BY l.name, r.is_current_title DESC, r.source_tier, r.as_of DESC""",
            (company_id,),
        ).fetchall()
        return [dict(r) for r in rows]
