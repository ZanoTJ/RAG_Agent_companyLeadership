"""SQLite storage: companies, leaders (one row per unique person), the supporting
evidence for each person's role, source passages, and a web-search cache."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import entities

DB_PATH = "leadership.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL UNIQUE,
    scraped_at  TEXT,
    resolved_version   INTEGER DEFAULT 0,  -- entity-resolution logic version last applied
    removed_unverified INTEGER DEFAULT 0,  -- people dropped for lack of acceptable evidence
    collection_report  TEXT                -- JSON: sources/candidates found, rejected, accepted
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
    is_current_title INTEGER DEFAULT 0,
    ended            INTEGER DEFAULT 0  -- 1 = evidence that this role ended (hidden; used
                                        --     so re-checks keep cancelling older claims)
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

# Columns added after the first release: (table, column, type).
_ADDED_COLUMNS = [
    ("leaders", "source_urls", "TEXT"), ("leaders", "evidence", "TEXT"),
    ("leaders", "aliases", "TEXT"), ("leaders", "as_of", "TEXT"),
    ("role_history", "name", "TEXT"), ("role_history", "source_type", "TEXT"),
    ("companies", "resolved_version", "INTEGER DEFAULT 0"),
    ("companies", "removed_unverified", "INTEGER DEFAULT 0"),
    ("sources", "source_type", "TEXT"), ("sources", "published", "TEXT"),
    ("sources", "origin", "TEXT"), ("leaders", "conflict", "TEXT"),
    ("companies", "collection_report", "TEXT"), ("role_history", "ended", "INTEGER DEFAULT 0"),
]
# Columns removed: status is no longer stored (freshness is computed from dates).
_DROPPED_COLUMNS = [("role_history", "status"), ("leaders", "status")]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def backup_db(db_path: str | None = None, label: str = "backup") -> Path | None:
    """Copy the database to backups/<name>-<label>-<timestamp>.db (SQLite backup API)."""
    src_path = Path(db_path or DB_PATH)
    if not src_path.exists():
        return None
    dest_dir = src_path.parent / "backups"
    dest_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dest = dest_dir / f"{src_path.stem}-{label}-{stamp}.db"
    src, dst = sqlite3.connect(src_path), sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return dest


def _columns(conn, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def init_db(db_path: str | None = None) -> None:
    """Create tables and migrate older databases. Safe to rerun; backs up the
    database before any destructive change."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        needs_drop = any(col in _columns(conn, t) for t, col in _DROPPED_COLUMNS)
    if needs_drop:
        backup_db(db_path, label="pre-migration")
    with get_conn(db_path) as conn:
        for table, col, typ in _ADDED_COLUMNS:
            if col not in _columns(conn, table):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        if "status" in _columns(conn, "role_history"):
            # Former positions are no longer stored. People left with no evidence at
            # all (every stored role was former) are removed too.
            had_roles = {r[0] for r in conn.execute("SELECT DISTINCT leader_id FROM role_history")}
            conn.execute("DELETE FROM role_history WHERE status = 'former'")
            still = {r[0] for r in conn.execute("SELECT DISTINCT leader_id FROM role_history")}
            for lid in had_roles - still:
                conn.execute("DELETE FROM leaders WHERE id = ?", (lid,))
            conn.execute("ALTER TABLE role_history DROP COLUMN status")
        if "status" in _columns(conn, "leaders"):
            conn.execute("ALTER TABLE leaders DROP COLUMN status")


# ---------------------------------------------------------------- companies

def upsert_company(name: str, url: str, db_path: str | None = None) -> int:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO companies (name, url, scraped_at) VALUES (?, ?, ?)
               ON CONFLICT(url) DO UPDATE SET name = excluded.name,
                                              scraped_at = excluded.scraped_at""",
            (name, url, _now()),
        )
        return conn.execute("SELECT id FROM companies WHERE url = ?", (url,)).fetchone()["id"]


def get_company(company_id: int, db_path: str | None = None) -> dict | None:
    with get_conn(db_path) as conn:
        r = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
        return dict(r) if r else None


def collected_on(company: dict | None) -> date | None:
    if company and company.get("scraped_at"):
        return datetime.fromisoformat(company["scraped_at"]).date()
    return None


def set_company_fields(company_id: int, db_path: str | None = None, **fields) -> None:
    allowed = {"resolved_version", "removed_unverified", "collection_report"}
    assert set(fields) <= allowed, fields
    with get_conn(db_path) as conn:
        for k, v in fields.items():
            conn.execute(f"UPDATE companies SET {k} = ? WHERE id = ?", (v, company_id))


def set_resolved_version(company_id: int, version: int, db_path: str | None = None) -> None:
    set_company_fields(company_id, db_path, resolved_version=version)


def list_companies(db_path: str | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT c.*, COUNT(l.id) AS leader_count
               FROM companies c LEFT JOIN leaders l ON l.company_id = c.id
               GROUP BY c.id ORDER BY c.scraped_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def clear_company_data(company_id: int, db_path: str | None = None) -> None:
    """Remove previous results so a re-collection doesn't leave stale people behind."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM leaders WHERE company_id = ?", (company_id,))
        conn.execute("DELETE FROM sources WHERE company_id = ?", (company_id,))
        conn.execute("DELETE FROM search_cache WHERE company_id = ?", (company_id,))
        conn.execute("UPDATE companies SET removed_unverified = 0 WHERE id = ?", (company_id,))


# ---------------------------------------------------------------- sources

def save_source(company_id: int, url: str, content: str, source_type: str = "",
                published: str = "", origin: str = "collect",
                db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO sources (company_id, url, content, fetched_at, source_type,
                                    published, origin)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(company_id, url) DO UPDATE SET
                   content = excluded.content, fetched_at = excluded.fetched_at,
                   source_type = excluded.source_type, published = excluded.published""",
            (company_id, url, content, _now(), source_type, published or "", origin),
        )


def get_sources(company_id: int, db_path: str | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT url, content, source_type, published, origin, fetched_at
               FROM sources WHERE company_id = ?""", (company_id,)).fetchall()
        return [dict(r) for r in rows]


def update_source(company_id: int, url: str, db_path: str | None = None, **fields) -> None:
    allowed = {"content", "source_type"}
    assert set(fields) <= allowed, fields
    with get_conn(db_path) as conn:
        for k, v in fields.items():
            conn.execute(f"UPDATE sources SET {k} = ? WHERE company_id = ? AND url = ?",
                         (v, company_id, url))


def delete_sources(company_id: int, urls: list[str], db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.executemany("DELETE FROM sources WHERE company_id = ? AND url = ?",
                         [(company_id, u) for u in urls])


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

def save_people(company_id: int, people: list[dict], db_path: str | None = None) -> int:
    """Store resolved people (from entities.resolve_people) and their supporting
    evidence. People carrying an `id` keep it. Returns the number stored."""
    stored = 0
    with get_conn(db_path) as conn:
        for p in people:
            if not p.get("name") or not p.get("source_url"):
                continue
            cur = conn.execute(
                """INSERT OR IGNORE INTO leaders
                   (id, company_id, name, title, department, seniority, source_url,
                    source_urls, evidence, aliases, as_of, conflict)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p.get("id"), company_id, p["name"].strip(), (p.get("title") or "").strip(),
                 (p.get("department") or "").strip(), p.get("seniority") or "other",
                 p["source_url"], "\n".join(p.get("source_urls") or [p["source_url"]]),
                 (p.get("evidence") or "").strip(), json.dumps(p.get("aliases") or []),
                 p.get("as_of") or "", p.get("conflict") or ""),
            )
            if not cur.rowcount:
                continue
            stored += 1
            conn.executemany(
                """INSERT INTO role_history (leader_id, name, title, as_of, source_url,
                                             source_tier, source_type, evidence,
                                             is_current_title, ended)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(cur.lastrowid, r.get("name") or p["name"], r["title"], r.get("as_of") or "",
                  r["source_url"], r.get("source_tier"), r.get("source_type") or "",
                  r.get("evidence") or "", int(bool(r.get("is_current_title"))), ended)
                 for ended, rows in ((0, p.get("roles", [])), (1, p.get("ended", [])))
                 for r in rows],
            )
    return stored


def replace_people(company_id: int, people: list[dict], db_path: str | None = None) -> int:
    """Swap a company's people for a re-resolved set (evidence rows cascade).
    People carrying an existing `id` keep it, so person IDs stay stable."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM leaders WHERE company_id = ?", (company_id,))
    return save_people(company_id, people, db_path)


def get_leaders(company_id: int, db_path: str | None = None) -> list[dict]:
    """One row per unique person, by seniority. `person_id` is internal (citations,
    counting); `may_be_outdated` is computed from the evidence date, not stored."""
    order = """CASE seniority WHEN 'executive' THEN 0 WHEN 'vp' THEN 1
               WHEN 'department_head' THEN 2 ELSE 3 END"""
    company = get_company(company_id, db_path)
    with get_conn(db_path) as conn:
        rows = conn.execute(
            f"""SELECT id AS person_id, name, aliases, title, department, seniority,
                       source_url, COALESCE(source_urls, source_url) AS source_urls,
                       evidence, as_of, COALESCE(conflict, '') AS conflict
                FROM leaders WHERE company_id = ? ORDER BY {order}, name""",
            (company_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["aliases"] = json.loads(d["aliases"]) if d["aliases"] else []
        d["may_be_outdated"] = entities.may_be_outdated(d["as_of"] or "", collected_on(company))
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
               WHERE l.company_id = ? AND COALESCE(r.ended, 0) = 0
               ORDER BY l.name, r.is_current_title DESC, r.source_tier, r.as_of DESC""",
            (company_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def load_resolution_records(company_id: int, db_path: str | None = None) -> list[dict]:
    """Rebuild per-source records from stored data so entity resolution can be re-run
    without re-collecting. Uses the evidence rows when present; otherwise (data
    stored before evidence rows existed) one record per leader plus its extra sources."""
    records = []
    with get_conn(db_path) as conn:
        leaders = conn.execute("SELECT * FROM leaders WHERE company_id = ?",
                               (company_id,)).fetchall()
        for l in leaders:
            base = {"person_id": l["id"], "department": l["department"] or "",
                    "seniority": l["seniority"] or "other", "status": "current",
                    "aliases": json.loads(l["aliases"]) if l["aliases"] else []}
            all_urls = (l["source_urls"] or l["source_url"] or "").split("\n")
            roles = conn.execute("SELECT * FROM role_history WHERE leader_id = ?",
                                 (l["id"],)).fetchall()
            if roles:
                for r in roles:
                    records.append({**base, "name": r["name"] or l["name"], "title": r["title"],
                                    "status": "former" if r["ended"] else "current",
                                    "as_of": r["as_of"], "source_url": r["source_url"],
                                    "source_type": r["source_type"] or "",
                                    "evidence": r["evidence"]})
                covered = {r["source_url"] for r in roles}
            else:
                records.append({**base, "name": l["name"], "title": l["title"], "as_of": "",
                                "source_url": l["source_url"], "source_type": "",
                                "evidence": l["evidence"]})
                covered = {l["source_url"]}
            records[-1]["extra_sources"] = [u for u in all_urls if u and u not in covered]
    return records
