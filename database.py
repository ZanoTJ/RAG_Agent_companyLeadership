"""SQLite storage for companies, leaders (one row per unique person), their role
histories, and the source pages they came from."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "leadership.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL UNIQUE,
    scraped_at  TEXT,
    resolved_version INTEGER DEFAULT 0   -- entity-resolution logic version last applied
);

CREATE TABLE IF NOT EXISTS leaders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    title       TEXT,
    department  TEXT,
    seniority   TEXT,          -- executive | vp | department_head | board | other
    source_url  TEXT NOT NULL,  -- primary (best) source
    source_urls TEXT,           -- all supporting sources, newline-separated
    evidence    TEXT,           -- verbatim quote from the source showing the role
    aliases     TEXT,           -- JSON list of other spellings of this person's name
    status      TEXT,           -- current | unverified | former
    as_of       TEXT,           -- date the current title is supported as of
    UNIQUE (company_id, name, title)
);

-- Every title reported for a person, by source (current and historical).
CREATE TABLE IF NOT EXISTS role_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    leader_id        INTEGER NOT NULL REFERENCES leaders(id) ON DELETE CASCADE,
    name             TEXT,      -- the person's name as written by this source
    title            TEXT,
    status           TEXT,      -- current | former | unknown (as stated by the source)
    as_of            TEXT,      -- date stated by the source, if any
    source_url       TEXT NOT NULL,
    source_tier      INTEGER,   -- 1 company/official, 2 news/reference, 3 aggregator/social
    evidence         TEXT,
    is_current_title INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    url         TEXT NOT NULL,
    content     TEXT,
    fetched_at  TEXT,
    UNIQUE (company_id, url)
);
"""


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


def init_db(db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        # Migrate databases created before these columns existed.
        for table, col, typ in [
            ("leaders", "source_urls", "TEXT"), ("leaders", "evidence", "TEXT"),
            ("leaders", "aliases", "TEXT"), ("leaders", "status", "TEXT"),
            ("leaders", "as_of", "TEXT"), ("role_history", "name", "TEXT"),
            ("companies", "resolved_version", "INTEGER DEFAULT 0"),
        ]:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def upsert_company(name: str, url: str, db_path: str | None = None) -> int:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO companies (name, url, scraped_at) VALUES (?, ?, ?)
               ON CONFLICT(url) DO UPDATE SET name = excluded.name,
                                              scraped_at = excluded.scraped_at""",
            (name, url, _now()),
        )
        return conn.execute("SELECT id FROM companies WHERE url = ?", (url,)).fetchone()["id"]


def clear_company_data(company_id: int, db_path: str | None = None) -> None:
    """Remove previous results so a re-scrape doesn't leave stale people behind."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM leaders WHERE company_id = ?", (company_id,))
        conn.execute("DELETE FROM sources WHERE company_id = ?", (company_id,))


def save_source(company_id: int, url: str, content: str, db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO sources (company_id, url, content, fetched_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(company_id, url) DO UPDATE SET content = excluded.content,
                                                          fetched_at = excluded.fetched_at""",
            (company_id, url, content, _now()),
        )


def save_people(company_id: int, people: list[dict], db_path: str | None = None) -> int:
    """Store resolved people (from entities.resolve_people) and their role histories.
    Returns the number of people stored."""
    stored = 0
    with get_conn(db_path) as conn:
        for p in people:
            if not p.get("name") or not p.get("source_url"):
                continue
            cur = conn.execute(
                """INSERT OR IGNORE INTO leaders
                   (id, company_id, name, title, department, seniority, source_url,
                    source_urls, evidence, aliases, status, as_of)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p.get("id"), company_id, p["name"].strip(), (p.get("title") or "").strip(),
                 (p.get("department") or "").strip(), p.get("seniority") or "other",
                 p["source_url"], "\n".join(p.get("source_urls") or [p["source_url"]]),
                 (p.get("evidence") or "").strip(), json.dumps(p.get("aliases") or []),
                 p.get("status") or "current", p.get("as_of") or ""),
            )
            if not cur.rowcount:
                continue
            stored += 1
            conn.executemany(
                """INSERT INTO role_history (leader_id, name, title, status, as_of,
                                             source_url, source_tier, evidence,
                                             is_current_title)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(cur.lastrowid, r.get("name") or p["name"], r["title"], r["status"], r.get("as_of") or "",
                  r["source_url"], r.get("source_tier"), r.get("evidence") or "",
                  int(bool(r.get("is_current_title")))) for r in p.get("roles", [])],
            )
    return stored


def replace_people(company_id: int, people: list[dict], db_path: str | None = None) -> int:
    """Swap a company's people for a re-resolved set (role history cascades).
    People carrying an existing `id` keep it, so person IDs stay stable."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM leaders WHERE company_id = ?", (company_id,))
    return save_people(company_id, people, db_path)


def set_resolved_version(company_id: int, version: int, db_path: str | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute("UPDATE companies SET resolved_version = ? WHERE id = ?",
                     (version, company_id))


def get_company(company_id: int, db_path: str | None = None) -> dict | None:
    with get_conn(db_path) as conn:
        r = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
        return dict(r) if r else None


def load_resolution_records(company_id: int, db_path: str | None = None) -> list[dict]:
    """Rebuild per-source records from stored data so entity resolution can be re-run
    without re-collecting. Uses role_history when present; otherwise (data stored
    before role histories existed) one record per leader row plus its extra sources."""
    records = []
    with get_conn(db_path) as conn:
        leaders = conn.execute("SELECT * FROM leaders WHERE company_id = ?",
                               (company_id,)).fetchall()
        for l in leaders:
            base = {"person_id": l["id"], "department": l["department"] or "",
                    "seniority": l["seniority"] or "other",
                    "aliases": json.loads(l["aliases"]) if l["aliases"] else []}
            all_urls = (l["source_urls"] or l["source_url"] or "").split("\n")
            roles = conn.execute("SELECT * FROM role_history WHERE leader_id = ?",
                                 (l["id"],)).fetchall()
            if roles:
                for r in roles:
                    records.append({**base, "name": r["name"] or l["name"], "title": r["title"],
                                    "status": r["status"], "as_of": r["as_of"],
                                    "source_url": r["source_url"], "evidence": r["evidence"]})
                covered = {r["source_url"] for r in roles}
            else:
                # Legacy rows: earlier extraction only kept roles stated as current.
                records.append({**base, "name": l["name"], "title": l["title"],
                                "status": "current", "as_of": "",
                                "source_url": l["source_url"], "evidence": l["evidence"]})
                covered = {l["source_url"]}
            records[-1]["extra_sources"] = [u for u in all_urls if u and u not in covered]
    return records


def list_companies(db_path: str | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT c.*, COUNT(l.id) AS leader_count
               FROM companies c LEFT JOIN leaders l ON l.company_id = c.id
               GROUP BY c.id ORDER BY c.scraped_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_leaders(company_id: int, db_path: str | None = None) -> list[dict]:
    """One row per unique person: current people first, then by seniority."""
    order = """CASE status WHEN 'former' THEN 2 WHEN 'unverified' THEN 1 ELSE 0 END,
               CASE seniority WHEN 'executive' THEN 0 WHEN 'board' THEN 1
               WHEN 'vp' THEN 2 WHEN 'department_head' THEN 3 ELSE 4 END"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            f"""SELECT id AS person_id, name, aliases, title, COALESCE(status, 'current') AS status, as_of,
                       department, seniority, source_url,
                       COALESCE(source_urls, source_url) AS source_urls, evidence
                FROM leaders WHERE company_id = ? ORDER BY {order}, name""",
            (company_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["aliases"] = json.loads(d["aliases"]) if d["aliases"] else []
        out.append(d)
    return out


def get_role_history(company_id: int, db_path: str | None = None) -> list[dict]:
    """All reported titles for a company's people, most authoritative first."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT l.id AS person_id, l.name, r.name AS name_in_source, r.title, r.status, r.as_of, r.source_url,
                      r.source_tier, r.evidence, r.is_current_title
               FROM role_history r JOIN leaders l ON l.id = r.leader_id
               WHERE l.company_id = ?
               ORDER BY l.name, r.is_current_title DESC, r.source_tier, r.as_of DESC""",
            (company_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_sources(company_id: int, db_path: str | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT url, content FROM sources WHERE company_id = ?", (company_id,)
        ).fetchall()
        return [dict(r) for r in rows]
