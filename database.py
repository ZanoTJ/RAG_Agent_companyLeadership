"""SQLite storage for companies, leaders, and the source pages they came from."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "leadership.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL UNIQUE,
    scraped_at  TEXT
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
    UNIQUE (company_id, name, title)
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
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        # Migrate databases created before these columns existed.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(leaders)")}
        for col in ("source_urls", "evidence"):
            if col not in cols:
                conn.execute(f"ALTER TABLE leaders ADD COLUMN {col} TEXT")


def upsert_company(name: str, url: str, db_path: str = DB_PATH) -> int:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO companies (name, url, scraped_at) VALUES (?, ?, ?)
               ON CONFLICT(url) DO UPDATE SET name = excluded.name,
                                              scraped_at = excluded.scraped_at""",
            (name, url, _now()),
        )
        return conn.execute("SELECT id FROM companies WHERE url = ?", (url,)).fetchone()["id"]


def clear_company_data(company_id: int, db_path: str = DB_PATH) -> None:
    """Remove previous results so a re-scrape doesn't leave stale people behind."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM leaders WHERE company_id = ?", (company_id,))
        conn.execute("DELETE FROM sources WHERE company_id = ?", (company_id,))


def save_source(company_id: int, url: str, content: str, db_path: str = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO sources (company_id, url, content, fetched_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(company_id, url) DO UPDATE SET content = excluded.content,
                                                          fetched_at = excluded.fetched_at""",
            (company_id, url, content, _now()),
        )


def save_leaders(company_id: int, leaders: list[dict], db_path: str = DB_PATH) -> int:
    rows = [
        (
            company_id,
            p["name"].strip(),
            (p.get("title") or "").strip(),
            (p.get("department") or "").strip(),
            (p.get("seniority") or "other").strip(),
            p["source_url"],
            "\n".join(p.get("source_urls") or [p["source_url"]]),
            (p.get("evidence") or "").strip(),
        )
        for p in leaders
        if p.get("name") and p.get("source_url")
    ]
    with get_conn(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            """INSERT OR IGNORE INTO leaders
               (company_id, name, title, department, seniority, source_url,
                source_urls, evidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return conn.total_changes - before


def list_companies(db_path: str = DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """SELECT c.*, COUNT(l.id) AS leader_count
               FROM companies c LEFT JOIN leaders l ON l.company_id = c.id
               GROUP BY c.id ORDER BY c.scraped_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_leaders(company_id: int, db_path: str = DB_PATH) -> list[dict]:
    order = """CASE seniority WHEN 'executive' THEN 0 WHEN 'board' THEN 1
               WHEN 'vp' THEN 2 WHEN 'department_head' THEN 3 ELSE 4 END"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            f"""SELECT name, title, department, seniority, source_url,
                       COALESCE(source_urls, source_url) AS source_urls, evidence
                FROM leaders WHERE company_id = ? ORDER BY {order}, name""",
            (company_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_sources(company_id: int, db_path: str = DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT url, content FROM sources WHERE company_id = ?", (company_id,)
        ).fetchall()
        return [dict(r) for r in rows]
