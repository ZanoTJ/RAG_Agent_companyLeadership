"""Source evaluation, collection filtering, database migration, simplified views,
deterministic counts, and hybrid retrieval. Offline: every LLM / search call is faked."""

import sqlite3
from datetime import date

import pytest

import entities
import source_policy
import views
from scraper import Page

NEWS_TEXT = "Reuters - Acme names Jane Doe chief financial officer. " * 10


# ---------------------------------------------------------------- fakes

class FakeClassifier:
    """Assigns source types by URL substring; records what it was asked to judge."""

    def __init__(self, mapping=None, default=("news", True)):
        self.mapping, self.default, self.seen = mapping or {}, default, []

    def __call__(self, items):
        out = []
        for it in items:
            self.seen.append(it["url"])
            stype, ok = next((v for k, v in self.mapping.items() if k in it["url"]), self.default)
            out.append((stype, ok, "fake"))
        return out


def broken_classifier(items):
    raise RuntimeError("Gemini down")


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    import database as db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return db


# ---------------------------------------------------------------- source policy: rules

@pytest.mark.parametrize("url,text,accepted,stype", [
    ("https://investors.acme.com/leadership", NEWS_TEXT, True, "official"),
    ("https://www.sec.gov/Archives/x.htm", NEWS_TEXT, True, "regulatory"),
    ("https://people.example.com/acme-email-format_123", NEWS_TEXT, False, "people_data_aggregator"),
    ("https://data.example.io/p/jane",
     "Jane Doe CFO. Email address and phone number. Reveal email. Direct dial. " * 5,
     False, "people_data_aggregator"),
    ("https://social.example.com/in/jane", "Sign in to view Jane's full profile. " * 3,
     False, "other"),
    ("https://news.example.com/a", "too short", False, "other"),
])
def test_rule_verdicts_need_no_llm(url, text, accepted, stype):
    clf = FakeClassifier()
    (v,) = source_policy.evaluate([{"url": url, "text": text}], "acme.com", clf)
    assert (v.accepted, v.source_type, v.decided_by) == (accepted, stype, "rule")
    assert clf.seen == []


def test_ambiguous_sources_go_to_llm_and_aggregators_are_rejected():
    clf = FakeClassifier({"linkedin.com": ("professional_profile", True),
                          "directory.example.ai": ("people_data_aggregator", False),
                          "blog.example.net": ("other", False)})
    items = [{"url": u, "text": NEWS_TEXT} for u in (
        "https://www.linkedin.com/in/janedoe", "https://directory.example.ai/acme-executives",
        "https://blog.example.net/acme", "https://www.reuters.com/acme-cfo")]
    verdicts = source_policy.evaluate(items, "acme.com", clf)
    assert [v.accepted for v in verdicts] == [True, False, False, True]
    assert all(v.decided_by == "llm" for v in verdicts)


def test_aggregator_marked_credible_is_still_rejected():
    clf = FakeClassifier({"x": ("people_data_aggregator", True)})
    (v,) = source_policy.evaluate([{"url": "https://x.example.com", "text": NEWS_TEXT}],
                                  "acme.com", clf)
    assert not v.accepted


def test_classifier_failure_rejects_unverified_sources():
    (v,) = source_policy.evaluate([{"url": "https://www.reuters.com/a", "text": NEWS_TEXT}],
                                  "acme.com", broken_classifier)
    assert not v.accepted and v.decided_by == "fallback"


def test_relevant_passages_keep_only_text_near_terms():
    text = "x" * 5000 + " Jane Doe, Chief Financial Officer, joined in 2024. " + "y" * 5000
    out = source_policy.relevant_passages(text, ["Jane Doe"], window=100)
    assert "Jane Doe" in out and len(out) < 400


# ---------------------------------------------------------------- collection pipeline

def test_crawler_drops_redirects_to_other_sites(monkeypatch):
    import scraper
    html = {"https://acme.com": '<a href="/leadership">Leadership</a>',
            "https://acme.com/leadership": "<p>moved</p>"}
    redirects = {"https://acme.com/leadership": "https://people-data.example.com/acme"}

    def fake_fetch(url, robots=None):
        url = url.rstrip("/")
        if url in html:
            return redirects.get(url, url), html[url]
        return None
    monkeypatch.setattr(scraper, "fetch", fake_fetch)
    monkeypatch.setattr(scraper, "_robots", lambda base: None)
    _, _, pages = scraper.crawl_company_site("acme.com", log=lambda m: None)
    assert all("people-data.example.com" not in p.url for p in pages)


def test_ingest_never_stores_or_cites_rejected_sources(tmp_db, monkeypatch):
    import rag
    import scraper
    import search
    db = tmp_db
    official = Page("https://acme.com/leadership",
                    "Leadership team. Jane Doe, Chief Executive Officer. " + "About Acme. " * 300)
    news = Page("https://www.reuters.com/acme", "Acme CFO John Roe said... " * 20)
    aggregator = Page("https://directory.example.ai/acme-executives",
                      "Acme executives: Mark Fake, VP Sales. " * 20)
    contact = Page("https://people.example.com/acme-email-format",
                   "Acme email format. Mark Fake phone number. " * 20)

    monkeypatch.setattr(scraper, "crawl_company_site",
                        lambda url, log: ("https://acme.com", "Acme", [official]))
    monkeypatch.setattr(scraper, "fetch_wikipedia", lambda *a, **k: None)
    monkeypatch.setattr(search, "discover", lambda *a, **k: [news, aggregator, contact])
    clf = FakeClassifier({"directory.example.ai": ("people_data_aggregator", False)})
    monkeypatch.setattr(rag, "make_llm_classifier", lambda *a: clf)
    monkeypatch.setattr(rag, "make_llm_resolver", lambda *a: (lambda a, b: (False, "fake")))
    extracted = []

    def fake_extract(company, domain, context, page):
        extracted.append(page.url)
        people = {official.url: ("Jane Doe", "Chief Executive Officer"),
                  news.url: ("John Roe", "CFO"),
                  aggregator.url: ("Mark Fake", "VP Sales"),
                  contact.url: ("Mark Fake", "VP Sales")}
        n, t = people[page.url]
        return [{"name": n, "title": t, "department": "", "seniority": "executive",
                 "status": "current", "as_of": "", "evidence": f"{n}, {t}",
                 "source_url": page.url}], []
    monkeypatch.setattr(rag, "extract_leaders", fake_extract)
    searched = []
    monkeypatch.setattr(search, "tavily_available", lambda: True)
    monkeypatch.setattr(search, "web_search", lambda q, **k: searched.append(q) or [])

    cid = rag.ingest_company("acme.com", log=lambda m: None)

    assert set(extracted) == {official.url, news.url}          # filtered before extraction
    assert contact.url not in clf.seen                          # rule-rejected, no LLM call
    leaders = db.get_leaders(cid)
    assert {p["name"] for p in leaders} == {"Jane Doe", "John Roe"}
    cited = {u for p in leaders for u in p["source_urls"].split("\n")}
    stored = {s["url"] for s in db.get_sources(cid)}
    for bad in (aggregator.url, contact.url):
        assert bad not in cited and bad not in stored
    # Sources are stored as passages, not whole pages.
    off = next(s for s in db.get_sources(cid) if s["url"] == official.url)
    assert "Jane Doe" in off["content"] and len(off["content"]) < len(official.text)
    assert all(r["source_url"] in cited for r in db.get_role_history(cid))
    # John Roe has a single non-official source: one corroboration search was made.
    assert len(searched) == 1 and '"John Roe"' in searched[0]
    # The collection report records every source decision.
    import json
    report = json.loads(db.get_company(cid)["collection_report"])
    assert len(report["pages"]) == 4 and report["people"] == 2
    assert {p["url"] for p in report["pages"] if not p["accepted"]} == {aggregator.url, contact.url}
    assert all(p["reason"] for p in report["pages"])


# ---------------------------------------------------------------- migration

def _old_schema_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE companies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE, scraped_at TEXT, resolved_version INTEGER DEFAULT 0);
        CREATE TABLE leaders (id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            name TEXT NOT NULL, title TEXT, department TEXT, seniority TEXT,
            source_url TEXT NOT NULL, source_urls TEXT, evidence TEXT, aliases TEXT,
            status TEXT, as_of TEXT, UNIQUE (company_id, name, title));
        CREATE TABLE role_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
            leader_id INTEGER NOT NULL REFERENCES leaders(id) ON DELETE CASCADE,
            name TEXT, title TEXT, status TEXT, as_of TEXT, source_url TEXT NOT NULL,
            source_tier INTEGER, evidence TEXT, is_current_title INTEGER DEFAULT 0);
        CREATE TABLE sources (id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            url TEXT NOT NULL, content TEXT, fetched_at TEXT, UNIQUE (company_id, url));
        INSERT INTO companies VALUES (1, 'Acme', 'https://acme.com/', '2026-11-12T00:00:00+00:00', 2);
        INSERT INTO leaders VALUES (1, 1, 'Jane Doe', 'CEO', '', 'executive',
            'https://acme.com/leadership', 'https://acme.com/leadership', 'Jane Doe, CEO',
            '[]', 'current', '2026-11-12');
        INSERT INTO leaders VALUES (2, 1, 'Old Boss', 'CEO', '', 'executive',
            'https://news.example.com/a', 'https://news.example.com/a', 'former CEO Old Boss',
            '[]', 'former', '2019');
        INSERT INTO role_history VALUES (1, 1, 'Jane Doe', 'CEO', 'current', '',
            'https://acme.com/leadership', 1, 'Jane Doe, CEO', 1);
        INSERT INTO role_history VALUES (2, 1, 'Jane Doe', 'President', 'former', '2020',
            'https://news.example.com/b', 2, 'former President', 0);
        INSERT INTO role_history VALUES (3, 2, 'Old Boss', 'CEO', 'former', '2019',
            'https://news.example.com/a', 2, 'former CEO Old Boss', 1);
    """)
    conn.commit()
    conn.close()


def test_migration_drops_status_safely_and_is_rerunnable(tmp_path, monkeypatch):
    import database as db
    path = tmp_path / "old.db"
    _old_schema_db(path)
    monkeypatch.setattr(db, "DB_PATH", str(path))

    db.init_db()
    conn = sqlite3.connect(path)
    cols = lambda t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}  # noqa: E731
    assert "status" not in cols("leaders") and "status" not in cols("role_history")
    assert {"as_of", "source_tier", "source_type"} <= cols("role_history")
    assert "as_of" in cols("leaders") and {"published", "origin"} <= cols("sources")
    # Former roles removed; a person whose only role was former is removed with it.
    assert [r[0] for r in conn.execute("SELECT id FROM leaders")] == [1]
    assert [r[0] for r in conn.execute("SELECT id FROM role_history")] == [1]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()
    backups = list((tmp_path / "backups").glob("old-pre-migration-*.db"))
    assert len(backups) == 1
    # The backup still holds the original data.
    assert sqlite3.connect(backups[0]).execute("SELECT COUNT(*) FROM leaders").fetchone()[0] == 2

    db.init_db()  # rerun: no-op, no new backup
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1
    (p,) = db.get_leaders(1)
    assert p["name"] == "Jane Doe" and "status" not in p


def test_reresolution_removes_unverified_evidence_and_flags_recollection(tmp_db):
    import rag
    db = tmp_db
    cid = db.upsert_company("Acme", "https://acme.com/")
    db.save_people(cid, [
        {"name": "Jane Doe", "title": "CEO", "source_url": "https://acme.com/leadership",
         "source_urls": ["https://acme.com/leadership", "https://exec-lists.example.com/acme"],
         "seniority": "executive", "roles": [
             {"title": "CEO", "source_url": "https://acme.com/leadership", "is_current_title": 1},
             {"title": "CEO", "source_url": "https://exec-lists.example.com/acme"}]},
        {"name": "Mark Fake", "title": "VP Sales",
         "source_url": "https://people.example.com/acme-email-format",
         "seniority": "vp", "roles": [
             {"title": "VP Sales", "source_url": "https://people.example.com/acme-email-format",
              "is_current_title": 1}]},
    ])
    db.save_source(cid, "https://exec-lists.example.com/acme", NEWS_TEXT)
    clf = FakeClassifier({"exec-lists.example.com": ("people_data_aggregator", False)})

    stats = rag.reresolve_company(cid, resolver=lambda a, b: (False, ""), classifier=clf,
                                  log=lambda m: None)
    (jane,) = db.get_leaders(cid)
    assert jane["name"] == "Jane Doe"
    assert jane["source_urls"] == "https://acme.com/leadership"
    assert stats["removed"] == 1 and db.get_company(cid)["removed_unverified"] == 1
    assert db.get_sources(cid) == []                 # rejected source deleted
    assert "people.example.com" not in clf.seen      # rejected by rule, no LLM call


def test_reresolution_changes_nothing_if_source_check_fails(tmp_db):
    import rag
    db = tmp_db
    cid = db.upsert_company("Acme", "https://acme.com/")
    db.save_people(cid, [{"name": "John Roe", "title": "CFO", "seniority": "executive",
                          "source_url": "https://www.reuters.com/a",
                          "roles": [{"title": "CFO", "source_url": "https://www.reuters.com/a"}]}])
    stats = rag.reresolve_company(cid, classifier=broken_classifier, log=lambda m: None)
    assert stats["failed"] == 1 and len(db.get_leaders(cid)) == 1
    assert db.get_company(cid)["resolved_version"] == 0  # retried later


# ---------------------------------------------------------------- views & counts

def test_user_facing_tables_hide_internal_fields():
    leaders = [{"person_id": 7, "name": "Jensen Huang", "aliases": ["Jen-Hsun Huang"],
                "title": "President and CEO", "department": "Executive", "seniority": "executive",
                "source_url": "https://nvidia.com/a", "source_urls": "https://nvidia.com/a\nhttps://b",
                "evidence": "...", "as_of": "2019", "may_be_outdated": True}]
    (row,) = views.people_rows(leaders)
    assert list(row) == views.PEOPLE_COLUMNS
    assert row["Title"] == "President and CEO (may be outdated)"
    assert row["Also known as"] == "Jen-Hsun Huang" and row["More sources"] == ["https://b"]
    for hidden in ("person_id", "status", "as_of", "ID"):
        assert hidden not in row
    roles = [{"person_id": 7, "name": "Jensen Huang", "title": "CEO", "as_of": "2019",
              "source_url": "https://nvidia.com/a", "source_tier": 1, "source_type": "official",
              "evidence": "Jensen Huang, CEO", "is_current_title": 1}]
    (erow,) = views.evidence_rows(roles)
    assert list(erow) == views.EVIDENCE_COLUMNS and erow["Source type"] == "Company website"


def test_counts_are_deterministic_and_count_unique_people():
    import rag
    recs = [
        {"name": "Dan Gallagher", "title": "Chief Legal Officer", "source_url": "https://a.com/1",
         "seniority": "executive"},
        {"name": "Daniel Gallagher", "title": "Chief Legal Officer", "source_url": "https://b.com/2",
         "seniority": "executive"},
        {"name": "Ann Lee", "title": "VP Sales", "source_url": "https://a.com/1", "seniority": "vp"},
        {"name": "Bo Kim", "title": "VP Product", "source_url": "https://a.com/1", "seniority": "vp"},
    ]
    people = entities.resolve_people(recs, "a.com", resolver=lambda a, b: (True, "same"),
                                     fetched_on=date(2026, 11, 12))
    counts = [rag.count_people(people) for _ in range(3)]
    assert counts[0] == counts[1] == counts[2]
    assert counts[0]["total"] == 3
    assert counts[0]["by_seniority"]["vp"] == 2 and counts[0]["by_seniority"]["executive"] == 1


# ---------------------------------------------------------------- hybrid retrieval

PEOPLE = [
    {"person_id": 1, "name": "Jensen Huang", "aliases": ["Jen-Hsun Huang"],
     "title": "Founder, President and Chief Executive Officer", "seniority": "executive",
     "may_be_outdated": False},
    {"person_id": 2, "name": "Colette Kress", "aliases": [], "seniority": "executive",
     "title": "EVP and Chief Financial Officer", "may_be_outdated": False},
    {"person_id": 3, "name": "Pat Lee", "aliases": [], "seniority": "vp",
     "title": "Vice President, Marketing", "may_be_outdated": False},
]


@pytest.mark.parametrize("question,expected", [
    ("Where is NVIDIA's CEO based?", ["Jensen Huang"]),
    ("What did Jen-Hsun Huang study?", ["Jensen Huang"]),
    ("Who heads marketing at NVIDIA?", ["Pat Lee"]),
    ("What is the CFO's professional background?", ["Colette Kress"]),
    ("How many VPs are listed?", []),
])
def test_find_people_resolves_names_aliases_and_roles(question, expected):
    import rag
    assert [p["name"] for p in rag.find_people(question, PEOPLE, "NVIDIA")] == expected


@pytest.fixture
def nvidia(tmp_db):
    db = tmp_db
    cid = db.save_collection("NVIDIA", "https://www.nvidia.com/", [{
        "name": "Jensen Huang", "aliases": ["Jen-Hsun Huang"], "seniority": "executive",
        "title": "Founder, President and Chief Executive Officer", "as_of": "2026-11-12",
        "source_url": "https://www.nvidia.com/board/jensen-huang",
        "roles": [{"title": "Founder, President and CEO", "is_current_title": 1,
                   "source_url": "https://www.nvidia.com/board/jensen-huang",
                   "source_type": "official", "evidence": "Jensen Huang founded NVIDIA in 1993"}]}],
        [{"url": "https://www.nvidia.com/board/jensen-huang", "source_type": "official",
          "content": "Jensen Huang founded NVIDIA in 1993 and has served as CEO since its inception."}],
        {})
    return db, cid


class FakeLLM:
    """Sufficient only when the context contains `needle`."""

    def __init__(self, needle, missing="based in city"):
        self.needle, self.missing, self.contexts = needle, missing, []

    def __call__(self, context, question, history):
        import rag
        self.contexts.append(context)
        if self.needle in context:
            return rag.AnswerDraft(
                answer=("He is based in Santa Clara [Example News](https://news.example.com/jh) "
                        "[P1] see [made up](https://invented.example.org/x)."),
                sufficient=True, missing_info="")
        return rag.AnswerDraft(answer="The collected sources don't say where he is based.",
                               sufficient=False, missing_info=self.missing)


def test_stored_evidence_answers_without_web_search(nvidia):
    import rag
    _db, cid = nvidia
    searched = []
    llm = FakeLLM(needle="founded NVIDIA in 1993")
    out = rag.answer_with_trace(cid, "When did the CEO found NVIDIA?", llm=llm,
                                searcher=lambda q: searched.append(q) or [])
    assert out["focus"] == ["Jensen Huang"] and not out["searched"] and searched == []
    assert len(llm.contexts) == 1


def test_web_fallback_filters_caches_and_cites(nvidia):
    import rag
    db, cid = nvidia
    queries = []
    results = [
        Page("https://news.example.com/jh",
             "Profile: NVIDIA CEO Jen-Hsun Huang works from Santa Clara. " * 5,
             published="2026-03-01"),
        # same name, but nothing ties this profile to the company: not evidence
        Page("https://www.linkedin.com/in/jensen-huang-123", "Jensen Huang, Austin, TX. " * 10),
        Page("https://directory.example.ai/nvidia-people", "Jensen Huang CEO, contacts. " * 10),
        Page("https://unrelated.example.com/x", "Nothing about the person here. " * 10),
    ]

    def searcher(q):
        queries.append(q)
        return results
    clf = FakeClassifier({"directory.example.ai": ("people_data_aggregator", False)})
    llm = FakeLLM(needle="works from Santa Clara")

    out = rag.answer_with_trace(cid, "Where is NVIDIA's CEO based?", llm=llm, searcher=searcher,
                                classifier=clf)
    assert out["searched"] and out["web_sources"] == ["https://news.example.com/jh"]
    assert len(queries) == 1 and '"Jensen Huang" NVIDIA' in queries[0]
    assert "based in city" in queries[0]
    assert "unrelated.example.com/x" not in clf.seen      # doesn't mention the person
    assert "https://www.linkedin.com/in/jensen-huang-123" not in clf.seen  # no company named
    # Accepted passage cached as a web source; aggregator never stored.
    stored = {s["url"]: s for s in db.get_sources(cid)}
    assert stored["https://news.example.com/jh"]["origin"] == "web"
    assert "https://directory.example.ai/nvidia-people" not in stored
    # Citations: real source kept, invented link and internal ID removed.
    assert "[Example News](https://news.example.com/jh)" in out["answer"]
    assert "invented.example.org" not in out["answer"] and "[P1]" not in out["answer"]

    # Repeating the question reuses the cached passage: no new search.
    again = rag.answer_with_trace(cid, "Where is NVIDIA's CEO based?", llm=llm,
                                  searcher=searcher, classifier=clf)
    assert not again["searched"] and len(queries) == 1


def test_identical_unanswerable_search_is_not_repeated(nvidia):
    import rag
    _db, cid = nvidia
    queries = []
    llm = FakeLLM(needle="never present", missing="university attended")
    for _ in range(2):
        out = rag.answer_with_trace(cid, "Which university did the CEO attend?", llm=llm,
                                    searcher=lambda q: queries.append(q) or [])
    assert len(queries) == 1 and out["cache_hit"]
    assert "don't say" in out["answer"]


def test_search_failure_still_answers(nvidia):
    import rag
    _db, cid = nvidia

    def down(q):
        raise RuntimeError("Tavily unavailable")
    out = rag.answer_with_trace(cid, "Where is the CEO based?", llm=FakeLLM("never present"),
                                searcher=down)
    assert out["search_error"] and "could not be checked" in out["answer"]


def test_customer_testimonials_are_not_extracted_as_leaders(monkeypatch):
    import rag
    text = ('"Acme cut our close time in half." See Globex\'s Story Brian Lee Head of Finance '
            "Our team: Jane Doe, Chief Executive Officer of Acme.")
    parsed = rag.LeaderList(organization_on_page="Acme", same_company=True, page_date="", leaders=[
        rag.Leader(name="Brian Lee", title="Head of Finance", department="Finance",
                   seniority="department_head", relationship="customer_or_partner",
                   status="current", as_of="", evidence="Brian Lee Head of Finance"),
        rag.Leader(name="Jane Doe", title="Chief Executive Officer", department="Executive",
                   seniority="executive", relationship="employee_or_board", status="current",
                   as_of="", evidence="Jane Doe, Chief Executive Officer of Acme"),
    ])
    monkeypatch.setattr(rag, "generate", lambda *a, **k: type("R", (), {"parsed": parsed})())
    people, rejected = rag.extract_leaders("Acme", "acme.com", "", Page("https://acme.com/p", text))
    assert [p["name"] for p in people] == ["Jane Doe"]
    assert [(r["name"], r["reason"]) for r in rejected] == [
        ("Brian Lee", "not an employee or board member")]
    assert "relationship" not in people[0]


def test_role_question_returns_all_equally_matching_people():
    import rag
    people = PEOPLE + [{"person_id": 4, "name": "Kim Park", "aliases": [], "seniority": "executive",
                        "title": "SVP, Growth and Marketing", "may_be_outdated": False}]
    found = [p["name"] for p in rag.find_people("Who heads marketing?", people, "NVIDIA")]
    assert found == ["Kim Park", "Pat Lee"]  # executive first, but the VP is not dropped
