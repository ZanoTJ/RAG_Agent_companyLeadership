"""Leadership scope, current-roster rule, one-person-role conflicts, current profile
evidence, source discrepancies, evidence dates, and source labels. Offline: all LLM
and search calls are faked. Fixtures mirror real Robinhood/Campfire cases; the code
has no knowledge of them."""

from datetime import date

import pytest

import entities
import source_policy
from entities import leadership_level, resolve_people, single_holder_roles
from scraper import Page

FETCHED = date(2026, 11, 13)


def rec(name, title, url, status="current", as_of="", seniority="executive", **extra):
    return {"name": name, "title": title, "source_url": url, "status": status,
            "as_of": as_of, "evidence": f"{name}, {title}", "seniority": seniority,
            "department": "", **extra}


def resolve(records, domain="robinhood.com", stats=None):
    return resolve_people(records, domain, "", resolver=None, fetched_on=FETCHED, stats=stats)


def names(people):
    return sorted(p["name"] for p in people)


# ---------------------------------------------------------------- leadership scope

@pytest.mark.parametrize("title,label,level", [
    ("Chairman & Chief Executive Officer", "", "executive"),
    ("EVP and Chief Financial Officer", "", "executive"),
    ("President of Robinhood UK and General Manager of Bitstamp UK", "", "executive"),
    ("Co-founder / CTO", "", "executive"),
    # a plain VP with a functional "chief" title holds VP rank
    ("Vice President and Chief Information Security Officer", "", "vp"),
    ("Vice President, Brokerage and President, Robinhood Financial", "executive", "vp"),
    ("Senior Vice President, General Counsel and Corporate Secretary", "", "vp"),
    ("Head of Product", "", "department_head"),
    # Senior Directors count only when they lead a department or function
    ("Senior Director, Communications", "department_head", "department_head"),
    ("Senior Director, Communications", "other", None),
    # not leadership roles
    ("Chief of Staff", "executive", None),
    ("Co-Founder", "board", None),
    ("Independent Director", "department_head", None),
    ("Lead Director, Independent Director", "board", None),
    ("Managing General Partner, New Enterprise Associates", "board", None),
    ("Founding Solution Consultant", "other", None),
])
def test_leadership_level_from_title(title, label, level):
    assert leadership_level(title, label) == level


def test_board_only_members_excluded_but_chair_ceo_kept():
    stats = {}
    people = resolve([
        rec("Vlad Tenev", "Chairman & Chief Executive Officer",
            "https://investors.robinhood.com/governance/board-of-directors"),
        rec("Susan Segal", "Independent Director",
            "https://investors.robinhood.com/governance/board-of-directors", seniority="board"),
        rec("Baiju Bhatt", "Co-Founder",
            "https://investors.robinhood.com/governance/board-of-directors", seniority="board"),
    ], stats=stats)
    assert names(people) == ["Vlad Tenev"]
    assert people[0]["seniority"] == "executive"
    assert {e["reason"] for e in stats["excluded"]} == {entities.REASON_NO_LEADERSHIP}


def test_vp_count_comes_from_titles_not_llm_labels():
    from rag import count_people
    people = resolve([
        rec("Matt Billings", "Vice President, Brokerage and President, Robinhood Financial",
            "https://investors.robinhood.com/governance/leadership", seniority="executive"),
        rec("Evan McHenry", "Vice President and Chief Information Security Officer",
            "https://investors.robinhood.com/governance/leadership", seniority="executive"),
        rec("Shiv Verma", "Chief Financial Officer",
            "https://investors.robinhood.com/governance/leadership", seniority="vp"),
    ])
    counts = count_people(people)
    assert counts["by_seniority"] == {"executive": 1, "vp": 2, "department_head": 0}


# ---------------------------------------------------------------- current roster

ROSTER = "https://investors.robinhood.com/governance/leadership"


def _roster(*extra):
    return [rec("Shiv Verma", "Chief Financial Officer", ROSTER),
            rec("Connie Schan", "Chief People Officer", ROSTER),
            rec("Lucas Moskowitz", "Senior Vice President, General Counsel", ROSTER),
            *extra]


def test_people_missing_from_current_roster_with_old_evidence_are_excluded():
    stats = {}
    people = resolve(_roster(
        # old, non-official evidence, not on today's roster
        rec("Rohit Ranjan", "VP Engineering", "https://www.cnbc.com/2021/robinhood",
            as_of="2021-03-01", source_type="news"),
        # undated listing on a third-party company page
        rec("Sanjay Kotte", "Chief Commercial Officer",
            "https://www.linkedin.com/company/robinhood", source_type="professional_profile"),
        # a tier-1 source (filing) keeps the person even when it is old
        rec("Christina Lai", "Corporate Secretary, VP, and Deputy General Counsel",
            "https://www.sec.gov/Archives/edgar/data/1/hood-20230428.htm", as_of="2023-04-28"),
    ), stats=stats)
    assert names(people) == ["Christina Lai", "Connie Schan", "Lucas Moskowitz", "Shiv Verma"]
    assert {e["name"] for e in stats["excluded"]} == {"Rohit Ranjan", "Sanjay Kotte"}
    assert {e["reason"] for e in stats["excluded"]} == {entities.REASON_NOT_ON_ROSTER}


# ---------------------------------------------------------------- current profile evidence

def test_own_profile_listing_a_current_role_counts_as_current():
    (p,) = resolve([rec("Jack Arrix", "Head of Partnerships", "https://www.linkedin.com/in/jackarrix",
                        source_type="professional_profile", seniority="department_head")],
                   domain="campfire.ai")
    assert p["as_of"] == FETCHED.isoformat()  # current as of the fetch date
    assert p["title"] == "Head of Partnerships"


def test_company_page_listing_is_not_a_personal_profile():
    (p,) = resolve([rec("Paul C Nichols", "CTO", "https://www.linkedin.com/company/meetcampfire",
                        source_type="professional_profile")], domain="campfire.ai")
    assert p["as_of"] == ""  # undated, not treated as current evidence


def test_profile_contradiction_is_flagged_not_resolved():
    url = "https://www.linkedin.com/in/hank-sun"
    (p,) = resolve([
        rec("Hank Sun", "Head of Solution Consulting", url, source_type="professional_profile",
            seniority="department_head"),
        rec("Hank Sun", "Founding Solution Consultant", url, source_type="professional_profile",
            seniority="other"),
    ], domain="campfire.ai")
    assert p["title"] == "Head of Solution Consulting"
    assert "Founding Solution Consultant" in p["conflict"]


# ---------------------------------------------------------------- one-person roles

def test_single_holder_roles():
    assert single_holder_roles("Co-founder / CTO") == {"cto"}
    assert single_holder_roles("Founder, President and Chief Executive Officer") == {"ceo"}
    assert single_holder_roles("CEO / CFO") == {"ceo", "cfo"}
    assert single_holder_roles("Crypto CFO") == set()
    assert single_holder_roles("Vice President and Chief Operating Officer, Robinhood Financial") == set()
    assert single_holder_roles("Co-CEO") == set()


def test_newer_cto_supersedes_older_cto_claim():
    stats = {}
    people = resolve([
        rec("Fernando San Martin", "Co-founder / CTO",
            "https://campfire.ai/blog/were-coming-out-of-stealth", as_of="2024-02-27"),
        rec("Paul C Nichols", "CTO", "https://www.linkedin.com/in/paulcnichols",
            source_type="professional_profile"),
    ], domain="campfire.ai", stats=stats)
    assert names(people) == ["Paul C Nichols"]
    (ex,) = stats["excluded"]
    assert ex["name"] == "Fernando San Martin" and "CTO role superseded" in ex["reason"]


def test_superseded_person_keeps_another_leadership_role():
    people = resolve([
        rec("Ann Old", "CTO", "https://news.example.com/a", as_of="2023-01-10", source_type="news"),
        rec("Ann Old", "Head of Research", "https://news.example.com/a", as_of="2023-01-10",
            source_type="news", seniority="department_head"),
        rec("Bo New", "Chief Technology Officer", "https://news.example.org/b",
            as_of="2026-03-01", source_type="news"),
    ], domain="acme.com")
    by = {p["name"]: p for p in people}
    assert by["Ann Old"]["title"] == "Head of Research"
    assert by["Bo New"]["title"] == "Chief Technology Officer"


def test_undecidable_role_conflict_is_flagged_on_both():
    people = resolve([
        rec("Ann One", "CTO", "https://news.example.com/a", source_type="news"),
        rec("Bo Two", "CTO", "https://other.example.org/b", source_type="reference"),
    ], domain="acme.com")
    assert len(people) == 2
    assert all("Sources disagree on who is CTO" in p["conflict"] for p in people)


def test_scoped_title_is_shown_and_is_not_a_company_wide_role():
    people = resolve([
        rec("R. Scott Victoria", "President and COO", "https://www.linkedin.com/in/rscottvictoria",
            source_type="professional_profile"),
        rec("R. Scott Victoria", "President and COO of TradePMR by Robinhood",
            "https://www.wealthtechstrategy.com/post/x", source_type="news"),
        rec("Jane Roe", "Chief Operating Officer", ROSTER)])
    by = {p["name"]: p for p in people}
    assert by["R. Scott Victoria"]["title"] == "President and COO of TradePMR by Robinhood"
    assert by["R. Scott Victoria"]["as_of"] == FETCHED.isoformat()  # dated by the profile
    assert not by["R. Scott Victoria"]["conflict"] and not by["Jane Roe"]["conflict"]
    # the company's own name is not a scope
    assert not entities._scope_of("CEO", "CEO of Robinhood", ("robinhood.com",))
    # more roles are not a scope
    assert not entities._scope_of(
        "Chairman & Chief Executive Officer",
        "Chairman, Chief Executive Officer, Co-Founder, and President", ("robinhood.com",))


def test_scoped_cfo_titles_do_not_conflict():
    people = resolve(_roster(rec("Roger Younan", "Crypto CFO", ROSTER)))
    assert all(not p["conflict"] for p in people) and len(people) == 4


# ---------------------------------------------------------------- dates & sources

def fake_gemini(monkeypatch, parsed):
    """Make every Gemini call return `parsed`."""
    import rag
    monkeypatch.setattr(rag, "generate", lambda *a, **k: type("R", (), {"parsed": parsed})())


def _fake_extract(monkeypatch, page_date, as_of):
    import rag as extraction
    parsed = extraction.LeaderList(
        organization_on_page="Campfire", same_company=True, page_date=page_date,
        leaders=[extraction.Leader(
            name="Brendan Doyle", title="Head of Product", department="Product",
            seniority="department_head", relationship="employee_or_board",
            status="current", as_of=as_of, evidence="Brendan Doyle Head of Product")])
    fake_gemini(monkeypatch, parsed)
    people, _ = extraction.extract_leaders("Campfire", "campfire.ai", "", Page(
        "https://campfire.ai/blog/x", "Brendan Doyle Head of Product. Related: Sep 18, 2026"))
    return people[0]["as_of"]


def test_evidence_date_is_the_page_date_not_the_role_start(monkeypatch):
    assert _fake_extract(monkeypatch, "February 27, 2024", "since 2016") == "February 27, 2024"
    assert _fake_extract(monkeypatch, "", "since 2016") == "since 2016"


def test_future_dates_are_ignored(monkeypatch):
    assert _fake_extract(monkeypatch, "2099-01-01", "2099-01-01") == ""


def test_llm_cannot_label_a_third_party_page_official():
    items = [{"url": "https://www.thecampfire.ai/about", "text": "About Campfire AI. " * 30},
             {"url": "https://stocktitan.net/sec-filings/HOOD/def-14a", "text": "Proxy. " * 50}]
    verdicts = source_policy.evaluate(items, "campfire.ai",
                                      lambda its: [("official", True, "x"), ("regulatory", True, "y")])
    assert [v.source_type for v in verdicts] == ["reference", "reference"]
    assert source_policy.trusted_type("https://investors.campfire.ai/x", "news", "campfire.ai") == "news"
    assert source_policy.trusted_type("https://www.sec.gov/x", "regulatory", "campfire.ai") == "regulatory"


def test_classifier_rejects_pages_about_a_namesake(monkeypatch):
    import rag as sc
    parsed = sc.SourceAssessments(pages=[
        sc.SourceAssessment(index=0, source_type="news", about_company=False, credible=True,
                            reason="Campfire Interactive, a manufacturing software firm"),
        sc.SourceAssessment(index=1, source_type="news", about_company=True, credible=True,
                            reason="Campfire funding news")])
    fake_gemini(monkeypatch, parsed)
    results = sc.make_llm_classifier("Campfire", "campfire.ai")(
        [{"url": "https://a.example.com", "text": "x"}, {"url": "https://b.example.com", "text": "y"}])
    assert results[0][1] is False and "different organization" in results[0][2]
    assert results[1][1] is True


def test_public_profiles_about_the_company_are_accepted(monkeypatch):
    import rag as sc
    parsed = sc.SourceAssessments(pages=[
        sc.SourceAssessment(index=0, source_type="professional_profile", about_company=True,
                            credible=False, reason="self-reported information"),
        sc.SourceAssessment(index=1, source_type="professional_profile", about_company=False,
                            credible=True, reason="a namesake's profile")])
    fake_gemini(monkeypatch, parsed)
    results = sc.make_llm_classifier("Robinhood", "robinhood.com")(
        [{"url": "https://linkedin.com/in/a", "text": "x"}, {"url": "https://linkedin.com/in/b", "text": "y"}])
    assert [r[1] for r in results] == [True, False]


# ---------------------------------------------------------------- discovery & corroboration

def test_discovery_searches_the_web_and_linkedin():
    import search
    wheres = [w for _, w in search.QUERY_TEMPLATES]
    assert None in wheres and ["linkedin.com"] in wheres


def test_same_page_under_country_subdomains_is_one_page():
    import search
    assert search.page_key("https://fx.linkedin.com/company/robinhood") ==         search.page_key("https://www.linkedin.com/company/robinhood/")
    assert search.page_key("https://linkedin.com/in/a") != search.page_key("https://linkedin.com/in/b")


def test_weakly_supported_people_are_corroborated():
    from rag import _weakly_supported as weakly_supported
    people = [
        {"name": "Chris McDonough", "title": "Vice President of Sales",
         "source_urls": ["https://www.linkedin.com/in/chris-mcdonough"]},
        {"name": "John Glasgow", "title": "CEO", "source_urls": ["https://campfire.ai/blog/x"]},
    ]
    excluded = [
        {"name": "Sanjay Kotte", "title": "CCO", "reason": entities.REASON_NOT_ON_ROSTER},
        {"name": "Susan Segal", "title": "Director", "reason": entities.REASON_NO_LEADERSHIP},
    ]
    targets = weakly_supported(people, excluded, "campfire.ai")
    assert [t["name"] for t in targets] == ["Chris McDonough", "Sanjay Kotte"]


def test_conflict_is_stored_and_shown(tmp_path, monkeypatch):
    import database as db
    import views
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    cid = db.save_collection("Acme", "https://acme.com/", resolve([
        rec("Ann One", "CTO", "https://news.example.com/a", source_type="news"),
        rec("Bo Two", "CTO", "https://other.example.org/b", source_type="reference"),
    ], domain="acme.com"), [], {})
    leaders = db.get_leaders(cid)
    assert all(l["conflict"] for l in leaders)
    assert all(r["Title"].endswith("(sources disagree)") for r in views.people_rows(leaders))
    assert len(views.disagreements(leaders)) == 2


# ---------------------------------------------------------------- departures

PINNER = [
    rec("Jeff Pinner", "CTO", "https://robinhood.com/newsroom/women-in-tech-mixer", as_of="2025-06-12"),
    rec("Jeff Pinner", "Chief Technology Officer",
        "https://robinhood.com/newsroom/robinhood-welcomes-jeff-pinner", as_of="2024-08-12"),
    # 8-K: "Jeffrey Pinner, Chief Technology Officer, would be separating from his position"
    rec("Jeff Pinner", "Chief Technology Officer",
        "https://investors.robinhood.com/static-files/8k", status="former", as_of="May 8, 2026"),
]


def test_departure_cancels_abbreviated_title_claims():
    stats = {}
    assert resolve(PINNER, stats=stats) == []
    assert stats["excluded"][0]["reason"] == entities.REASON_FORMER


def test_older_departure_does_not_cancel_a_newer_claim():
    (p,) = resolve([PINNER[0], {**PINNER[2], "as_of": "2023-01-01"}])
    assert p["title"] == "CTO"


def test_departure_keeps_the_persons_other_current_role(tmp_path, monkeypatch):
    import database as db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    people = resolve([*PINNER, rec("Jeff Pinner", "Head of Platform Research",
                                   "https://investors.robinhood.com/governance/leadership",
                                   seniority="department_head")])
    cid = db.save_collection("Robinhood", "https://robinhood.com/", people, [], {})
    (p,) = db.get_leaders(cid)
    assert p["title"] == "Head of Platform Research"  # the CTO claims are cancelled
    assert [r["title"] for r in db.get_role_history(cid)] == ["Head of Platform Research"]


# ---------------------------------------------------------------- parallel searches

def test_parallel_discovery_merges_in_template_order(monkeypatch):
    """Later templates finish first; dedup must still keep the first template's page."""
    import sys
    import time
    import types

    import search
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    order = {t.format(name="Acme", domain="acme.com"): i
             for i, (t, _) in enumerate(search.QUERY_TEMPLATES)}

    class FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, query, **kw):
            i = order[query]
            time.sleep(0.01 * (len(order) - i))  # first template is slowest
            return {"results": [{"url": "https://news.example.com/shared",
                                 "raw_content": f"Acme from query {i}"},
                                {"url": f"https://news.example.com/{i}",
                                 "raw_content": "Acme"}]}

    monkeypatch.setitem(sys.modules, "tavily", types.SimpleNamespace(TavilyClient=FakeClient))
    pages = search.discover("Acme", "https://acme.com/", log=lambda m: None)
    assert pages[0].url.endswith("/shared") and "query 0" in pages[0].text
    assert [p.url for p in pages[1:]] == [f"https://news.example.com/{i}" for i in range(len(order))]


def test_parallel_corroboration_keeps_target_order(monkeypatch):
    import time

    import rag
    import search
    from scraper import Page

    def fake_search(query, **k):
        if "Bo Two" in query:
            raise RuntimeError("quota")
        time.sleep(0.03 if "Ann One" in query else 0)  # first target finishes last
        who = "Ann One" if "Ann One" in query else "Cy Three"
        return [Page(f"https://news.example.com/{who.split()[1]}", f"{who} joins Acme")]

    monkeypatch.setattr(search, "web_search", fake_search)
    monkeypatch.setattr(rag, "_evaluate_pages", lambda pages, *a: pages)
    logs, report = [], {"corroboration": []}
    targets = [{"name": n, "title": "VP Sales"} for n in ("Ann One", "Bo Two", "Cy Three")]
    pages = rag._corroborate("Acme", "acme.com", targets, set(), None, report, logs.append)
    assert [p.url for p in pages] == ["https://news.example.com/One", "https://news.example.com/Three"]
    assert [c["name"] for c in report["corroboration"]] == ["Ann One", "Cy Three"]
    assert any("failed for Bo Two" in m for m in logs)
