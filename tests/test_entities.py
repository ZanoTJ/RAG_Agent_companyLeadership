"""Regression tests for entity resolution. Offline: the LLM resolver is faked.

The record fixtures mirror real duplicates seen when collecting Robinhood, but
the code under test has no knowledge of these people.
"""

from datetime import date

import pytest

import entities
from entities import parse_name, resolve_people

COMPANY = "robinhood.com"
FETCHED = date(2026, 11, 12)


def rec(name, title, url, status="current", as_of="", evidence=None, seniority="executive",
        **extra):
    return {"name": name, "title": title, "source_url": url, "status": status,
            "as_of": as_of, "evidence": evidence or f"{name}, {title}",
            "seniority": seniority, "department": "", **extra}


class FakeResolver:
    """Records calls; answers with a fixed decision."""

    def __init__(self, same: bool):
        self.same, self.calls = same, []

    def __call__(self, a, b):
        self.calls.append(({r["name"] for r in a}, {r["name"] for r in b}))
        return self.same, "fake"


def never_called(a, b):
    raise AssertionError(f"resolver should not be called for {a[0]['name']} / {b[0]['name']}")


def resolve(records, resolver=None):
    return resolve_people(records, COMPANY, "Robinhood", resolver=resolver, fetched_on=FETCHED)


def by_name(people):
    return {p["name"]: p for p in people}


# ---------------------------------------------------------------- Dan / Daniel Gallagher

GALLAGHER = [
    rec("Dan Gallagher", "Chief Legal, Compliance and Corporate Affairs Officer",
        "https://exa.ai/websets/directory/robinhood-executives"),
    rec("Dan Gallagher", "Chief Legal, Compliance and Corporate Affairs Officer",
        "https://www.linkedin.com/company/robinhood"),
    rec("Daniel Gallagher", "Chief Legal Officer",
        "https://www.comparably.com/companies/robinhood/executive-team"),
]


def test_dan_daniel_is_flagged_for_review_not_auto_merged():
    resolver = FakeResolver(same=True)
    people = resolve(GALLAGHER, resolver)
    assert resolver.calls == [({"Dan Gallagher"}, {"Daniel Gallagher"})]
    assert len(people) == 1


def test_dan_daniel_merge_preserves_aliases_sources_and_roles():
    (p,) = resolve(GALLAGHER, FakeResolver(same=True))
    assert {p["name"], *p["aliases"]} == {"Dan Gallagher", "Daniel Gallagher"}
    assert len(p["source_urls"]) == 3
    assert {r["title"] for r in p["roles"]} == {
        "Chief Legal, Compliance and Corporate Affairs Officer", "Chief Legal Officer"}
    assert sum(r["is_current_title"] for r in p["roles"]) == 1


def test_dan_daniel_stay_separate_when_evidence_check_says_no():
    assert len(resolve(GALLAGHER, FakeResolver(same=False))) == 2


def test_dan_daniel_stay_separate_without_resolver():
    # A similar name alone never merges two people.
    assert len(resolve(GALLAGHER, resolver=None)) == 2


# ---------------------------------------------------------------- Baiju Bhat / Bhatt

BHATT = [
    rec("Baiju Bhat", "Co-Founder",
        "https://fintechleaders.substack.com/p/robinhoods-coo-gretchen-howard",
        evidence="Founded in 2013 by Vlad Tenev and Baiju Bhat, Robinhood"),
    rec("Baiju Bhatt", "Co-Founder and Co-CEO",
        "https://robinhood.com/newsroom/robinhood-welcomes-anne-hoge", as_of="2021-03",
        evidence="Co-Founders and Co-CEOs Vlad Tenev and Baiju Bhatt"),
    rec("Baiju Bhatt", "Co-CEO", "https://www.sec.gov/Archives/edgar/data/1/proxy.htm",
        status="former", as_of="2024-04-25",
        evidence="Baiju Bhatt, Co-Founder, previously served as Co-CEO"),
    rec("Baiju Bhatt", "Co-Founder and Chief Creative Officer",
        "https://investors.robinhood.com/governance/leadership"),
]


def test_bhat_bhatt_spelling_variant_is_flagged_and_merged():
    resolver = FakeResolver(same=True)
    people = resolve(BHATT, resolver)
    assert resolver.calls == [({"Baiju Bhat"}, {"Baiju Bhatt"})]
    (p,) = people
    # Canonical spelling comes from the authoritative (company) sources.
    assert p["name"] == "Baiju Bhatt"
    assert p["aliases"] == ["Baiju Bhat"]
    # The SEC record only reports a *former* role, and the 2021 press release only
    # supports the Co-CEO claim that role ended - neither is kept as evidence.
    assert set(p["source_urls"]) == {
        "https://investors.robinhood.com/governance/leadership",
        "https://fintechleaders.substack.com/p/robinhoods-coo-gretchen-howard"}


def test_bhat_bhatt_stay_separate_when_evidence_check_says_no():
    stats = {}
    people = resolve_people(BHATT, COMPANY, "Robinhood", resolver=FakeResolver(same=False),
                            fetched_on=FETCHED, stats=stats)
    # Kept separate; the "Baiju Bhat" record alone ("Co-Founder") is not a leadership role.
    assert [p["name"] for p in people] == ["Baiju Bhatt"]
    assert stats["excluded"][0]["name"] == "Baiju Bhat"
    assert stats["excluded"][0]["reason"] == entities.REASON_NO_LEADERSHIP


def test_bhatt_outdated_role_excluded_and_current_title_kept():
    (p,) = resolve(BHATT, FakeResolver(same=True))
    # Undated live company leadership page (fetched today) is the current title.
    assert p["title"] == "Co-Founder and Chief Creative Officer"
    titles = {r["title"] for r in p["roles"]}
    # The former Co-CEO role is not stored, and the 2021 "Co-Founder and Co-CEO" claim
    # is dropped because a newer source says the Co-CEO role ended.
    assert "Co-CEO" not in titles
    assert "Co-Founder and Co-CEO" not in titles
    assert titles == {"Co-Founder", "Co-Founder and Chief Creative Officer"}
    assert "status" not in p and all("status" not in r for r in p["roles"])


# ---------------------------------------------------------------- no false merges

def test_shared_first_name_different_surname_never_compared():
    people = resolve([
        rec("Daniel Gallagher", "Chief Legal Officer", "https://www.comparably.com/a"),
        rec("Daniel Kelati", "CFO, Robinhood Securities", "https://exa.ai/b"),
    ], resolver=never_called)
    assert len(people) == 2


def test_same_title_different_people_never_compared():
    people = resolve([
        rec("Ruth Ashford", "Head of Data Science", "https://rocketreach.co/x"),
        rec("Kane Sweeney", "Head of Data Science", "https://rocketreach.co/x"),
    ], resolver=never_called)
    assert len(people) == 2


def test_resolver_failure_keeps_records_separate():
    def broken(a, b):
        raise RuntimeError("API down")
    assert len(resolve(GALLAGHER, broken)) == 2


def test_exact_normalized_match_merges_without_llm():
    people = resolve([
        rec("Stephanie Guild, CFA", "Chief Investment Officer",
            "https://investors.robinhood.com/governance/leadership"),
        rec("Stephanie  Guild", "CIO", "https://www.linkedin.com/company/robinhood"),
        rec("Dr. Stéphanie Q. Guild", "Chief Investment Officer", "https://news.example.com/a"),
    ], resolver=never_called)
    assert len(people) == 1


def test_first_name_or_initials_only_are_dropped():
    assert resolve([rec("John", "Founder", "https://a.com"),
                    rec("S. C.", "Owner", "https://b.com")]) == []


# ---------------------------------------------------------------- authority & dates

def test_company_source_beats_aggregator_for_current_title():
    (p,) = resolve([
        rec("Jane Doe", "VP Marketing", "https://www.linkedin.com/in/janedoe", as_of="2026-10"),
        rec("Jane Doe", "Chief Marketing Officer",
            "https://investors.robinhood.com/governance/leadership"),
    ])
    assert p["title"] == "Chief Marketing Officer"


def test_newer_record_wins_within_same_tier():
    (p,) = resolve([
        rec("Jane Doe", "VP Product", "https://news.example.com/a", as_of="2022-01-10"),
        rec("Jane Doe", "SVP Product", "https://news.example.org/b", as_of="April 2025"),
    ])
    assert p["title"] == "SVP Product"


def test_person_with_only_former_roles_is_excluded():
    stats = {}
    people = resolve_people([rec("Jane Doe", "CFO", "https://news.example.com/a",
                                 status="former", as_of="2023")],
                            COMPANY, "Robinhood", fetched_on=FETCHED, stats=stats)
    assert people == [] and stats["excluded_former"] == 1


def test_newer_verified_role_replaces_older_one():
    (p,) = resolve([
        rec("Jane Doe", "VP Finance", "https://news.example.com/a", as_of="2021-01-01"),
        rec("Jane Doe", "VP Finance", "https://news.example.org/b", status="former",
            as_of="2025-03-01"),
        rec("Jane Doe", "Chief Financial Officer", "https://news.example.org/b", as_of="2025-03-01"),
    ])
    assert p["title"] == "Chief Financial Officer"
    assert {r["title"] for r in p["roles"]} == {"Chief Financial Officer"}


@pytest.mark.parametrize("url,tier", [
    ("https://investors.robinhood.com/governance", 1),
    ("https://www.robinhood.com/us/en/newsroom/x", 1),
    ("https://www.sec.gov/Archives/x", 1),
    ("https://www.cnbc.com/2025/robinhood", 2),
    ("https://www.linkedin.com/company/robinhood", 2),
])
def test_source_tiers(url, tier):
    assert entities.source_tier(url, COMPANY) == tier


def test_llm_label_never_raises_source_tier():
    # Only the URL decides: a third-party page labeled "official" stays tier 2.
    assert entities.source_tier("https://ir.example-host.com/x", COMPANY, "official") == 2
    assert entities.source_tier("https://stocktitan.net/sec-filings/x", COMPANY, "regulatory") == 2


def test_aggregator_records_never_become_evidence():
    people = resolve([rec("Jane Doe", "CFO", "https://people-data.example.com/x",
                          source_type="people_data_aggregator")])
    assert people == []


def test_name_normalization():
    assert parse_name("Stephanie Guild, CFA").key == "stephanie guild"
    assert parse_name("R. Scott Victoria").key == "scott victoria"
    assert parse_name("Mariano-Florentino (Tino) Cuéllar").nicknames == ("tino",)
    assert parse_name("John") is None


# ---------------------------------------------------------------- chatbot counts people

def test_chat_context_counts_unique_people_not_records(tmp_path, monkeypatch):
    import database as db
    import rag

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    people = resolve(GALLAGHER + BHATT, FakeResolver(same=True))
    cid = db.save_collection("Robinhood", "https://robinhood.com/", people, [])
    assert len(db.get_leaders(cid)) == 2

    ctx, _urls = rag.build_context(cid, "How many executives are there?")
    assert "Unique people listed: 2" in ctx
    assert "C-level executives (incl. presidents): 2" in ctx
    assert ctx.count("\n- [P") == 2  # one bullet per person, not per source record
    assert "also written: Baiju Bhat" in ctx
    assert "Co-CEO" not in ctx.split("QUESTION IS ABOUT")[0]  # former role not listed


# ---------------------------------------------------------------- review regressions

ROSTER = "https://investors.robinhood.com/governance/leadership"
ROSTER_RECS = [rec("Shiv Verma", "Chief Financial Officer", ROSTER),
               rec("Connie Schan", "Chief People Officer", ROSTER),
               rec("Vlad Tenev", "Chief Executive Officer", ROSTER)]


def test_vp_only_in_recent_news_is_kept_even_with_a_roster():
    stats = {}
    people = resolve_people(ROSTER_RECS + [
        rec("Rohit Ranjan", "VP Engineering", "https://www.cnbc.com/2026/robinhood",
            as_of="2026-05-01", source_type="news", seniority="vp"),
        rec("Old Name", "VP Platforms", "https://www.cnbc.com/2019/robinhood",
            as_of="2019-05-01", source_type="news", seniority="vp"),
    ], COMPANY, "Robinhood", fetched_on=FETCHED, stats=stats)
    assert "Rohit Ranjan" in by_name(people)
    assert [(e["name"], e["reason"]) for e in stats["excluded"]] == [
        ("Old Name", entities.REASON_NOT_ON_ROSTER)]


def test_promotion_on_one_page_keeps_the_new_title():
    page = "https://www.cnbc.com/2026/robinhood-promotes"
    (p,) = resolve([
        rec("Ann Lee", "VP of Product", page, status="former", as_of="2026-06-01", seniority="vp"),
        rec("Ann Lee", "SVP of Product", page, as_of="2026-06-01", seniority="vp"),
    ])
    assert p["title"] == "SVP of Product"
    # the same "former" claim from another, equally new source still ends the role
    assert resolve([
        rec("Ann Lee", "SVP of Product", page, as_of="2026-06-01", seniority="vp"),
        rec("Ann Lee", "SVP of Product", "https://news.example.com/x", status="former",
            as_of="2026-06-01"),
    ]) == []


@pytest.mark.parametrize("title", ["General Counsel", "General Counsel and Corporate Secretary"])
def test_general_counsel_is_executive(title):
    assert entities.leadership_level(title) == "executive"
    assert entities.leadership_level("Deputy General Counsel") is None


def test_nicknames_are_not_merged_without_a_resolver():
    people = resolve([
        rec("Bob Smith", "Chief Revenue Officer", "https://investors.robinhood.com/a"),
        rec("Robert Smith", "Chief Revenue Officer", "https://www.cnbc.com/2026/b",
            as_of="2026-01-01", source_type="news"),
    ])
    assert sorted(by_name(people)) == ["Bob Smith", "Robert Smith"]


def test_count_question_uses_counts_and_makes_no_web_search(tmp_path, monkeypatch):
    import database as db
    import rag

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    people = resolve([
        rec("Vlad Tenev", "Chief Executive Officer", ROSTER),
        rec("Pat Lee", "Vice President, Product", ROSTER, seniority="vp"),
        rec("Kim Park", "VP, Marketing", ROSTER, seniority="vp"),
    ])
    cid = db.save_collection("Robinhood", "https://robinhood.com/", people, [])
    contexts, searches = [], []

    def llm(context, question, history):
        contexts.append(context)
        return rag.AnswerDraft(answer="2 VPs are listed.", sufficient=True, missing_info="")

    out = rag.answer_with_trace(cid, "How many VPs do they have?", llm=llm,
                                searcher=lambda q: searches.append(q) or [])
    counts = contexts[0].split("\nPEOPLE (")[0]  # the COUNTS block, before the people list
    assert "Vice Presidents (incl. SVPs/EVPs): 2" in counts
    assert searches == [] and not out["searched"]


# ---------------------------------------------------------------- role phrases as names

@pytest.mark.parametrize("name,ok", [
    ("Head of Data", False), ("Chief Executive Officer", False), ("VP Marketing", False),
    ("Vlad Tenev", True), ("JB Mackenzie", True), ("R. Scott Victoria", True),
    ("Paul C Nichols", True), ("Lily Cao", True),
])
def test_person_name_check(name, ok):
    assert entities.looks_like_person_name(name) is ok


def test_role_phrase_from_a_job_post_is_rejected(monkeypatch):
    import rag
    from scraper import Page
    text = "AI Engineer, Agents. Who to email about it: The Head of Data at Campfire."
    parsed = rag.LeaderList(organization_on_page="Campfire", same_company=True, page_date="",
                            leaders=[rag.Leader(
                                name="Head of Data", title="Head of Data", department="Data",
                                seniority="department_head", relationship="employee_or_board",
                                status="current", as_of="",
                                evidence="The Head of Data at Campfire")])
    monkeypatch.setattr(rag, "generate", lambda *a, **k: type("R", (), {"parsed": parsed})())
    people, rejected = rag.extract_leaders("Campfire", "campfire.ai", "",
                                           Page("https://jobs.example.com/x", text))
    assert people == []
    assert [r["reason"] for r in rejected] == ["name is a role, not a person"]
    # the backstop in resolve_people drops it too
    assert resolve([rec("Head of Data", "Head of Data", "https://campfire.ai/team")]) == []


def test_ceo_with_extra_non_scope_words_is_still_a_ceo_claim():
    assert "ceo" in entities.single_holder_roles("CEO & Managing Partner")
    assert "ceo" in entities.single_holder_roles("CEO & Founder")
    assert entities.single_holder_roles("COO, Robinhood Financial") == set()
