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


def rec(name, title, url, status="current", as_of="", evidence=None, seniority="executive"):
    return {"name": name, "title": title, "source_url": url, "status": status,
            "as_of": as_of, "evidence": evidence or f"{name}, {title}",
            "seniority": seniority, "department": ""}


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
    assert len(p["source_urls"]) == 4


def test_bhat_bhatt_stay_separate_when_evidence_check_says_no():
    assert len(resolve(BHATT, FakeResolver(same=False))) == 2


def test_bhatt_current_title_vs_role_history():
    (p,) = resolve(BHATT, FakeResolver(same=True))
    assert p["status"] == "current"
    # Undated company leadership page (fetched today) beats the dated 2021 press release.
    assert p["title"] == "Co-Founder and Chief Creative Officer"
    roles = {r["title"]: r for r in p["roles"]}
    assert roles["Co-CEO"]["status"] == "former"
    assert roles["Co-CEO"]["as_of"] == "2024-04-25"
    # The 2021 "Co-CEO" claim is kept as history, not presented as current.
    assert not roles["Co-Founder and Co-CEO"]["is_current_title"]


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


def test_all_former_records_mark_person_former():
    (p,) = resolve([rec("Jane Doe", "CFO", "https://news.example.com/a",
                        status="former", as_of="2023")])
    assert p["status"] == "former"


@pytest.mark.parametrize("url,as_of,expected", [
    # old press release on the company's own site: can't confirm the role today
    ("https://robinhood.com/newsroom/new-cfo", "2019-11-21", "unverified"),
    # undated third-party profile
    ("https://www.linkedin.com/company/robinhood", "", "unverified"),
    # recent dated news article
    ("https://www.cnbc.com/2026/robinhood", "2026-05-01", "current"),
    # live, undated company leadership page
    ("https://investors.robinhood.com/governance/leadership", "", "current"),
    # live company page that mentions an old *start* date ("on our board since 2021")
    ("https://investors.robinhood.com/governance/board-of-directors", "June 2021", "current"),
    # an SEC filing is a dated document, not a live page
    ("https://www.sec.gov/Archives/edgar/data/1/old-proxy.htm", "2021-04-01", "unverified"),
])
def test_current_requires_recent_or_live_company_evidence(url, as_of, expected):
    (p,) = resolve([rec("Jane Doe", "Chief Financial Officer", url, as_of=as_of)])
    assert p["status"] == expected


@pytest.mark.parametrize("url,tier", [
    ("https://investors.robinhood.com/governance", 1),
    ("https://www.robinhood.com/us/en/newsroom/x", 1),
    ("https://www.sec.gov/Archives/x", 1),
    ("https://www.cnbc.com/2025/robinhood", 2),
    ("https://www.linkedin.com/company/robinhood", 3),
])
def test_source_tiers(url, tier):
    assert entities.source_tier(url, COMPANY) == tier


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
    cid = db.upsert_company("Robinhood", "https://robinhood.com/")
    people = resolve(GALLAGHER + BHATT, FakeResolver(same=True))
    assert db.save_people(cid, people) == 2

    ctx = rag.build_context(cid, "How many executives are there?")
    # Bhatt: confirmed by the live company page. Gallagher: only undated aggregator sources.
    assert "UNIQUE PEOPLE: 2 (1 current, 1 unverified, 0 former)" in ctx
    assert ctx.count("\n- ") == 2  # one bullet per person, not per source record
    assert "also written: Baiju Bhat" in ctx
    assert "other reported title: Co-CEO (former, 2024-04-25" in ctx
