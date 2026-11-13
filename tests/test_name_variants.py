"""Regression tests for alternative name forms, candidate generation, canonical
names, re-resolution of stored data, and canonical person IDs. Offline: the LLM
resolver is faked. Fixtures mirror real cases; the code has no knowledge of them."""

from datetime import date

import pytest

import entities
from entities import is_candidate_pair, parse_name, resolve_people


def rec(name, title, url, status="current", as_of="", evidence=None, **extra):
    return {"name": name, "title": title, "source_url": url, "status": status,
            "as_of": as_of, "evidence": evidence or f"{name}, {title}",
            "seniority": "executive", "department": "", **extra}


class FakeResolver:
    def __init__(self, same: bool):
        self.same, self.calls = same, []

    def __call__(self, a, b):
        self.calls.append(frozenset({a[0]["name"], b[0]["name"]}))
        return self.same, "fake"


def never_called(a, b):
    raise AssertionError(f"resolver should not be called for {a[0]['name']} / {b[0]['name']}")


def resolve(records, domain, resolver=None, name=""):
    return resolve_people(records, domain, name, resolver=resolver, fetched_on=date(2026, 11, 12))


# ---------------------------------------------------------------- Jen-Hsun / Jensen Huang

HUANG = [
    rec("Jen-Hsun Huang", "Founder, President and Chief Executive Officer",
        "https://vectorshift.ai/research/companies/nvidia/people"),
    rec("Jen-Hsun Huang", "President and Chief Executive Officer",
        "https://www.sec.gov/Archives/edgar/data/1045810/proxy.htm", as_of="2026-05-01"),
    rec("Jensen Huang", "Co-founder, President and Chief Executive Officer",
        "https://www.nvidia.com/en-us/about-nvidia/board-of-directors/jensen-huang"),
    rec("Jensen Huang", "Founder, President and CEO",
        "https://investor.nvidia.com/governance/management-team/default.aspx"),
]


def test_jensen_jen_hsun_flagged_merged_with_company_name_canonical():
    resolver = FakeResolver(same=True)
    (p,) = resolve(HUANG, "nvidia.com", resolver)
    assert resolver.calls == [frozenset({"Jen-Hsun Huang", "Jensen Huang"})]
    # The company's own site says "Jensen"; filings use the legal name.
    assert p["name"] == "Jensen Huang"
    assert p["aliases"] == ["Jen-Hsun Huang"]
    assert len(p["source_urls"]) == 4
    assert {r["name"] for r in p["roles"]} == {"Jensen Huang", "Jen-Hsun Huang"}


def test_alternative_name_without_name_similarity_found_by_role_overlap():
    # "JenHsun" shares no prefix/nickname with "Jensen"; only surname + role link them.
    a, b = parse_name("JenHsun Huang"), parse_name("Jensen Huang")
    assert not entities.first_names_related(a.first, b.first)
    assert is_candidate_pair(a, b, ["Founder, President and CEO"],
                             ["Co-founder, President and Chief Executive Officer"])
    resolver = FakeResolver(same=True)
    people = resolve([rec("JenHsun Huang", "Founder, President and CEO", "https://a.example.com"),
                      HUANG[2]], "nvidia.com", resolver)
    assert len(people) == 1 and len(resolver.calls) == 1


def test_alternative_name_kept_separate_when_evidence_check_says_no():
    assert len(resolve(HUANG, "nvidia.com", FakeResolver(same=False))) == 2


# ---------------------------------------------------------------- earlier cases

@pytest.mark.parametrize("a,b,domain", [
    (rec("Dan Gallagher", "Chief Legal, Compliance and Corporate Affairs Officer",
         "https://investors.robinhood.com/governance/leadership"),
     rec("Daniel Gallagher", "Chief Legal Officer", "https://www.comparably.com/x"),
     "robinhood.com"),
    (rec("Vlad Tenev", "Co-Founder and CEO", "https://investors.robinhood.com/leadership"),
     rec("Vladimir Tenev", "Chair of the Board and Chief Executive Officer",
         "https://www.sec.gov/Archives/x.htm", as_of="2025-04-25"),
     "robinhood.com"),
    (rec("Baiju Bhatt", "Co-Founder", "https://robinhood.com/newsroom/x"),
     rec("Baiju Bhat", "Co-Founder", "https://fintechleaders.substack.com/p/x"),
     "robinhood.com"),
])
def test_known_variants_merge_with_company_spelling_canonical(a, b, domain):
    resolver = FakeResolver(same=True)
    (p,) = resolve([a, b], domain, resolver)
    assert len(resolver.calls) == 1
    assert p["name"] == a["name"]          # spelling used on the company's own domain
    assert p["aliases"] == [b["name"]]
    assert set(p["source_urls"]) == {a["source_url"], b["source_url"]}


# ---------------------------------------------------------------- no false merges

def test_same_surname_unrelated_roles_never_compared():
    people = resolve([
        rec("Jensen Huang", "President and CEO", "https://www.nvidia.com/a"),
        rec("Lisa Huang", "Vice President, Marketing", "https://www.nvidia.com/b"),
    ], "nvidia.com", never_called)
    assert len(people) == 2


def test_same_surname_and_same_generic_title_never_compared():
    people = resolve([
        rec("Mark Stevens", "Director", "https://www.nvidia.com/board"),
        rec("Mary Stevens", "Independent Director", "https://www.nvidia.com/board"),
    ], "nvidia.com", never_called)
    assert len(people) == 2


def test_same_title_different_surname_never_compared():
    people = resolve([
        rec("Bill Dally", "Chief Scientist", "https://www.nvidia.com/research"),
        rec("David Kirk", "Chief Scientist", "https://www.nvidia.com/research"),
    ], "nvidia.com", never_called)
    assert len(people) == 2


def test_same_surname_overlapping_role_is_only_a_candidate():
    # Two different people could share a surname and department: the evidence check decides.
    resolver = FakeResolver(same=False)
    people = resolve([
        rec("Amy Chen", "VP, Hardware Engineering", "https://www.nvidia.com/a"),
        rec("Brian Chen", "Senior Director, Hardware Engineering", "https://www.nvidia.com/b"),
    ], "nvidia.com", resolver)
    assert len(resolver.calls) == 1 and len(people) == 2


# ---------------------------------------------------------------- stored data & IDs

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    import database as db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return db


def _legacy_rows(db, cid):
    """Rows as stored by the pre-entity-resolution pipeline: no role history, status NULL."""
    with db.get_conn() as conn:
        for name, title, url, urls in [
            ("Jensen Huang", "Co-founder, President and Chief Executive Officer",
             "https://www.nvidia.com/en-us/about-nvidia/board-of-directors/jensen-huang",
             "https://www.nvidia.com/en-us/about-nvidia/board-of-directors/jensen-huang\n"
             "https://www.nvidia.com/en-us/about-nvidia/"),
            ("Jen-Hsun Huang", "Founder, President and Chief Executive Officer",
             "https://vectorshift.ai/research/companies/nvidia/people",
             "https://vectorshift.ai/research/companies/nvidia/people"),
            ("Colette Kress", "EVP and Chief Financial Officer",
             "https://www.nvidia.com/en-us/about-nvidia/governance/management-team/colette-kress",
             "https://www.nvidia.com/en-us/about-nvidia/governance/management-team/colette-kress"),
        ]:
            conn.execute("""INSERT INTO leaders (company_id, name, title, department, seniority,
                                                 source_url, source_urls, evidence)
                            VALUES (?, ?, ?, '', 'executive', ?, ?, ?)""",
                         (cid, name, title, url, urls, f"{name} {title}"))
        return {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM leaders")}


def test_reresolution_merges_existing_records_and_keeps_lowest_id(tmp_db):
    import rag
    db = tmp_db
    cid = db.upsert_company("NVIDIA", "https://www.nvidia.com/en-us/")
    old_ids = _legacy_rows(db, cid)
    assert db.get_company(cid)["resolved_version"] == 0

    stats = rag.reresolve_company(cid, resolver=FakeResolver(same=True), log=lambda m: None)
    assert stats == {"checked": 1, "merged": 1, "failed": 0}

    people = {p["name"]: p for p in db.get_leaders(cid)}
    assert set(people) == {"Jensen Huang", "Colette Kress"}
    huang = people["Jensen Huang"]
    assert huang["person_id"] == min(old_ids["Jensen Huang"], old_ids["Jen-Hsun Huang"])
    assert people["Colette Kress"]["person_id"] == old_ids["Colette Kress"]  # untouched ID
    assert huang["aliases"] == ["Jen-Hsun Huang"]
    assert len(huang["source_urls"].split("\n")) == 3  # incl. the extra legacy source
    assert db.get_company(cid)["resolved_version"] == entities.RESOLUTION_VERSION

    # Idempotent: a second run keeps the same people, IDs, and aliases.
    rag.reresolve_company(cid, resolver=never_called, log=lambda m: None)
    again = {p["name"]: p for p in db.get_leaders(cid)}
    assert {n: p["person_id"] for n, p in again.items()} == \
        {n: p["person_id"] for n, p in people.items()}
    assert again["Jensen Huang"]["aliases"] == ["Jen-Hsun Huang"]


def test_failed_check_leaves_company_marked_for_retry(tmp_db):
    import rag
    db = tmp_db
    cid = db.upsert_company("NVIDIA", "https://www.nvidia.com/en-us/")
    _legacy_rows(db, cid)

    def broken(a, b):
        raise RuntimeError("API down")
    stats = rag.reresolve_company(cid, resolver=broken, log=lambda m: None)
    assert stats["failed"] == 1
    assert len(db.get_leaders(cid)) == 3                     # nothing merged
    assert db.get_company(cid)["resolved_version"] == 0      # retried next time


def test_db_table_and_chat_share_canonical_person_ids(tmp_db):
    import rag
    db = tmp_db
    cid = db.upsert_company("NVIDIA", "https://www.nvidia.com/en-us/")
    _legacy_rows(db, cid)
    rag.reresolve_company(cid, resolver=FakeResolver(same=True), log=lambda m: None)

    leader_ids = {p["person_id"] for p in db.get_leaders(cid)}
    history_ids = {r["person_id"] for r in db.get_role_history(cid)}
    assert history_ids == leader_ids
    ctx = rag.build_context(cid, "Who is the CEO?")
    assert "UNIQUE PEOPLE: 2" in ctx
    for pid in leader_ids:
        assert ctx.count(f"[P{pid}]") == 1
