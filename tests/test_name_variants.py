"""Regression tests for alternative name forms, candidate generation, canonical
names. Offline: the LLM
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
    # Kept separate - but then two people claim the one-person CEO role, and the live
    # company page (current) supersedes the other claim.
    stats = {}
    people = resolve_people(HUANG, "nvidia.com", "", resolver=FakeResolver(same=False),
                            fetched_on=date(2026, 11, 12), stats=stats)
    assert [p["name"] for p in people] == ["Jensen Huang"]
    assert "CEO role superseded" in stats["excluded"][0]["reason"]


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
    (rec("Baiju Bhatt", "Co-Founder and Chief Creative Officer", "https://robinhood.com/newsroom/x"),
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
    assert people == []  # never compared, and board-only roles are not leadership


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
        rec("Brian Chen", "Senior Director, Hardware Engineering", "https://www.nvidia.com/b",
            seniority="department_head"),
    ], "nvidia.com", resolver)
    assert len(resolver.calls) == 1 and len(people) == 2
