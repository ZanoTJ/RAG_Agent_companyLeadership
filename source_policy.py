"""Decide whether a fetched page may be used as leadership evidence.

Two layers, applied before extraction and storage:

1. Deterministic rules (no API calls):
   - reject pages with no readable content or behind a login wall;
   - accept the company's own domain (any subdomain) as "official";
   - accept government domains (e.g. sec.gov filings) as "regulatory";
   - reject pages that look like people/contact-data listings (email formats,
     phone numbers, "reveal contact" prompts) - judged by URL and content
     patterns, not by a domain blocklist.
2. LLM assessment (batched) for everything else: the classifier labels each page
   (news, reference, public professional profile, people-data aggregator, ...),
   whether it is about this company (not a namesake), and whether it is credible.
   Only credible pages about the company, of accepted types, are used. The LLM can
   never make a page "official" or "regulatory": those labels (and the trust that
   comes with them) are decided by the URL rules alone.

If the classifier is unavailable or fails, ambiguous pages are rejected:
unverified sources are never used as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from entities import registrable_domain

SOURCE_TYPES = ["official", "regulatory", "news", "reference", "professional_profile",
                "people_data_aggregator", "other"]
ACCEPTED_TYPES = {"official", "regulatory", "news", "reference", "professional_profile"}
SOURCE_LABEL = {
    "official": "Company website", "regulatory": "Regulatory filing", "news": "News",
    "reference": "Reference", "professional_profile": "Public profile",
}

MIN_TEXT_CHARS = 200
_LOGIN_URL = re.compile(r"/(login|signin|sign-in|authwall|checkpoint|uas/login)(/|\?|$)", re.I)
_LOGIN_TEXT = re.compile(
    r"(sign in|log in|login|join now|create (an|your|a free) account) to (see|view|continue|"
    r"access|read)|you must be logged in|please (sign|log) in to", re.I)
_PEOPLE_DATA_URL = re.compile(
    r"(email[-_]?format|/emails?([-_/]|$)|contact[-_]?info|employee[-_]?directory|"
    r"org[-_]?chart|/contacts([-_/]|$)|phone[-_]?numbers?)", re.I)
_PEOPLE_DATA_TEXT = [re.compile(p, re.I) for p in (
    r"email formats?", r"e-?mail address(es)?", r"phone numbers?", r"direct dials?",
    r"contact (info|information|details)", r"(reveal|unlock|get|view) (the )?(email|contact|phone)",
    r"verified e-?mails?", r"employee directory", r"org(anizational)? chart",
    r"@\[?[a-z0-9-]+\]?\.(com|ai|io)\b", r"\b(first|last)[._]?(name|initial)\b.*@",
)]


@dataclass
class Verdict:
    accepted: bool
    source_type: str
    reason: str
    decided_by: str  # "rule" | "llm" | "fallback"


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().split(":")[0]


def rule_verdict(url: str, text: str, company_domain: str,
                 check_access: bool = True) -> Verdict | None:
    """Deterministic decision, or None if the page needs an LLM assessment."""
    text = text or ""
    path = urlparse(url).path
    if check_access:
        if len(text.strip()) < MIN_TEXT_CHARS:
            return Verdict(False, "other", "no readable public content", "rule")
        if _LOGIN_URL.search(path) or (_LOGIN_TEXT.search(text[:4000]) and len(text) < 4000):
            return Verdict(False, "other", "requires login to read", "rule")
    domain = registrable_domain(url)
    if domain == registrable_domain(company_domain):
        return Verdict(True, "official", "company's own website", "rule")
    if _host(url).endswith(".gov"):
        return Verdict(True, "regulatory", "government / regulatory source", "rule")
    hits = sum(bool(p.search(text)) for p in _PEOPLE_DATA_TEXT)
    if _PEOPLE_DATA_URL.search(url) or hits >= 3:
        return Verdict(False, "people_data_aggregator", "contact/people-data listing", "rule")
    return None


# classifier(items) -> one (source_type, credible, reason) per item, same order.
# Each item is {"url", "title", "text"}.
Classifier = Callable[[list[dict]], list[tuple[str, bool, str]]]


def evaluate(items: list[dict], company_domain: str, classifier: Classifier | None,
             check_access: bool = True,
             log: Callable[[str], None] = lambda _m: None) -> list[Verdict]:
    """Verdict for every item (dicts with url/title/text), in order."""
    verdicts: list[Verdict | None] = [
        rule_verdict(it["url"], it.get("text", ""), company_domain, check_access) for it in items]
    pending = [i for i, v in enumerate(verdicts) if v is None]
    if pending:
        results = None
        if classifier is not None:
            try:
                results = classifier([items[i] for i in pending])
                if len(results) != len(pending):
                    raise ValueError("classifier returned the wrong number of results")
            except Exception as e:
                log(f"Source assessment failed ({e}); unverified sources skipped.")
                results = None
        for n, i in enumerate(pending):
            if results is None:
                verdicts[i] = Verdict(False, "other", "could not be verified", "fallback")
                continue
            stype, credible, reason = results[n]
            stype = trusted_type(items[i]["url"], stype, company_domain)
            verdicts[i] = Verdict(credible and stype in ACCEPTED_TYPES, stype, reason, "llm")
    return verdicts  # type: ignore[return-value]


def trusted_type(url: str, source_type: str, company_domain: str) -> str:
    """"official" only for the company's own domain and "regulatory" only for
    government domains; the same label from an LLM becomes "reference"."""
    source_type = source_type if source_type in SOURCE_TYPES else "other"
    if source_type in ("official", "regulatory"):
        if registrable_domain(url) == registrable_domain(company_domain):
            return "official"
        if _host(url).endswith(".gov"):
            return "regulatory"
        return "reference"
    return source_type


# ---------------------------------------------------------------- passages

def relevant_passages(text: str, terms: list[str], window: int = 600,
                      max_chars: int = 8000) -> str:
    """Keep only the text around mentions of `terms` (names, roles), merged and
    capped - instead of storing whole pages."""
    if not text or not terms:
        return ""
    spans = []
    low = text.lower()
    for term in {t.lower() for t in terms if t and len(t) > 2}:
        for m in re.finditer(re.escape(term), low):
            spans.append((max(0, m.start() - window), min(len(text), m.end() + window)))
    if not spans:
        return ""
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = "\n...\n".join(text[s:e].strip() for s, e in merged)
    return out[:max_chars]
