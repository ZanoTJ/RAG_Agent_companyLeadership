"""Tavily-powered web search: discovery of pages that mention a company's leaders
(at collection time) and targeted searches for answer-time fallback.

Complements scraper.py for sites without a crawlable leadership page. Queries are
built only from the company name and domain - nothing company-specific is hardcoded.
"""

import os

from dotenv import load_dotenv

from scraper import MAX_CHARS_PER_PAGE, Page, bare_domain

load_dotenv()

# (query template, restrict to company domain?). Phrased to surface primary
# sources - the company's own pages, press releases about appointments, filings,
# and news coverage. Every result is still evaluated by source_policy.
QUERY_TEMPLATES = [
    ('"{name}" {domain} leadership team executives', False),
    ('"{name}" {domain} CEO founder co-founder', False),
    ('"{name}" appoints OR names chief officer', False),
    ('"{name}" {domain} vice president OR "head of"', False),
    ('"{name}" executive officers directors SEC filing proxy', False),
    ("{name} leadership team management board", True),
]
RESULTS_PER_QUERY = 5


def tavily_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))


def discover(company_name: str, company_url: str, seen_urls: set[str] | None = None,
             log=print) -> list[Page]:
    """Run targeted Tavily searches and return pages (with full text) not already seen."""
    if not tavily_available():
        log("TAVILY_API_KEY not set - skipping web search.")
        return []
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    domain = bare_domain(company_url)
    seen = {u.rstrip("/") for u in (seen_urls or set())}
    pages: list[Page] = []

    for template, on_site in QUERY_TEMPLATES:
        query = template.format(name=company_name, domain=domain)
        try:
            resp = client.search(
                query,
                search_depth="advanced",
                max_results=RESULTS_PER_QUERY,
                include_raw_content="text",
                include_domains=[domain] if on_site else None,
            )
        except Exception as e:  # network / quota errors shouldn't abort ingestion
            log(f"Tavily search failed for '{query}': {e}")
            continue

        new = 0
        for r in resp.get("results", []):
            url = (r.get("url") or "").rstrip("/")
            if not url or url in seen:
                continue
            # Prefer full page text; fall back to the search snippet.
            text = (r.get("raw_content") or r.get("content") or "").strip()
            # Cheap relevance filter: the page must at least mention the company.
            if company_name.lower() not in text.lower() and domain not in text.lower():
                continue
            seen.add(url)
            title = r.get("title") or ""
            pages.append(Page(url, f"{title}\n{text}"[:MAX_CHARS_PER_PAGE],
                              published=r.get("published_date") or ""))
            new += 1
        log(f"Search: {query} -> {new} new pages")
    return pages


def web_search(query: str, max_results: int = 5) -> list[Page]:
    """One targeted search (answer-time fallback). Raises on API errors so the
    caller can tell "search failed" from "nothing found"."""
    if not tavily_available():
        raise RuntimeError("TAVILY_API_KEY not set")
    from tavily import TavilyClient

    resp = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
        query, search_depth="basic", max_results=max_results, include_raw_content="text")
    pages = []
    for r in resp.get("results", []):
        url = (r.get("url") or "").rstrip("/")
        text = (r.get("raw_content") or r.get("content") or "").strip()
        if url and text:
            pages.append(Page(url, f"{r.get('title') or ''}\n{text}"[:MAX_CHARS_PER_PAGE],
                              published=r.get("published_date") or ""))
    return pages
