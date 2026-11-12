"""Tavily-powered discovery of public pages that mention a company's leaders.

Complements scraper.py for sites without a crawlable leadership page. Queries are
built only from the company name and domain - nothing company-specific is hardcoded.
"""

import os

from dotenv import load_dotenv

from scraper import MAX_CHARS_PER_PAGE, Page, bare_domain

load_dotenv()

# (query template, restrict to company domain?)
QUERY_TEMPLATES = [
    ('"{name}" {domain} leadership team executives', False),
    ('"{name}" {domain} CEO founder co-founder', False),
    ('"{name}" {domain} CFO CTO COO chief officer', False),
    ('"{name}" {domain} vice president VP', False),
    ('"{name}" {domain} head of director department', False),
    ("{name} about us team leadership founders", True),
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
            pages.append(Page(url, f"{title}\n{text}"[:MAX_CHARS_PER_PAGE]))
            new += 1
        log(f"Search: {query} -> {new} new pages")
    return pages
