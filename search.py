"""Tavily-powered web search: discovery of pages that mention a company's leaders
(at collection time) and targeted searches for answer-time fallback.

Complements scraper.py for sites without a crawlable leadership page. Queries are
built only from the company name and domain - nothing company-specific is hardcoded.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from scraper import MAX_CHARS_PER_PAGE, Page, bare_domain

load_dotenv()

# (query template, where to search). Phrased to surface primary sources - the
# company's own pages (incl. blog posts naming officers), appointment announcements,
# filings, news, and public profiles. Every result is still evaluated by source_policy.
# where: None = whole web, "site" = the company's own domain, or a list of domains.
QUERY_TEMPLATES = [
    ('"{name}" {domain} leadership team executives', None),
    ('"{name}" executive officers directors SEC filing proxy', None),
    ("{name} {domain} leadership team", ["linkedin.com"]),
    
    ('"{name}" {domain} CEO founder co-founder', None),
    ('"{name}" {domain} CEO CTO CFO COO CMO', None),

    ('"{name}" {domain} vice president OR "head of"', None),
    ('"{name}" {domain} VP engineering product marketing sales', None),

    ('"{name}" {domain} chief marketing officer head of marketing', None),

    ('"{name}" {domain} "head of" department', None),
    ('"{name}" {domain} head of engineering product technology', None),
    ('"{name}" {domain} head of marketing sales partnerships', None),
    ('"{name}" {domain} head of finance operations people', None),
]
RESULTS_PER_QUERY = 5
RESULTS_RESTRICTED = 10  # site- or domain-restricted queries
SEARCH_WORKERS = 5  # concurrent Tavily requests


def page_key(url: str) -> str:
    """Same page under country subdomains or www. (e.g. fx.linkedin.com/company/x vs
    www.linkedin.com/company/x) gets one key: registrable domain + path."""
    from urllib.parse import urlparse

    from entities import registrable_domain
    return registrable_domain(url) + urlparse(url).path.rstrip("/").lower()


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
    seen = {page_key(u) for u in (seen_urls or set())}
    pages: list[Page] = []

    def run(template_where):
        template, where = template_where
        query = template.format(name=company_name, domain=domain)
        try:
            return query, client.search(
                query,
                search_depth="advanced",
                max_results=RESULTS_RESTRICTED if where else RESULTS_PER_QUERY,
                include_raw_content="text",
                include_domains=[domain] if where == "site" else where,
            ), None
        except Exception as e:  # network / quota errors shouldn't abort ingestion
            return query, None, e

    # Queries run concurrently; results are merged in template order so dedup and the
    # page list are the same as a sequential run.
    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
        responses = list(pool.map(run, QUERY_TEMPLATES))

    for query, resp, err in responses:
        if err is not None:
            log(f"Tavily search failed for '{query}': {err}")
            continue

        new = 0
        for r in resp.get("results", []):
            url = (r.get("url") or "").rstrip("/")
            if not url or page_key(url) in seen:
                continue
            # Prefer full page text; fall back to the search snippet.
            text = (r.get("raw_content") or r.get("content") or "").strip()
            # Cheap relevance filter: the page must at least mention the company.
            if company_name.lower() not in text.lower() and domain not in text.lower():
                continue
            seen.add(page_key(url))
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
