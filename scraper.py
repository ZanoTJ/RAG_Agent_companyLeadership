"""Find and fetch public pages likely to list a company's leadership.

Sources:
  1. The company's own website: leadership / team / about pages, found by
     scoring links on the homepage (and one level below) plus common paths.
  2. Wikipedia: the company's article (infobox "Key people" + intro), accepted
     only if the article links to the company's domain.
"""

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 LeadershipRAG/0.1"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 15
MAX_PAGES = 8               # company pages to fetch (excluding homepage)
MAX_CHARS_PER_PAGE = 30_000

# Keyword -> score. Matched against link path and anchor text.
LINK_KEYWORDS = {
    "leadership": 10, "executive": 9, "management": 8, "our-team": 8, "team": 6,
    "board": 6, "directors": 6, "officers": 6, "founders": 6, "people": 5,
    "governance": 4, "about": 3, "company": 2, "who-we-are": 3, "investor": 2,
}
COMMON_PATHS = [
    "/leadership", "/about/leadership", "/company/leadership", "/about-us/leadership",
    "/team", "/our-team", "/about/team", "/about", "/about-us", "/company",
    "/management", "/executives",
]
SKIP_EXT = re.compile(r"\.(pdf|jpe?g|png|gif|svg|zip|mp4|mp3|docx?|xlsx?)$", re.I)
# Content-style sections that mention "team" etc. but rarely list leadership.
NEGATIVE_PATH = re.compile(
    r"/(blog|news|research|press|articles?|posts?|stories|insights|resources|careers?|"
    r"jobs|events?|webinars?|podcasts?|search|tags?|category|customers?|case-stud\w*|"
    r"docs?|support|help|legal|privacy|terms|login|signup)(/|$)", re.I
)

# Official blog / newsroom: crawled separately from the leadership-page search, because
# company posts often announce founders, hires and appointments ("X joins as CFO").
MAX_BLOG_POSTS = 8
BLOG_INDEX_PATHS = ["/blog", "/news", "/newsroom", "/press", "/company/blog", "/about/news"]
_BLOG_SECTION = re.compile(r"^/(?:[\w-]+/)?(blog|news|newsroom|press|press-releases|updates|"
                           r"announcements|stories)/?$", re.I)
_BLOG_NOT_POST = re.compile(r"/(category|categories|tags?|authors?|page|topics?|feed|rss)(/|$)",
                            re.I)
# Post titles/slugs that tend to name leaders.
_LEADERSHIP_POST = {
    "join": 5, "welcom": 5, "appoint": 6, "hire": 4, "named": 4, "promot": 5, "new-": 1,
    "introduc": 3, "meet": 3, "founder": 5, "ceo": 5, "cfo": 5, "cto": 5, "coo": 5,
    "chief": 5, "vp": 4, "vice-president": 4, "head-of": 4, "leader": 5, "executive": 5,
    "board": 4, "advisor": 2, "stealth": 4, "launch": 2, "series": 3, "seed": 2,
    "funding": 3, "raise": 3, "announc": 2, "team": 2, "our-story": 4, "why-we": 3,
}
_DATE_META = ("article:published_time", "og:published_time", "datePublished",
              "publish_date", "pubdate", "date")

# Used to skip pages that clearly mention no leadership roles (saves LLM calls).
ROLE_PATTERN = re.compile(
    r"\b(CEO|CFO|CTO|COO|CMO|CIO|CISO|CRO|CPO|Chief|President|Vice President|VP|"
    r"SVP|EVP|Director|Head of|Founder|Co-?founder|Chair(man|woman|person)?|"
    r"Officer|General Counsel|Managing Director|Partner|Board|Directors)\b"
)


@dataclass
class Page:
    url: str
    text: str
    published: str = ""   # publication date, when the source provides one


# ---------------------------------------------------------------- utilities

def normalize_url(url: str) -> str:
    """'acme.com/x' -> 'https://acme.com'."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc.lower()}"


def bare_domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _robots(base: str) -> RobotFileParser | None:
    rp = RobotFileParser()
    try:
        r = requests.get(urljoin(base, "/robots.txt"), headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        rp.parse(r.text.splitlines())
        return rp
    except requests.RequestException:
        return None


def fetch(url: str, robots: RobotFileParser | None = None) -> tuple[str, str] | None:
    """Return (final_url, html) or None. Respects robots.txt when provided."""
    if robots and not robots.can_fetch(HEADERS["User-Agent"], url):
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None
    if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
        return None
    return r.url, r.text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text[:MAX_CHARS_PER_PAGE]


def company_name_from_html(html: str, fallback_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    stem = bare_domain(fallback_url).split(".")[0]
    og = soup.find("meta", property="og:site_name")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.string:
        # "Home \ Acme" / "Acme Corp | Home" -> "Acme" / "Acme Corp"
        parts = [p.strip() for p in re.split(r"\s[|\-–—:\\/·]\s", soup.title.string.strip())]
        parts = [p for p in parts if p and p.lower() not in {"home", "homepage", "welcome"}]
        for p in parts:  # prefer the segment that matches the domain
            if stem.lower() in p.lower().replace(" ", ""):
                return p
        if parts:
            return parts[0]
    return stem.title()


def _score_links(html: str, page_url: str, domain: str) -> dict[str, int]:
    soup = BeautifulSoup(html, "html.parser")
    scores: dict[str, int] = {}
    for a in soup.find_all("a", href=True):
        url, _ = urldefrag(urljoin(page_url, a["href"]))
        parsed = urlparse(url)
        if (bare_domain(url) != domain or parsed.query or SKIP_EXT.search(parsed.path)
                or NEGATIVE_PATH.search(parsed.path)):
            continue
        haystack = (parsed.path + " " + a.get_text(" ")).lower()
        score = sum(w for kw, w in LINK_KEYWORDS.items() if kw in haystack)
        depth = len([seg for seg in parsed.path.split("/") if seg])
        score -= 2 * max(0, depth - 2)  # leadership pages are usually shallow
        if score > 0:
            url = url.rstrip("/")
            scores[url] = max(scores.get(url, 0), score)
    return scores


# ---------------------------------------------------------------- public API

def crawl_company_site(url: str, log=print) -> tuple[str, str, list[Page]]:
    """Return (base_url, company_name, pages) for the company's own website."""
    base = normalize_url(url)
    robots = _robots(base)
    home = fetch(base, robots)
    if not home:
        raise RuntimeError(f"Could not fetch {base} (blocked, offline, or not HTML).")
    base_final, home_html = home
    domain = bare_domain(base_final)
    name = company_name_from_html(home_html, base_final)
    log(f"Company: {name} ({domain})")

    # Level 1: links on homepage + common paths.
    candidates = _score_links(home_html, base_final, domain)
    for path in COMMON_PATHS:
        u = urljoin(base_final, path).rstrip("/")
        candidates.setdefault(u, 3)

    pages = [Page(base_final, html_to_text(home_html))]
    visited = {base_final.rstrip("/")}

    # Fetch best candidates; leadership pages often hang off /about, so also
    # harvest links from fetched pages (level 2) and re-rank as we go.
    while candidates and len(pages) <= MAX_PAGES:
        best = max(candidates, key=candidates.get)
        candidates.pop(best)
        if best in visited:
            continue
        visited.add(best)
        res = fetch(best, robots)
        if not res:
            continue
        final_url, html = res
        if bare_domain(final_url) != domain:
            continue  # redirected off-site
        if final_url.rstrip("/") in visited - {best}:
            continue  # redirected to a page we already have
        visited.add(final_url.rstrip("/"))
        pages.append(Page(final_url, html_to_text(html)))
        log(f"Fetched {final_url}")
        for u, s in _score_links(html, final_url, domain).items():
            if u not in visited:
                candidates[u] = max(candidates.get(u, 0), s)

    pages += crawl_company_blog(base_final, home_html, domain, robots, visited, log=log)
    return base_final, name, pages


# ---------------------------------------------------------------- official blog

def _on_site(url: str, domain: str) -> bool:
    """The company's domain or one of its subdomains (blog.acme.com)."""
    host = bare_domain(url)
    return host == domain or host.endswith("." + domain)


def _blog_indexes(home_html: str, base: str, domain: str) -> list[str]:
    """Blog/newsroom index pages: links from the homepage (query strings kept, e.g.
    /blog?category=&showAll=true lists every post) plus the usual paths."""
    found = []
    for a in BeautifulSoup(home_html, "html.parser").find_all("a", href=True):
        url, _ = urldefrag(urljoin(base, a["href"]))
        p = urlparse(url)
        if _on_site(url, domain) and (_BLOG_SECTION.match(p.path or "/")
                                      or (bare_domain(url) != domain
                                          and bare_domain(url).split(".")[0] in
                                          ("blog", "news", "newsroom", "press")
                                          and p.path in ("", "/"))):
            found.append(url.rstrip("/"))
    found += [urljoin(base, path) for path in BLOG_INDEX_PATHS]
    return list(dict.fromkeys(found))


def _blog_post_links(html: str, index_url: str, domain: str) -> dict[str, int]:
    """Post links on a blog index -> leadership-likelihood score (0 = keep as filler)."""
    index = urlparse(index_url)
    section = index.path.rstrip("/")
    posts: dict[str, int] = {}
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        url, _ = urldefrag(urljoin(index_url, a["href"]))
        p = urlparse(url)
        if not _on_site(url, domain) or p.query or SKIP_EXT.search(p.path) \
                or _BLOG_NOT_POST.search(p.path):
            continue
        same_host = p.netloc.lower() == index.netloc.lower()
        if section and not p.path.lower().startswith(section.lower() + "/"):
            continue  # not under /blog/
        if not section and not (same_host and p.path.strip("/")):
            continue
        hay = (p.path + " " + a.get_text(" ")).lower().replace(" ", "-")
        url = url.rstrip("/")
        posts[url] = max(posts.get(url, 0),
                         sum(w for kw, w in _LEADERSHIP_POST.items() if kw in hay))
    return posts


def published_date(html: str) -> str:
    """Publication date from <meta> tags or the first <time datetime>, else ""."""
    soup = BeautifulSoup(html, "html.parser")
    for key in _DATE_META:
        tag = soup.find("meta", attrs={"property": key}) or soup.find(
            "meta", attrs={"name": key}) or soup.find(attrs={"itemprop": key})
        val = tag and (tag.get("content") or tag.get("datetime"))
        if val:
            return val.strip()[:10]
    t = soup.find("time", datetime=True)
    return t["datetime"].strip()[:10] if t else ""


def crawl_company_blog(base: str, home_html: str, domain: str, robots=None,
                       visited: set | None = None, log=print) -> list[Page]:
    """Posts from the company's official blog/newsroom that mention leadership roles.
    Posts whose title suggests a leadership announcement are fetched first."""
    visited = visited if visited is not None else set()
    posts: dict[str, int] = {}
    pages: list[Page] = []
    for index in _blog_indexes(home_html, base, domain):
        if index.rstrip("/") in visited:
            continue
        visited.add(index.rstrip("/"))
        res = fetch(index, robots)
        if not res or not _on_site(res[0], domain):
            continue
        found = _blog_post_links(res[1], res[0], domain)
        if found:
            log(f"Blog index {res[0]}: {len(found)} posts")
            text = html_to_text(res[1])
            if mentions_roles(text):  # author bylines: "Jane Doe • Head of Product"
                pages.append(Page(res[0], text))
            for u, s in found.items():
                posts[u] = max(posts.get(u, 0), s)
    fetched = 0
    for url in sorted(posts, key=lambda u: -posts[u]):
        if fetched >= MAX_BLOG_POSTS:
            break
        if url in visited:
            continue
        visited.add(url)
        res = fetch(url, robots)
        if not res or not _on_site(res[0], domain):
            continue
        text = html_to_text(res[1])
        if not mentions_roles(text):
            continue
        pages.append(Page(res[0], text, published_date(res[1])))
        fetched += 1
        log(f"Fetched blog post {res[0]}")
    return pages


def fetch_wikipedia(company_name: str, company_url: str, log=print) -> Page | None:
    """Fetch the company's Wikipedia article if it links to the company's domain."""
    api = "https://en.wikipedia.org/w/api.php"
    domain = bare_domain(company_url)
    try:
        r = requests.get(
            api,
            params={"action": "query", "list": "search", "srsearch": f"{company_name} company",
                    "srlimit": 3, "format": "json"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        results = r.json().get("query", {}).get("search", [])
    except (requests.RequestException, ValueError):
        return None

    for hit in results:
        page_url = "https://en.wikipedia.org/wiki/" + hit["title"].replace(" ", "_")
        res = fetch(page_url)
        if not res:
            continue
        soup = BeautifulSoup(res[1], "html.parser")
        infobox = soup.select_one("table.infobox")
        if not infobox or domain not in str(infobox):
            continue  # not the right company
        intro = []
        for p in soup.select("div.mw-parser-output > p"):
            if p.get_text(strip=True):
                intro.append(p.get_text(" ", strip=True))
            if len(intro) >= 4:
                break
        text = "INFOBOX:\n" + html_to_text(str(infobox)) + "\n\nINTRO:\n" + "\n".join(intro)
        log(f"Fetched {page_url}")
        return Page(page_url, text[:MAX_CHARS_PER_PAGE])
    return None


def site_description(homepage_text: str, max_chars: int = 600) -> str:
    """Short description of what the company does: its title line plus the first
    sentence-like lines of the homepage (skipping short nav labels)."""
    lines = homepage_text.splitlines()
    picked = lines[:1] + [ln for ln in lines[1:] if len(ln.split()) >= 6]
    return " ".join(picked)[:max_chars]


def mentions_roles(text: str) -> bool:
    return bool(ROLE_PATTERN.search(text))
