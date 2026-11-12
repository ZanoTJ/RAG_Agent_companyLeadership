"""Gemini-powered extraction (ingest) and question answering (retrieve + generate)."""

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

import database as db
import scraper
import search

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Tried in order if the primary model is overloaded (503) or rate-limited (429).
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
_client: genai.Client | None = None
_client_lock = threading.Lock()


def client() -> genai.Client:
    global _client
    with _client_lock:  # extraction runs in threads; build exactly one client
        if _client is None:
            _client = _make_client()
    return _client


def _make_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    # Bounded timeout/retries so an overloaded model fails fast instead of hanging.
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=60_000, retry_options=types.HttpRetryOptions(attempts=2)
        ),
    )


def generate(contents, config: types.GenerateContentConfig):
    """generate_content with fallback to other models on transient errors."""
    models = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]
    last_err = None
    for model in models:
        try:
            return client().models.generate_content(model=model, contents=contents, config=config)
        except genai.errors.APIError as e:
            if e.code not in (404, 429, 500, 503, 504):
                raise
            last_err = e
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


# ================================================================ extraction

class Leader(BaseModel):
    name: str
    title: str
    department: str
    seniority: Literal["executive", "vp", "department_head", "board", "other"]
    evidence: str


class LeaderList(BaseModel):
    organization_on_page: str      # the organization the page actually describes
    same_company: bool             # is that the company behind the given website?
    leaders: list[Leader]


EXTRACT_PROMPT = """You extract leadership information from a web page.

Company: {company}
Company website: {domain}
What the company does (from its homepage): {context}
Page URL: {url}

List every person on this page who is a CURRENT leader of THIS company:
C-suite executives, presidents, founders, SVPs/EVPs/VPs, department heads,
directors who lead a function, and board members.

Rules:
- Only include people explicitly named on the page with a role at this company.
- First decide which organization the page describes (organization_on_page) and whether
  it is the same company as the website above (same_company). Many companies share
  names: a different full name (e.g. "{company} Interactive", "{company} Labs"), a
  different website, or a different industry/product means same_company=false.
  If same_company is false or you are unsure, set it to false and return an empty list.
- Exclude former employees ("former", "ex-", "previously") and people at other companies.
- Copy names and titles exactly as written. Do not guess, infer, or invent titles.
- evidence: a short verbatim quote (copied character-for-character from the page text)
  that shows this person's name and role. Do not paraphrase.
- department: the function they lead (e.g. "Engineering", "Finance", "Sales").
  Use "Executive" for the CEO/President, "Board" for board members, "" if unclear.
- seniority: "executive" (C-suite, President, Founder in an operating role),
  "vp" (any Vice President level), "department_head" (Head of / Director leading a team),
  "board" (board of directors), "other" otherwise.
- Ignore customers, investors, authors of blog posts, and testimonial quotes.
- If nobody qualifies, return an empty list.

PAGE TEXT:
---
{text}
---"""


def _loose(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace - for tolerant substring checks."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def extract_leaders(company: str, domain: str, context: str,
                    page: scraper.Page) -> tuple[list[dict], int]:
    """Return (verified people, number rejected by grounding checks)."""
    resp = generate(
        contents=EXTRACT_PROMPT.format(company=company, domain=domain, context=context,
                                       url=page.url, text=page.text),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=LeaderList,
        ),
    )
    parsed: LeaderList | None = resp.parsed
    if not parsed:
        return [], 0
    if not parsed.same_company:  # page describes a different organization
        return [], len(parsed.leaders)

    # Grounding checks: the name AND the evidence quote must appear in the page text,
    # and the title must name an actual leadership role (not just "engineering").
    haystack = _loose(page.text)
    out, rejected = [], 0
    for p in parsed.leaders:
        name, quote = _loose(p.name), _loose(p.evidence)
        if (name and name in haystack and quote and quote in haystack
                and scraper.mentions_roles(p.title)):
            out.append({**p.model_dump(), "source_url": page.url})
        else:
            rejected += 1
    return out, rejected


# ---------------------------------------------------------------- dedup

_HONORIFICS = {"dr", "mr", "mrs", "ms", "prof", "sir"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "mba", "cpa", "esq"}
_SENIORITY_RANK = {"executive": 0, "board": 1, "vp": 2, "department_head": 3, "other": 4}


def _person_key(name: str) -> str:
    """'Dr. Jane Q. Doe, PhD' / 'Jane (JD) Doe' -> 'jane doe'."""
    name = re.sub(r"\(.*?\)", " ", name)
    tokens = [t for t in _loose(name).split()
              if t not in _HONORIFICS and t not in _SUFFIXES and len(t) > 1]
    return f"{tokens[0]} {tokens[-1]}" if len(tokens) >= 2 else " ".join(tokens)


def dedupe(people: list[dict], company_domain: str, company_name: str) -> list[dict]:
    """Merge records of the same person across sources.

    Records without a verifiable full name (e.g. "John", "S. C.") are dropped.
    The primary record (name, title) comes from the company's own site when
    available, otherwise the most senior / most specific title. All sources are kept.
    """
    at_company = re.compile(rf"\s+(at|@)\s+{re.escape(company_name)}\.?$", re.I)
    groups: dict[str, list[dict]] = {}
    for p in people:
        key = _person_key(p["name"])
        if len(key.split()) < 2:
            continue
        p["title"] = at_company.sub("", (p.get("title") or "").strip())  # "CTO at Acme" -> "CTO"
        groups.setdefault(key, []).append(p)

    merged = []
    for recs in groups.values():
        recs.sort(key=lambda r: (
            scraper.bare_domain(r["source_url"]) != company_domain,  # own site first
            _SENIORITY_RANK.get(r.get("seniority"), 4),
            -len(r.get("title") or ""),
        ))
        best = dict(recs[0])
        best["department"] = best.get("department") or next(
            (r["department"] for r in recs if r.get("department")), "")
        best["source_urls"] = list(dict.fromkeys(r["source_url"] for r in recs))
        merged.append(best)
    return merged


# ---------------------------------------------------------------- pipeline

def ingest_company(url: str, log=print) -> int:
    """Crawl site + Wikipedia + Tavily search -> Gemini extraction -> dedupe -> SQLite."""
    db.init_db()
    try:
        base, name, pages = scraper.crawl_company_site(url, log=log)
    except RuntimeError as e:
        if not search.tavily_available():
            raise
        log(f"{e} Continuing with web search only.")
        base = scraper.normalize_url(url)
        name = scraper.bare_domain(base).split(".")[0].title()
        pages = []
    domain = scraper.bare_domain(base)
    context = (scraper.site_description(pages[0].text) if pages else "") or "(unknown)"

    wiki = scraper.fetch_wikipedia(name, base, log=log)
    if wiki:
        pages.append(wiki)
    pages += search.discover(name, base, seen_urls={p.url for p in pages}, log=log)

    company_id = db.upsert_company(name, base)
    db.clear_company_data(company_id)
    for page in pages:
        db.save_source(company_id, page.url, page.text)

    # Extract in parallel; log from this thread (Streamlit can't log from workers).
    candidates = [p for p in pages if scraper.mentions_roles(p.text)]
    found: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(extract_leaders, name, domain, context, p): p
                   for p in candidates}
        for fut in as_completed(futures):
            page = futures[fut]
            try:
                people, rejected = fut.result()
            except Exception as e:  # keep going if one page fails
                log(f"Extraction failed for {page.url}: {e}")
                continue
            found += people
            note = f", {rejected} rejected (not supported by page text)" if rejected else ""
            log(f"{page.url}: {len(people)} verified{note}")

    leaders = dedupe(found, domain, name)
    stored = db.save_leaders(company_id, leaders)
    log(f"Done. {len(found)} records -> {stored} unique leaders stored for {name}.")
    return company_id


# ================================================================ answering

SYSTEM_PROMPT = """You are a research assistant answering questions about a company's leadership.

Answer ONLY from the CONTEXT provided (a leadership table extracted from public
web pages, plus excerpts of those pages). Rules:
- Cite the source URL for each person or fact you mention, like [source](URL).
- If the context does not contain the answer, say so plainly. Never guess names or titles.
- Sources include third-party sites (news, LinkedIn, data aggregators). Prefer the company's
  own website when sources disagree. If two people appear to hold the same role, report both
  with their sources and say the information may be conflicting or outdated.
- Note that the data reflects the pages at the time they were collected and may be outdated.
- Be concise. Use bullet points or a short table for lists of people."""


def _chunks(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    return [text[i:i + size] for i in range(0, max(len(text), 1), size - overlap)]


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


def retrieve_excerpts(company_id: int, question: str, k: int = 4) -> list[tuple[str, str]]:
    """Simple keyword-overlap retrieval over stored page text."""
    q = _tokens(question)
    scored = []
    for src in db.get_sources(company_id):
        for chunk in _chunks(src["content"] or ""):
            score = len(q & _tokens(chunk))
            if score:
                scored.append((score, src["url"], chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(url, chunk) for _, url, chunk in scored[:k]]


def build_context(company_id: int, question: str) -> str:
    company = next((c for c in db.list_companies() if c["id"] == company_id), None)
    leaders = db.get_leaders(company_id)
    lines = [f"COMPANY: {company['name']} ({company['url']}), collected {company['scraped_at']}"]
    lines.append("\nLEADERSHIP TABLE (name | title | department | seniority | sources):")
    if leaders:
        for p in leaders:
            sources = ", ".join((p["source_urls"] or p["source_url"]).split("\n"))
            lines.append(f"- {p['name']} | {p['title']} | {p['department']} | "
                         f"{p['seniority']} | {sources}")
            if p.get("evidence"):
                lines.append(f'    evidence: "{p["evidence"]}"')
    else:
        lines.append("(no leaders found)")
    excerpts = retrieve_excerpts(company_id, question)
    if excerpts:
        lines.append("\nPAGE EXCERPTS:")
        for url, chunk in excerpts:
            lines.append(f"[from {url}]\n{chunk}\n")
    return "\n".join(lines)


def answer(company_id: int, question: str, history: list[dict] | None = None) -> str:
    contents = []
    for msg in (history or [])[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    prompt = f"CONTEXT:\n{build_context(company_id, question)}\n\nQUESTION: {question}"
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    resp = generate(
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
    )
    return resp.text or "Sorry, I couldn't generate an answer."
