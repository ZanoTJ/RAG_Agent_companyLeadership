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
import entities
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
    status: Literal["current", "former", "unknown"]
    as_of: str                     # date the page gives for this role, or ""
    evidence: str


class LeaderList(BaseModel):
    organization_on_page: str      # the organization the page actually describes
    same_company: bool             # is that the company behind the given website?
    page_date: str                 # publication/updated date shown on the page, or ""
    leaders: list[Leader]


EXTRACT_PROMPT = """You extract leadership information from a web page.

Company: {company}
Company website: {domain}
What the company does (from its homepage): {context}
Page URL: {url}

List every person on this page who is or was a leader of THIS company:
C-suite executives, presidents, founders, SVPs/EVPs/VPs, department heads,
directors who lead a function, and board members. One entry per person per title.

Rules:
- Only include people explicitly named on the page with a role at this company.
- First decide which organization the page describes (organization_on_page) and whether
  it is the same company as the website above (same_company). Many companies share
  names: a different full name (e.g. "{company} Interactive", "{company} Labs"), a
  different website, or a different industry/product means same_company=false.
  If same_company is false or you are unsure, set it to false and return an empty list.
- Exclude people at other companies.
- status: "former" if the page says the person no longer holds that title ("former",
  "ex-", "previously", "stepped down", "until 2020"); "current" if the page presents it
  as held at the time of writing; "unknown" if unclear. A person whose title changed gets
  one entry per title (e.g. the old title as "former", the new title as "current").
- as_of: the date the page associates with this role (e.g. "since March 2024",
  "2025-04-25"), else "". page_date: the page's publication or "last updated" date if
  shown, else "". Never invent dates.
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
            rec = {**p.model_dump(), "source_url": page.url}
            rec["as_of"] = p.as_of or parsed.page_date or page.published
            out.append(rec)
        else:
            rejected += 1
    return out, rejected


# ---------------------------------------------------------------- entity resolution (stage 2)

class SamePersonDecision(BaseModel):
    same_person: bool
    confidence: Literal["high", "medium", "low"]
    reason: str


RESOLVE_PROMPT = """Two groups of records were extracted from public web pages about the
leadership of {company} ({domain}). The names are similar but not identical. Decide
whether both groups describe the SAME individual.

Rules:
- A similar name alone is NOT enough. Base the decision on the titles, dates, and
  evidence quotes: e.g. the same or a consistent role (same title, the same founding
  story, a plausible title progression) supports same_person=true.
- Different roles that two different people could hold at the same time (e.g. one is
  "Head of Data Science", the other "Chief Legal Officer") mean same_person=false.
- Nicknames (Dan/Daniel) and one-letter spelling differences (a typo on one site)
  are common for the same person, but still require supporting evidence.
- confidence "high" only if the evidence clearly shows one person.

GROUP A:
{a}

GROUP B:
{b}"""


def _describe_cluster(recs: list[dict]) -> str:
    return "\n".join(
        f'- name: "{r["name"]}" | title: "{r["title"]}" | status: {r.get("status")} | '
        f'as_of: {r.get("as_of") or "?"} | source: {r["source_url"]}\n'
        f'  evidence: "{(r.get("evidence") or "")[:300]}"'
        for r in recs[:8])


def make_llm_resolver(company: str, domain: str):
    """Gemini-backed resolver for entities.resolve_people. Merges only on a
    high-confidence 'same person' decision."""
    def resolve(a: list[dict], b: list[dict]) -> tuple[bool, str]:
        resp = generate(
            contents=RESOLVE_PROMPT.format(company=company, domain=domain,
                                           a=_describe_cluster(a), b=_describe_cluster(b)),
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=SamePersonDecision,
            ),
        )
        d: SamePersonDecision | None = resp.parsed
        if not d:
            return False, "no decision returned"
        return d.same_person and d.confidence == "high", f"{d.confidence}: {d.reason}"
    return resolve


# ---------------------------------------------------------------- pipeline

def ingest_company(url: str, log=print) -> int:
    """Crawl site + Wikipedia + Tavily search -> Gemini extraction ->
    entity resolution -> SQLite."""
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

    people = entities.resolve_people(found, domain, name,
                                     resolver=make_llm_resolver(name, domain), log=log)
    stored = db.save_people(company_id, people)
    counts = {s: sum(p["status"] == s for p in people) for s in ("current", "unverified", "former")}
    log(f"Done. {len(found)} source records -> {stored} unique people "
        f"({counts['current']} current, {counts['unverified']} unverified, "
        f"{counts['former']} former) stored for {name}.")
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
- Each entry in the PEOPLE list is ONE unique person. Aliases are other spellings of the
  same person, and a person may have several sources and titles. When counting people,
  use the counts given in the context; never count aliases, sources, or titles as people.
- status "current": confirmed by the company's own pages or a recent source.
  "unverified": only old or undated third-party sources - say the role may be outdated.
  "former": sources say they no longer hold the role.
- "current title" is the best-supported current role (company/official sources first,
  then most recent). Titles under "other reported titles" are historical or less
  authoritative; label former roles as former and give their dates when known.
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
    people = db.get_leaders(company_id)
    history: dict[int, list[dict]] = {}
    for r in db.get_role_history(company_id):
        history.setdefault(r["leader_id"], []).append(r)
    counts = {s: sum(p["status"] == s for p in people) for s in ("current", "unverified", "former")}

    lines = [f"COMPANY: {company['name']} ({company['url']}), collected {company['scraped_at']}",
             f"\nUNIQUE PEOPLE: {len(people)} ({counts['current']} current, "
             f"{counts['unverified']} unverified, {counts['former']} former). "
             "Each bullet below is one person."]
    if not people:
        lines.append("(no leaders found)")
    for p in people:
        alias = f" (also written: {', '.join(p['aliases'])})" if p["aliases"] else ""
        as_of = f", as of {p['as_of']}" if p.get("as_of") else ""
        lines.append(f"- {p['name']}{alias} | status: {p['status']} | current title: "
                     f"{p['title']}{as_of} | {p['department']} | {p['seniority']}")
        lines.append(f"    source: {p['source_url']}")
        if p.get("evidence"):
            lines.append(f'    evidence: "{p["evidence"]}"')
        others = [r for r in history.get(p["id"], []) if not r["is_current_title"]
                  and r["title"].strip().lower() != (p["title"] or "").strip().lower()]
        for r in others[:4]:
            when = f", {r['as_of']}" if r["as_of"] else ""
            lines.append(f"    other reported title: {r['title']} ({r['status']}{when}; "
                         f"{entities.TIER_LABEL.get(r['source_tier'], '?')} source) "
                         f"{r['source_url']}")
        more = len(p["source_urls"].split("\n")) - 1
        if more > 0:
            lines.append(f"    (+{more} more supporting sources)")

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
