# Company Leadership RAG Agent

Give it a company website URL. It finds the company's executives, VPs, and department heads on public pages: the company's own site, Wikipedia, and pages found through Tavily web search. It stores them in SQLite with the URLs each one came from, and lets you ask questions about them in a Streamlit chat. Answers come from Gemini and use only the stored data.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create `.env`:

```
GEMINI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here      # optional; without it, only the site crawler and Wikipedia are used
# optional; defaults to gemini-3.5-flash, with automatic fallback
GEMINI_MODEL=gemini-3.5-flash
```

Run:

```bash
streamlit run app.py
```

Enter a URL such as `meetcampfire.com` in the sidebar, click **Collect**, then ask questions like *"Who runs engineering?"* or *"List all VPs with their sources."*

## Architecture

```
            ┌─ scraper.py: crawl company site ─┐
URL ──►     ├─ scraper.py: Wikipedia infobox  ─┼─► rag.ingest_company ──► database.py ──► rag.answer ──► app.py
            └─ search.py:  Tavily web search  ─┘   Gemini extraction      (SQLite)        retrieve +     Streamlit
                                                    + verification                         generate       chat UI
                                                    + entity resolution
                                                      (entities.py)
```

| File | Role |
|---|---|
| `scraper.py` | Normalizes the URL, reads `robots.txt`, and scores links on the homepage by leadership keywords. It also tries common paths like `/about/leadership`, follows one more level of links, and skips blog, news, and careers sections. It pulls the company's Wikipedia infobox, but only when the article links to the company's domain. |
| `search.py` | **Tavily discovery layer.** Runs 6 query templates built only from the company name and domain: leadership team; CEO/founder; C-suite; VP; head of / director; plus one search limited to the company's own site. It asks Tavily for full page text, removes duplicate URLs, and drops results that don't mention the company. This finds leaders on sites with no crawlable team page, through news coverage, press releases, the company blog, LinkedIn, and similar pages. |
| `rag.py` | **Ingest:** sends each candidate page to Gemini (in parallel) with a Pydantic response schema, then checks the results (see below) and passes them to `entities.py`. It also provides the Gemini resolver used in step 2 of entity resolution. **Answer:** builds the context from the people table (explicit unique-people counts, aliases, current title, role history) and the page chunks that best match the question by keyword overlap. Gemini must answer from that context only and cite URLs. |
| `entities.py` | Entity resolution: turns per-source records into unique people (see below). Makes no network calls; the LLM resolver is passed in. |
| `database.py` | Holds four tables. `companies`; `leaders`, one row per **unique person** (canonical name, `aliases`, current title, `status`, `as_of`, primary `source_url`, all `source_urls`, `evidence`); `role_history`, every title reported for that person by each source (status, date, source trust level, evidence); and `sources`, raw page text for retrieval. Older databases get the new columns added automatically. Collecting a company again replaces its old data. |
| `app.py` | Streamlit app. The sidebar collects a new company or picks a stored one. The main area shows the leadership table with sources and evidence, and the chat. |

## Avoiding hallucination

A record is stored only if it passes every check below. All of them are generic, and no company-specific data is hardcoded.

1. **Same company.** Gemini first names the organization the page describes and decides whether it is the company behind the given website. Pages about a different company with the same name (e.g. "Campfire Interactive" vs. campfire.ai) are thrown out.
2. **Name on the page.** The person's name must appear in the fetched page text.
3. **Verbatim evidence.** Gemini must quote the passage that shows the role, and the quote must actually be in the page text (case and punctuation are ignored).
4. **A real title.** The title must name a leadership role (CEO, VP, Head of, Director, Founder, Board…). Vague wording like "Jane in engineering" is rejected.
5. **Full name.** First-name-only or initials-only records are dropped.
6. **URLs come from code.** Source URLs are attached by the code, never written by the model.

At answer time, Gemini says plainly when the data has no answer. When sources disagree (e.g. two people listed as CTO), it reports both with their sources instead of picking one.

## Entity resolution (`entities.py`)

The same person often appears under different names across sources ("Dan" vs. "Daniel" Gallagher, "Baiju Bhat" vs. "Bhatt", "Vlad" vs. "Vladimir" Tenev). Resolution has two steps.

1. **Rule-based (exact) matching.** Names are cleaned up: accents, punctuation, honorifics (Dr.), credentials (", CFA", PhD), and middle initials are removed. Nicknames in parentheses are kept as aliases. Records with the same cleaned first and last name merge with no LLM call.
2. **Gemini check, only for unclear pairs.** A pair counts as unclear when the first names are related (same, a common nickname, or a prefix of 3+ letters, like Dan/Daniel) **and** the surnames match or are 1–2 letters apart (Bhat/Bhatt). Gemini gets both groups' titles, dates, evidence quotes, and sources, and is told that a similar name alone is not enough. The records merge only on a **high-confidence** "same person". On any other answer, or if the call fails, they stay separate. Names that are not similar (Daniel Kelati vs. Daniel Gallagher) are never sent to the check.

**Combining a merged person.**
- The canonical name is the spelling with the most trusted support. Other spellings become `aliases`. All source URLs and all reported titles are kept (`role_history`).
- **Source trust levels** are set by domain type only: (1) the company's own domain, including subdomains like `investors.`, plus sec.gov; (2) news and reference sites; (3) data brokers and social sites (LinkedIn, RocketReach, Comparably…).
- **Current title** = the most trusted non-former record, then the newest. A live (non-article) page on the company's own site counts as evidence as of the day it was fetched. If a source says a title is former and is at least as new as an older claim of the same title, that older "current" claim is demoted.
- **Status:**
  - `current`: supported by the live company site or a source from the last 2 years.
  - `unverified`: only old or undated third-party evidence, such as a 2019 press release or an undated LinkedIn snippet.
  - `former`: every source says the person no longer holds the role.

The chatbot is given explicit counts ("UNIQUE PEOPLE: 47 (33 current, 14 unverified, 0 former)") with one bullet per person. It is told to count people, never aliases, sources, or titles.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

`tests/test_entities.py` runs offline with a fake resolver. It covers:
- Dan/Daniel Gallagher and Baiju Bhat/Bhatt: flagged for review, merged only when the check confirms, aliases, sources, and role history kept.
- Similar or same-title names that are different people are never merged.
- A resolver failure leaves records separate.
- Source trust levels and current-vs-historical status.
- The chat context counts unique people.

## Limitations / next steps

- Dates come from the pages (or Tavily's `published_date`) and may be missing or wrong. A missing date makes a third-party record `unverified` rather than `current`.
- The 2-year cutoff (`STALE_AFTER_DAYS`) and the domain lists that set trust levels are judgment calls.
- When two people claim the same title (e.g. a stale CFO and the current one), status shows which claim is better supported, but there is no explicit rule across people.
- Tavily results change between runs, so collecting the same company twice may give a slightly different set of people.
- Sites that render their team page with JavaScript, or that block bots, still return little text from the crawler. Tavily often makes up for it.
- The nickname table is English-centric. A missed pair stays a duplicate rather than being merged wrongly.
