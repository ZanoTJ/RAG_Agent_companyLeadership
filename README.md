# Company Leadership RAG Agent

Give it a company website. It finds the company's leadership (C-level executives,
VPs incl. SVPs/EVPs, and heads of departments) in public sources, stores each unique
person in SQLite with the evidence behind them, and answers questions about them in a
Streamlit chat. Every fact is cited with a link to its source, and every person carries
the date of their newest evidence so stale information is flagged rather than presented
as current.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```

Fill in `.env`:

| Variable | |
|---|---|
| `GEMINI_API_KEY` | Required. Extraction, source assessment, entity checks, answers. |
| `TAVILY_API_KEY` | Optional. Without it only the company site and Wikipedia are used, and the chat can't search the web. |
| `GEMINI_MODEL` | Optional, default `gemini-3.5-flash`. A mistyped name fails immediately; overloads (429/5xx) fall back to other models and retry for up to ~2 minutes. |

## Run

```bash
streamlit run app.py
python -m pytest -q               # offline: LLM and search are faked, no keys needed
```

Enter a URL (e.g. `meetcampfire.com`) in the sidebar and click **Collect**. The page then
shows:

- **Leadership**: a summary table (name, title, department, seniority, and a
  **May be Outdated** flag).
- **Person Details**: one expandable entry per person with bio, profile link, *As Of*
  date, location, and every supporting source.
- **Chat**: ask e.g. *"Who's their CTO?"*, *"How many VPs do they have?"*, *"Who heads
  marketing?"*, *"Where is their CEO based?"*, *"Which of these might be outdated?"*.

Collection progress, including every source and person that was skipped and why, is
logged to the terminal.

The app always calls the real Gemini API; fakes exist only in `tests/`.

## Data collection rules

The agent is deliberately conservative about what it collects and what it treats as fact.

**Only public, trustworthy sources**
- Accepted: the company's own website (*official*), government filings such as `sec.gov`
  (*regulatory*), news, reference works such as Wikipedia, and public professional profiles.
- Rejected: pages behind a login wall, pages with no readable content, job postings on
  third-party sites, and people-data / contact-data aggregators (email formats, phone
  numbers, "reveal contact" pages, org-chart sites).
- Non-official pages are checked by Gemini for credibility and for being about *this*
  company rather than a namesake. If that check can't run, the page is rejected:
  unverified sources are never used.
- Only the company's own domain (or `.gov`) can make a source "official"; the LLM can't.
- `robots.txt` is respected, and Wikipedia is used only if the article links to the
  company's domain.

**Only the company itself, not its subsidiaries or other employers**
- A role counts only if it is held at the company or one of its own "<Company> X" entities
  (e.g. "Robinhood UK", "Robinhood Financial LLC").
- Roles at separately branded subsidiaries or acquired companies (e.g. "Sherwood Media",
  "TradePMR") are dropped, as are outside jobs named in a source (e.g. a board member's
  "CMO, Autodesk, Inc." in a proxy filing). A combined title spanning two organizations is
  split, and only the company's part is kept.
- Customers, partners, and investors quoted on the company's site are not leaders.

**Only operating leadership**
- C-level executives, VPs (incl. SVPs/EVPs), and heads of departments. Board members and
  founder-only titles are excluded ("Co-Founder and CTO" is kept, "Co-Founder" is not).
- Names must be a specific person's name, not a role, first name, or initials.

**Only facts grounded in the page**
- The name and a verbatim evidence quote must appear in the page text. Titles are copied
  as written, never inferred.
- Location (professional base only, never a home address) and profile URL are kept only if
  they appear on the page; otherwise they are left empty.
- No personal contact data (emails, phone numbers) is collected.

**Dated, with stale information flagged**
- Each person's *As Of* is the date of their newest supporting source (when the role was
  last confirmed, not when it started).
- People whose newest evidence is more than a year old are marked **outdated**
  (`OUTDATED_AFTER_DAYS` in `database.py`). The flag is recomputed against today's date
  each time the data is read; undated evidence is never flagged on a guess.
- Roles that a source says have ended, or that a newer source gives to someone else, are
  dropped. People supported only by non-official evidence older than 2 years
  (`STALE_AFTER_DAYS`) are excluded.

## Architecture

```
collect ──► source policy ──► grounded extraction ──► entity resolution ──► SQLite
(scraper.py,  (source_policy.py)  (rag.py, Gemini)       (entities.py)        (database.py)
 search.py)

question ──► SQLite people + exact counts ──► stored passages ──► Gemini answer
             ──► (only if insufficient) ONE targeted web search, same source policy,
                 cached ──► second answer ──► links checked against the evidence
```

1. **Collect.** `scraper.py` crawls the company site (robots.txt respected, leadership-like
   links scored, common paths tried) and fetches the Wikipedia article if it links to the
   company's domain. `search.py` runs a fixed set of Tavily queries built only from the
   company name and domain.
2. **Source policy** (`source_policy.py`), before anything is extracted or stored. URL rules
   decide what is *official* (the company's domain, both the entered one and the one it
   redirects to) or *regulatory* (`.gov`); login walls and people-data/contact listings are
   rejected by rule. Everything else is labelled by Gemini (news / reference / public profile
   / aggregator, and whether it's about this company rather than a namesake). If that check
   fails, the page is rejected.
3. **Grounded extraction** (`rag.extract_leaders`). Gemini returns structured records. A
   record is kept only if the name *and* a verbatim evidence quote appear in the page text,
   the title names a leadership role, the person works for the company (not a customer
   quoted in a testimonial), and the role is held at the company rather than a subsidiary
   or another employer (Gemini returns the organization for each title). Location must appear in the page text and a profile URL in the
   page text or URL; otherwise they are left empty. Source URLs are attached by code.
4. **Entity resolution** (`entities.py`, no network). Names are normalized and exact matches
   merged. Similar names (nicknames, one-letter variants, transliterations) are merged only if
   a Gemini check on the evidence says "same person" with high confidence; without a resolver
   they stay separate. Titles naming an outside organization with a legal suffix
   ("..., Autodesk, Inc.") are dropped as a second, code-level check. Seniority comes from
   the title in code, so counts are deterministic.
   Roles that a source says have ended, or that a newer source gives to someone else (one-person
   roles like CEO/CTO), are dropped. The company's own spelling becomes the display name.
5. **SQLite.** A collection runs entirely in memory, then replaces the company's previous data
   in one transaction. If every extraction fails (e.g. Gemini is down), the previous data is kept.
   Each person row stores an `as_of` date and an `outdated` flag; older databases get the
   column added in place on startup.
6. **Hybrid answering** (`rag.answer_with_trace`). The prompt gets one line per person, counts
   computed in Python (including how many people are outdated or undated), each person's
   *evidence as of* date and OUTDATED marker, and stored passages about the person the question is about. Only if
   Gemini says the context is insufficient does it run one Tavily search, filtered by the same
   source policy and cached for 7 days. Any link not present in the context is stripped.
   Gemini is told what *as of* and OUTDATED mean, so it can answer questions about how
   current the data is.

## Design trade-offs

- **SQLite + passages, no vector store.** The data is small and structured; questions are
  mostly "who holds role X" or "how many", which SQL and exact counts answer better than
  similarity search. Passages around each person's name cover the rest.
- **Precision over recall.** Unverified sources are dropped, similar names aren't merged on a
  guess, and the app says "listed in the collected sources" rather than claiming completeness.
- **Freshness from dates.** A person missing from a company roster page is excluded
  only if all their evidence is non-official and undated or older than 2 years
  (`STALE_AFTER_DAYS`). Weakly supported people get one corroboration search that can supply a
  recent source. Kept people whose newest evidence is over a year old are shown with a
  *May be Outdated* flag instead of being silently dropped.
- **Bounded cost.** A fixed number of collection searches, up to 10 corroboration searches, and
  at most one web search per question.

## Known limitations

- **Board members and founder-only titles are excluded by design**: only operating leadership
  roles count (a "Co-Founder and CTO" is kept; a "Co-Founder" or "Independent Director" is not).
- **Subsidiaries are excluded by design**: leaders of separately branded subsidiaries aren't
  collected. Telling a subsidiary from a company entity relies on the name ("<Company> X"),
  so an entity with an unrelated name is treated as a subsidiary.
- **JavaScript-rendered sites**: the crawler doesn't run JS, so team pages built client-side
  yield little text. Tavily often makes up for it.
- **Old official sources keep people.** Any company-domain or `.gov` source keeps a person, so
  an officer named in an old press release or proxy can remain listed after leaving.
- **Runs vary.** Tavily results and Gemini judgments change between runs.
- **Dates are best effort**, taken from the page or from Tavily.
- **The nickname table is English-centric.** A missed pair stays a duplicate, never a wrong merge.

## Repository contents

| Path | |
|---|---|
| `app.py` | Streamlit UI (leadership table, person details, chat) |
| `rag.py` | Collection pipeline, Gemini calls, hybrid answering |
| `entities.py` | Entity resolution, leadership rules, company-vs-subsidiary check, date handling |
| `source_policy.py` | Source acceptance rules |
| `scraper.py`, `search.py` | Company-site crawler and Wikipedia, Tavily search |
| `database.py`, `views.py` | SQLite schema and queries (incl. the `outdated` flag), HTML tables for the UI |
| `tests/` | Offline pytest suite |