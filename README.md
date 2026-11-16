# Company Leadership RAG Agent

Give it a company website. It finds the company's leadership (C-level executives,
VPs incl. SVPs/EVPs, and heads of departments) in public sources, stores each unique
person in SQLite with the evidence behind them, and answers questions about them in a
Streamlit chat. Every fact is cited with a link to its source.

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

Enter a URL (e.g. `meetcampfire.com`) in the sidebar and click **Collect**, then ask
e.g. *"Who's their CTO?"*, *"How many VPs do they have?"*, *"Who heads marketing?"*,
*"Where is their CEO based?"*. The **Collection report** expander shows every source
accepted or rejected and every person excluded, each with the reason.

The app always calls the real Gemini API; fakes exist only in `tests/`.

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
   the title names a leadership role, and the person works for the company (not a customer
   quoted in a testimonial). Location must appear in the page text and a profile URL in the
   page text or URL; otherwise they are left empty. Source URLs are attached by code.
4. **Entity resolution** (`entities.py`, no network). Names are normalized and exact matches
   merged. Similar names (nicknames, one-letter variants, transliterations) are merged only if
   a Gemini check on the evidence says "same person" with high confidence; without a resolver
   they stay separate. Seniority comes from the title in code, so counts are deterministic.
   Roles that a source says have ended, or that a newer source gives to someone else (one-person
   roles like CEO/CTO), are dropped. The company's own spelling becomes the display name.
5. **SQLite.** A collection runs entirely in memory, then replaces the company's previous data
   in one transaction. If every extraction fails (e.g. Gemini is down), the previous data is kept.
6. **Hybrid answering** (`rag.answer_with_trace`). The prompt gets one line per person, counts
   computed in Python, and stored passages about the person the question is about. Only if
   Gemini says the context is insufficient does it run one Tavily search, filtered by the same
   source policy and cached for 7 days. Any link not present in the context is stripped.

## Design trade-offs

- **SQLite + passages, no vector store.** The data is small and structured; questions are
  mostly "who holds role X" or "how many", which SQL and exact counts answer better than
  similarity search. Passages around each person's name cover the rest.
- **Precision over recall.** Unverified sources are dropped, similar names aren't merged on a
  guess, and the app says "listed in the collected sources" rather than claiming completeness.
- **Freshness from dates, not flags.** A person missing from a company roster page is excluded
  only if all their evidence is non-official and undated or older than 2 years
  (`STALE_AFTER_DAYS`). Weakly supported people get one corroboration search that can supply a
  recent source.
- **Bounded cost.** A fixed number of collection searches, up to 10 corroboration searches, and
  at most one web search per question.

## Known limitations

- **Board members and founder-only titles are excluded by design**: only operating leadership
  roles count (a "Co-Founder and CTO" is kept; a "Co-Founder" or "Independent Director" is not).
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
| `app.py` | Streamlit UI |
| `rag.py` | Collection pipeline, Gemini calls, hybrid answering |
| `entities.py` | Entity resolution and leadership rules |
| `source_policy.py` | Source acceptance rules |
| `scraper.py`, `search.py` | Company-site crawler and Wikipedia, Tavily search |
| `database.py`, `views.py` | SQLite schema and queries, table rows for the UI |
| `tests/` | Offline pytest suite |
| `scripts/export_fixtures.py` | Writes each collected company's people and evidence to `fixtures/<domain>.json` |
| `fixtures/` | Exported results from real collections (e.g. `campfire.ai.json`, `robinhood.com.json`) |
| `sessions/` | Exported Claude Code session transcripts from building this project |
