# Company Leadership RAG Agent

Give it a company website URL. It finds the company's executives, VPs, and department heads in public sources: the company's own site, regulatory filings, news, reference pages, and public profiles. It checks each source's credibility before using it and stores every person in SQLite with the evidence and URLs behind them. A Streamlit chat answers questions from that data, from stored source passages, and, only when needed, from one targeted web search. Every fact is cited with a clickable source.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create `.env`:

```
GEMINI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here      # optional; without it, only the site crawler and Wikipedia are used,
                                  # and the chat can't search the web
# optional; defaults to gemini-3.5-flash, with automatic fallback and retry
GEMINI_MODEL=gemini-3.5-flash
```

Run:

```bash
streamlit run app.py
python -m pytest tests -q          # 72 offline tests; no API keys needed
```

Enter a URL such as `meetcampfire.com` in the sidebar and click **Collect**. Then ask questions such as *"Who is the CEO?"*, *"How many VPs are listed?"*, or *"Where is their CEO based?"*

## Architecture

```
            ┌─ scraper.py: crawl company site ─┐      source_policy.py       rag.py             entities.py         database.py
URL ──►     ├─ scraper.py: Wikipedia          ─┼──►  rules, then Gemini  ──► Gemini        ──► entity          ──► SQLite
            └─ search.py:  Tavily search      ─┘     (accept / reject)       extraction        resolution

Question ──► rag.answer: SQLite people + exact counts ─► stored passages about the person ─► (only if needed) one
             targeted Tavily search, same source policy, cached ─► Gemini answer ─► links checked ─► app.py
```

| File | Role |
|---|---|
| `scraper.py` | Crawls the company site (`robots.txt`, keyword-scored links, common leadership paths). Redirects to other sites are dropped. Also fetches Wikipedia when the article links to the company's domain. |
| `search.py` | Tavily. At collection time it runs 6 query templates built from the company name and domain, phrased to surface primary sources (leadership pages, appointment announcements, filings). At answer time it runs one targeted search (`web_search`). Tavily is only used to find pages; its results are judged like any other source. |
| `source_policy.py` | Decides whether a page may be used as evidence (see below), and keeps only the relevant passages of accepted pages. |
| `rag.py` | Collection pipeline, Gemini extraction, source classifier, entity resolver, clean-up of stored data, and hybrid answering. |
| `entities.py` | Entity resolution and title freshness (see below). Makes no network calls; the LLM functions are passed in. |
| `views.py` | The user-facing table rows. Internal fields stay hidden. |
| `database.py` | SQLite schema, safe migrations, and queries. |
| `app.py` | Streamlit UI: collect or pick a company, leadership table, supporting evidence, and the chat. |

## Choosing sources (`source_policy.py`)

There is no domain blocklist. Every page, including the final URL after a redirect, is evaluated **before extraction and storage**.

1. **Rule-based checks (no API calls):**
   - pages with no readable content, or behind a login wall, are rejected;
   - the company's own domain (any subdomain) is accepted as *official*;
   - `.gov` domains (e.g. SEC filings) are accepted as *regulatory*;
   - pages that look like contact or people-data listings (email formats, phone numbers, "reveal contact", employee directories) are rejected, judged by URL and content patterns.
2. **Gemini assessment, only for unclear pages, in batches.** Each page is labeled *news*, *reference*, *professional_profile* (e.g. a public LinkedIn page readable without login), *people_data_aggregator*, or *other*, together with whether it is credible. Only credible news, reference, and profile pages are accepted. A people-data aggregator is never evidence, even if marked credible: such claims count only when an authoritative source shows the same thing, and that source then stands on its own.
3. **If the assessment fails, unclear pages are rejected.** Unverified sources are never used.

## Avoiding hallucination

A leadership record is stored only if **all** of these hold:
- it comes from an accepted source;
- the page is about the right company (pages about another company with the same name are rejected);
- the person is the company's own employee or board member, not a customer quoted in a testimonial, an investor, or a partner;
- the name and a verbatim evidence quote appear in the page text;
- the title names a real leadership role;
- the record has a full name.

Source URLs are attached by the code, never written by the model. The app never claims the list is complete. Counts are worded "listed in the collected sources".

## Current vs. outdated roles

The database stores no status. Freshness is computed from evidence dates when the data is read.
- **Outdated roles are not stored.** If a credible source says a person no longer holds a role ("former", "stepped down") and that source is at least as new as the claim, the role is dropped. People with only former roles are excluded.
- **The current title** comes from the most authoritative source (official/filing first), then the newest.
- **"(may be outdated)"** is added to a title unless it is backed by a live company page (counted as of the day it was fetched) or a source dated within the last 2 years (`STALE_AFTER_DAYS`). Old press releases and undated third-party profiles are therefore flagged.

## Entity resolution (`entities.py`)

1. **Exact matching after cleanup.** Accents, punctuation, honorifics, credentials, and middle initials are removed, and records with the same cleaned name merge.
2. **Gemini check for unclear pairs only.** Surnames must be similar **and** either the first names are related (nickname, or a prefix such as Dan/Daniel, Vlad/Vladimir) or the titles share a specific role word (Jen-Hsun/Jensen Huang, both "founder … CEO"). The pair merges only on a high-confidence "same person" based on the evidence. A shared surname or title alone never merges two people.
3. **Display name and IDs.** The spelling used on the company's own site becomes the display name, and other spellings become aliases. `leaders.id` is the internal canonical person ID, used by the database, counts, and chat citations. It is never displayed.

**Re-checking stored data.** When this logic changes (`RESOLUTION_VERSION`), the app re-checks stored companies once on load. The sidebar button **Re-check stored data** does the same on demand.
- Stored sources are evaluated, and evidence and sources that fail are removed.
- People left with no acceptable evidence are removed. Their number is shown, with a suggestion to collect the company again.
- Entity resolution runs again, and merged people keep their lowest ID.
- Pages are trimmed to relevant passages.
- If the source check fails, nothing changes and it is retried later.

## Hybrid answering (`rag.answer`)

1. **SQLite.** One line per unique person, plus counts by seniority computed in Python, so "How many VPs?" is deterministic.
2. **Who is the question about?** This is resolved in Python: names and aliases first (Jen-Hsun Huang finds Jensen Huang), otherwise role words matched against titles ("CEO", "marketing").
3. **Stored evidence.** Passages around that person's names in the accepted sources.
4. **One Gemini call** returns `{answer, sufficient, missing_info}`. Most questions end here, with no search.
5. **Web fallback, only if the answer is insufficient:**
   - one Tavily search, `"<canonical name>" <company> <missing detail>`;
   - results must mention the person and pass the same source policy;
   - accepted passages are cached in `sources` (`origin=web`), and the query in `search_cache` (7 days);
   - a second Gemini call answers from the new evidence.

   A repeated question reuses the cached passages. If nothing reliable turns up, the answer says so.
6. **Safety.**
   - Web text is passed in delimited blocks marked as untrusted data.
   - Location answers give only a person's publicly reported professional base, distinct from company headquarters and never a home address.
   - After the answer is generated, any link not present in the evidence is removed, along with internal IDs.
   - If search is down, the answer uses the collected data and says the details couldn't be checked.

## Database

`companies` (incl. `resolved_version`, `removed_unverified`), `leaders` (one row per person), `role_history` (supporting evidence per source: title, `as_of`, source type, quote), `sources` (accepted sources as passages, with `source_type`, `published`, `origin`), and `search_cache`.

Migrations run automatically, are safe to rerun, and copy the database to `backups/` before any destructive step. The `status` columns were removed this way, along with the former-role rows they marked.

## Limitations / next steps

- **Collections vary between runs.** Tavily results and Gemini judgments change, so collecting twice can give slightly different people. Gemini overloads (503) are retried, but extraction can still fail for a page.
- **Being dropped from a roster isn't detected.** If a newer filing simply omits a former director, there is no explicit "former" statement, so the old record stays and is flagged "(may be outdated)".
- **Dates are best-effort.** They come from the page or Tavily, and a missing date leads to "may be outdated".
- **The source classifier is an LLM judgment.** A wrong rejection loses evidence; it never invents a record.
- **The crawler can't render JavaScript.** Sites that build their team page with JavaScript, or that block bots, yield little crawler text; Tavily often makes up for it.
- **The nickname table is English-centric.** A missed pair stays a duplicate rather than being merged wrongly.
