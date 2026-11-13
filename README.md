# Company Leadership RAG Agent

Give it a company website URL. It finds the company's leadership - C-level executives, Vice Presidents (incl. SVPs/EVPs), and heads of departments - in public sources: the company's own site, regulatory filings, news, reference pages, and public profiles. It checks each source's credibility before using it and stores every person in SQLite with the evidence and URLs behind them. A Streamlit chat answers questions from that data, from stored source passages, and, only when needed, from one targeted web search. Every fact is cited with a clickable source.

## Quick start

Requires **Python 3.10+**.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env              # then add your keys (Windows: copy .env.example .env)
```

`.env` holds `GEMINI_API_KEY` (required), `TAVILY_API_KEY` (optional; without it only the site crawler and Wikipedia are used, and the chat can't search the web), and `GEMINI_MODEL` (optional; defaults to `gemini-3.5-flash`, with automatic fallback and retry).

Run:

```bash
streamlit run app.py
python -m pytest tests -q          # 115 offline tests; no API keys needed
```

**Real LLMs only.** The app always calls Gemini (extraction, source assessment, entity checks, answers). Fake LLM and search functions exist only in the offline unit tests under `tests/`, so the tests run without keys or network.

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
| `search.py` | Tavily. At collection time it runs 8 query templates built from the company name and domain, phrased to surface primary sources: leadership pages, appointment announcements (incl. marketing leadership), filings, the company's own pages that name officers (e.g. blog posts), and public LinkedIn profiles. `web_search` runs one targeted search, used to corroborate weakly supported people and as the chat fallback. Tavily is only used to find pages; its results are judged like any other source. |
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
   - only these URL rules can make a source official or regulatory (the most trusted tier), never an LLM label;
   - pages that look like contact or people-data listings (email formats, phone numbers, "reveal contact", employee directories) are rejected, judged by URL and content patterns.
2. **Gemini assessment, only for unclear pages, in batches.** Each page is labeled *news*, *reference*, *professional_profile* (e.g. a public LinkedIn page readable without login), *people_data_aggregator*, or *other*, together with whether it is **about this company** (not a namesake such as "Campfire Interactive") and whether it is credible. Only credible news, reference, and profile pages about the company are accepted. A people-data aggregator is never evidence, even if marked credible: such claims count only when an authoritative source shows the same thing, and that source then stands on its own.
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

## Who counts as leadership

The level is decided **from the title, in code** (`entities.leadership_level`), not by the LLM, so counts are deterministic:
- **C-level**: Chief … Officer, CEO/CFO/CTO/…, or President (incl. presidents of subsidiaries).
- **VP**: Vice President, SVP, EVP. A plain "Vice President and Chief X Officer" has VP rank.
- **Department head**: "Head of …"; or a Director / Senior Director / General Manager only when the title or page shows they lead a department or function (e.g. "Senior Director, Communications" on the company's leadership page).
- **Not leadership**: board-only members, founder-only titles, Chief of Staff, and directors' jobs at other organizations listed in proxy statements. These people are excluded.

## Current vs. outdated roles

The database stores no status. Freshness is computed from evidence dates when the data is read.
- **Evidence date = when the source said it**: the page's publication or filing date, not a role's start date ("director since 2016"). Future dates are ignored.
- **Current evidence**: a live company page (as of the day it was fetched), the person's own public profile that lists the role as current (e.g. LinkedIn "to present", even without a date), or a source dated within the last 2 years (`STALE_AFTER_DAYS`). Anything else is shown "(may be outdated)".
- **Outdated roles are not stored.** A role is dropped when a credible source says it ended ("former", "stepped down", an 8-K "separating from his position") and is at least as new as the claim; "Chief Technology Officer" and "CTO" count as the same role. The departure evidence itself is kept as a hidden row (`role_history.ended`), so later re-checks still apply it; it is never shown as a role.
- **Current roster.** If the company has a live roster page (3+ people listed on a non-article company page), people who are on no live company page and have no current evidence are excluded - e.g. officers from an old proxy or a 2020 appointment announcement.
- **One-person roles.** If different people claim the same company-wide role (CEO, CFO, CTO, COO, CMO, CIO, CISO, CLO, General Counsel), the newer dated claim wins and the older one is dropped (the person is excluded if no other leadership role remains). If it can't be decided, both are flagged.
- **Disagreeing sources are flagged, not hidden**: e.g. a profile saying "Head of Solution Consulting" and "Founding Solution Consultant". The table shows "(sources disagree)" and the chat says so.
- **The current title** comes from the most authoritative source (official/filing first), then the newest.

## Corroboration

After extraction, people backed by a single non-official source - and people excluded only for lack of current evidence whose sources are all non-official - get one targeted search (up to 10 per collection). New pages must mention the person and the company and pass the same source policy; entity resolution then runs again on all records.

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

`rag.reextract_company(id)` re-runs extraction over the stored source passages (Gemini only, no crawling or search) and resolves the results with the stored records - used when evidence the stored records lack must be recovered after a rule change.

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

`companies` (incl. `resolved_version`, `removed_unverified`, `collection_report`), `leaders` (one row per person; `seniority` is the title-based level, `conflict` an unresolved source disagreement), `role_history` (supporting evidence per source: title, `as_of`, source type, quote), `sources` (accepted sources as passages, with `source_type`, `published`, `origin`), and `search_cache`.

Migrations run automatically, are safe to rerun, and copy the database to `backups/` before any destructive step. The `status` columns were removed this way, along with the former-role rows they marked.

## Limitations / next steps

- **Collections vary between runs.** Tavily results and Gemini judgments change, so collecting twice can give slightly different people. Gemini overloads (503) are retried, but extraction can still fail for a page.
- **Roster detection needs a roster page.** For companies without a live leadership page (e.g. small startups), people with only old evidence are kept and flagged "(may be outdated)" rather than excluded.
- **Self-reported profiles.** A current LinkedIn position counts as current evidence; it is corroborated when another source can be found, and contradictions are flagged, but a profile that is simply stale can't be detected.
- **Discovery is bounded.** 8 company-level searches plus up to 10 corroboration searches can still miss people who appear only on individual profiles.
- **Dates are best-effort.** They come from the page or Tavily, and a missing date leads to "may be outdated".
- **The source classifier is an LLM judgment.** A wrong rejection loses evidence; it never invents a record.
- **The crawler can't render JavaScript.** Sites that build their team page with JavaScript, or that block bots, yield little crawler text; Tavily often makes up for it.
- **The nickname table is English-centric.** A missed pair stays a duplicate rather than being merged wrongly.
