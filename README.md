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
                                                    + deduplication
```

| File | Role |
|---|---|
| `scraper.py` | Normalizes the URL, reads `robots.txt`, and scores links on the homepage by leadership keywords. It also tries common paths like `/about/leadership`, follows one more level of links, and skips blog, news, and careers sections. It pulls the company's Wikipedia infobox, but only when the article links to the company's domain. |
| `search.py` | **Tavily discovery layer.** Runs 6 query templates built only from the company name and domain: leadership team; CEO/founder; C-suite; VP; head of / director; plus one search limited to the company's own site. It asks Tavily for full page text, removes duplicate URLs, and drops results that don't mention the company. This finds leaders on sites with no crawlable team page, through news coverage, press releases, the company blog, LinkedIn, and similar pages. |
| `rag.py` | **Ingest:** sends each candidate page to Gemini (in parallel) with a Pydantic response schema, then checks the results (see below) and removes duplicates. **Answer:** builds the context from the full leadership table (all sources plus evidence quotes) and the page chunks that best match the question by keyword overlap. Gemini must answer from that context only and cite URLs. |
| `database.py` | Holds three tables. `companies`; `leaders` (name, title, department, seniority, primary `source_url`, all `source_urls`, and a verbatim `evidence` quote); and `sources` (raw page text for retrieval). Older databases get the new columns added automatically. Collecting a company again replaces its old data. |
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

## Deduplication

Records are grouped by a normalized name key: lowercase, with punctuation, nicknames in parentheses, honorifics (Dr.), suffixes (PhD, Jr.), and middle initials removed. The key uses the first and last name. For each person, the title and department come from the company's own website when possible, otherwise from the most senior and most specific record. Every supporting URL is kept in `source_urls`. Phrases like "CTO at Acme" are cleaned to "CTO".

## Limitations / next steps

- Third-party sources (data aggregators, old blog posts) can be out of date. The chat flags conflicts but can't tell which source is current. Weighting by date or by type of source would help.
- Tavily results change between runs, so collecting the same company twice may give a slightly different set of people.
- Sites that render their team page with JavaScript, or that block bots, still return little text from the crawler. Tavily often makes up for it.
- Name matching uses first and last name only, so "Bob" vs. "Robert" are not merged.
