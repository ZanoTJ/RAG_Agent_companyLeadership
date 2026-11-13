"""User-facing table rows. Internal fields (person IDs, dates, source tiers) are
used for logic but never shown."""

import json
from collections import Counter

import source_policy

SENIORITY_DISPLAY = {"executive": "C-level", "vp": "VP", "department_head": "Department head"}
PEOPLE_COLUMNS = ["Name", "Also known as", "Title", "Department", "Seniority", "Source",
                  "More sources"]
EVIDENCE_COLUMNS = ["Name", "Title", "Source type", "Source", "Evidence"]


def people_rows(leaders: list[dict]) -> list[dict]:
    rows = []
    for p in leaders:
        urls = [u for u in (p.get("source_urls") or p["source_url"]).split("\n") if u]
        rows.append({
            "Name": p["name"],
            "Also known as": ", ".join(p.get("aliases") or []),
            "Title": (p.get("title") or "")
            + (" (may be outdated)" if p.get("may_be_outdated") else "")
            + (" (sources disagree)" if p.get("conflict") else ""),
            "Department": p.get("department") or "",
            "Seniority": SENIORITY_DISPLAY.get(p.get("seniority"), "Other"),
            "Source": p["source_url"],
            "More sources": [u for u in urls if u != p["source_url"]],
        })
    return rows


def disagreements(leaders: list[dict]) -> list[str]:
    return [f"{p['name']}: {p['conflict']}" for p in leaders if p.get("conflict")]