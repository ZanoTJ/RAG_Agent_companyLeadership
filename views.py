"""User-facing table rows. Internal fields (person IDs, dates, source tiers) are
used for logic but never shown."""

import source_policy

SENIORITY_DISPLAY = {"executive": "Executive", "vp": "VP", "department_head": "Department head",
                     "board": "Board", "other": "Other"}
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
            "Title": (p.get("title") or "") + (" (may be outdated)" if p.get("may_be_outdated")
                                               else ""),
            "Department": p.get("department") or "",
            "Seniority": SENIORITY_DISPLAY.get(p.get("seniority"), "Other"),
            "Source": p["source_url"],
            "More sources": [u for u in urls if u != p["source_url"]],
        })
    return rows


def evidence_rows(roles: list[dict]) -> list[dict]:
    return [{
        "Name": r["name"],
        "Title": r["title"],
        "Source type": source_policy.SOURCE_LABEL.get(r.get("source_type") or "",
                                                      "Official / filing" if r.get("source_tier") == 1
                                                      else "Public source"),
        "Source": r["source_url"],
        "Evidence": r.get("evidence") or "",
    } for r in roles]
