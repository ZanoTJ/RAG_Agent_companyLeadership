"""User-facing leadership views: a compact summary table plus per-person details.
Internal fields (person IDs, source tiers) are used for logic but never shown;
"As Of" is the date of the evidence for the title.

Every value comes from untrusted web pages, so it is HTML-escaped before it is put
into markup, and only http(s) URLs become links."""

import html
import re

SENIORITY_DISPLAY = {"executive": "C-level", "vp": "VP", "department_head": "Department head"}
# (column, share of the table width): Title gets the most room.
SUMMARY_COLUMNS = [("Name", 19), ("Title", 34), ("Department", 16), ("Seniority", 14),
                   ("Location", 17)]
# Table cells break a word only if it can't fit on a line at all; details (long URLs)
# may break anywhere.
_WRAP_CELL = "white-space:normal;overflow-wrap:break-word;word-break:normal;"
_WRAP = "white-space:normal;overflow-wrap:anywhere;word-break:normal;"


def _esc(value: str) -> str:
    # "$" is encoded too, so Streamlit's markdown never reads it as LaTeX math.
    return html.escape(value or "").replace("$", "&#36;")


def _link(url: str) -> str:
    if not re.match(r"https?://", url or "", re.I):
        return _esc(url)
    return (f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer">'
            f"{_esc(url)}</a>")


def summary_rows(leaders: list[dict]) -> list[dict]:
    """One row per person (already ordered by seniority, then name, by the DB)."""
    return [{
        "Name": f'{p["name"]} ({", ".join(p["aliases"])})' if p.get("aliases") else p["name"],
        "Title": (p.get("title") or "") + (" (sources disagree)" if p.get("conflict") else ""),
        "Department": p.get("department") or "",
        "Seniority": SENIORITY_DISPLAY.get(p.get("seniority"), "Other"),
        "Location": p.get("location") or "",
    } for p in leaders]


def summary_html(leaders: list[dict]) -> str:
    """The summary rows as a full-width table whose cells wrap instead of truncating."""
    cell = f'style="{_WRAP_CELL}text-align:left;vertical-align:top;padding:4px 6px;"'
    cols = "".join(f'<col style="width:{w}%">' for _, w in SUMMARY_COLUMNS)
    head = "".join(f"<th {cell}>{name}</th>" for name, _ in SUMMARY_COLUMNS)
    body = "".join("<tr>" + "".join(f"<td {cell}>{_esc(r[name])}</td>"
                                    for name, _ in SUMMARY_COLUMNS) + "</tr>"
                   for r in summary_rows(leaders))
    return (f'<table style="width:100%;table-layout:fixed;border-collapse:collapse;'
            f'font-size:0.9rem;">'
            f"<colgroup>{cols}</colgroup><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>")


def person_label(p: dict) -> str:
    """'Name — Title' for an expander label, with markdown characters escaped."""
    label = f'{p["name"]} — {p.get("title") or ""}'.rstrip(" —")
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~<>$:])", r"\\\1", label)


def person_details(p: dict) -> list[tuple[str, str | list[str]]]:
    """(label, value) pairs for one person; empty fields are left out."""
    more = [u for u in (p.get("source_urls") or "").split("\n")
            if u and u != p.get("source_url")]
    fields = [("Bio", p.get("bio")), ("Profile", p.get("profile_url")),
              ("As Of", p.get("as_of")), ("Source", p.get("source_url")),
              ("More sources", more)]
    return [(k, v) for k, v in fields if v]


def details_html(p: dict) -> str:
    """person_details as wrapped HTML lines; URLs are clickable and shown in full."""
    lines = []
    for label, value in person_details(p):
        if isinstance(value, list):
            items = "".join(f"<li>{_link(u)}</li>" for u in value)
            lines.append(f"<div><b>{label}:</b><ul style='margin:0'>{items}</ul></div>")
        else:
            text = _link(value) if label in ("Profile", "Source") else _esc(value)
            lines.append(f"<div><b>{label}:</b> {text}</div>")
    return f'<div style="{_WRAP}">{"".join(lines)}</div>'


def disagreements(leaders: list[dict]) -> list[str]:
    return [f"{p['name']}: {p['conflict']}" for p in leaders if p.get("conflict")]
