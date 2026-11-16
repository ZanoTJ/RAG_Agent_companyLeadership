"""Dump each collected company's people and their supporting evidence to
fixtures/<domain>.json.  Usage: python scripts/export_fixtures.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import database as db  # noqa: E402
from scraper import bare_domain  # noqa: E402

PERSON_FIELDS = ("name", "aliases", "title", "seniority", "department", "location",
                 "profile_url", "bio", "as_of", "conflict", "source_url")
EVIDENCE_FIELDS = ("name_in_source", "title", "as_of", "source_url", "source_type", "evidence")


def main() -> None:
    out_dir = ROOT / "fixtures"
    out_dir.mkdir(exist_ok=True)
    for c in db.list_companies():
        evidence: dict[int, list[dict]] = {}
        for r in db.get_role_history(c["id"]):
            evidence.setdefault(r["person_id"], []).append({k: r[k] for k in EVIDENCE_FIELDS})
        people = [{**{k: p[k] for k in PERSON_FIELDS},
                   "source_urls": p["source_urls"].split("\n"),
                   "evidence": evidence.get(p["person_id"], [])}
                  for p in db.get_leaders(c["id"])]
        path = out_dir / f"{bare_domain(c['url'])}.json"
        path.write_text(json.dumps({"company": c["name"], "url": c["url"],
                                    "collected_at": c["scraped_at"], "people": people},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {len(people)} people")


if __name__ == "__main__":
    main()
