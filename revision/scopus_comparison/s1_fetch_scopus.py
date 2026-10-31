#!/usr/bin/env python3
"""
s1_fetch_scopus.py  —  Scopus raw-affiliation mini-test, step 1 (fetch)

Purpose (Reviewer 1, comment 5): test whether the geocoding-failure mechanism
is specific to OpenAlex or a property of raw affiliation strings in general,
and show that Scopus users need not geocode raw strings at all because Scopus
supplies structured city/country fields.

Design: paired comparison on the SAME works.
  1. Draw a seeded random sample of works with DOIs from the silicon-carbide
     topic (OpenAlex T10361, 2018-2024) via the OpenAlex `sample` parameter.
  2. For each DOI, call the Scopus Abstract Retrieval API (default view; this
     key's service level does not expose view=FULL / author-group / ce:source-text)
     and keep the work-level structured affiliation list: affilname,
     affiliation-city, affiliation-country, afid.
  3. Keep the OpenAlex raw_affiliation_strings for the same works from the
     OpenAlex response, so both databases' raw strings refer to identical papers.

Outputs (all LOCAL ONLY — Scopus records may not be redistributed):
  scopus_test/cache/abstract_full.sqlite   raw API responses, resumable
  scopus_test/work_sample.csv              openalex_id, doi, year, scopus status
  scopus_test/scopus_affils.csv            doi, idx, affilname, city, country, afid
  scopus_test/openalex_affils.csv          doi, openalex_id, raw_affiliation_string, institution_ids, country_codes
  scopus_test/fetch_log.json               counts, quota headers, probe result

Credentials: read from env SCOPUS_API_KEY / SCOPUS_INSTTOKEN, or from
  a local scopus_api_key.txt (and scopus_insttoken.txt if present).
  The key is never printed or written anywhere by this script.

Run on the campus network:
  python s1_fetch_scopus.py --n 2000 --seed 20260811
  python s1_fetch_scopus.py --probe          # only test 3 DOIs and show which fields come back
"""
import argparse
import csv
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
KEY_ROOT = Path(os.environ.get("SCOPUS_KEY_DIR", "."))

OPENALEX = "https://api.openalex.org/works"
SCOPUS_ABS = "https://api.elsevier.com/content/abstract/doi"
MAILTO = "dddsss5419@gmail.com"
TOPIC = "T10361"
SLEEP_SCOPUS = 0.35      # ~3 req/s, well under the 9 req/s ceiling
UA = {"User-Agent": f"geocoding-failure-openalex/scopus-test (mailto:{MAILTO})"}


# ---------------------------------------------------------------- credentials
def get_key() -> str:
    k = os.environ.get("SCOPUS_API_KEY", "").strip()
    if k:
        return k
    f = KEY_ROOT / "scopus_api_key.txt"
    if f.exists() and f.read_text(encoding="utf-8").strip():
        return f.read_text(encoding="utf-8").strip()
    sys.exit("No Scopus API key: set SCOPUS_API_KEY or create scopus_api_key.txt")


def get_insttoken() -> str | None:
    t = os.environ.get("SCOPUS_INSTTOKEN", "").strip()
    if t:
        return t
    f = KEY_ROOT / "scopus_insttoken.txt"
    if f.exists() and f.read_text(encoding="utf-8").strip():
        return f.read_text(encoding="utf-8").strip()
    return None


def scopus_headers() -> dict:
    h = {"X-ELS-APIKey": get_key(), "Accept": "application/json", **UA}
    tok = get_insttoken()
    if tok:
        h["X-ELS-Insttoken"] = tok
    return h


# ---------------------------------------------------------------- cache
class Cache:
    def __init__(self, path: Path):
        self.c = sqlite3.connect(path)
        self.c.execute("CREATE TABLE IF NOT EXISTS r (doi TEXT PRIMARY KEY, status INTEGER, payload TEXT, t REAL)")
        self.c.commit()

    def get(self, doi):
        row = self.c.execute("SELECT status, payload FROM r WHERE doi=?", (doi,)).fetchone()
        return row

    def put(self, doi, status, payload):
        self.c.execute("INSERT OR REPLACE INTO r VALUES (?,?,?,?)", (doi, status, payload, time.time()))
        self.c.commit()


# ---------------------------------------------------------------- OpenAlex sample
def openalex_sample(n: int, seed: int) -> list[dict]:
    """Seeded random sample of works with DOIs in the topic. Uses OpenAlex's
    `sample` parameter (max 10,000 per sample; per-page 200)."""
    out, page = [], 1
    while len(out) < n:
        params = {
            "filter": f"topics.id:{TOPIC},publication_year:2018-2024,has_doi:true,type:article|review",
            "sample": n, "seed": seed, "per-page": 200, "page": page, "mailto": MAILTO,
            "select": "id,doi,publication_year,authorships",
        }
        r = requests.get(OPENALEX, params=params, headers=UA, timeout=60)
        r.raise_for_status()
        res = r.json().get("results", [])
        if not res:
            break
        out.extend(res)
        page += 1
        time.sleep(0.2)
    return out[:n]


def openalex_affil_rows(w: dict) -> list[dict]:
    rows = []
    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
    for a in w.get("authorships", []):
        insts = a.get("institutions", []) or []
        for s in a.get("raw_affiliation_strings", []) or []:
            rows.append({
                "doi": doi, "openalex_id": w["id"].rsplit("/", 1)[-1],
                "raw_affiliation_string": s,
                "institution_ids": "|".join(i.get("id", "").rsplit("/", 1)[-1] for i in insts),
                "country_codes": "|".join(sorted({i.get("country_code") or "" for i in insts} - {""})),
            })
    return rows


# ---------------------------------------------------------------- Scopus
def aslist(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def fetch_abstract(doi: str, headers: dict, cache: Cache) -> tuple[int, dict | None, dict]:
    hit = cache.get(doi)
    if hit:
        st, payload = hit
        return st, (json.loads(payload) if payload else None), {}
    for attempt in range(4):
        try:
            r = requests.get(f"{SCOPUS_ABS}/{doi}", headers=headers, timeout=60)
        except requests.RequestException as e:
            time.sleep(2 ** attempt)
            continue
        quota = {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
        if r.status_code == 429:
            print("  429 rate-limited; sleeping 60 s"); time.sleep(60); continue
        if r.status_code in (200, 404):
            cache.put(doi, r.status_code, r.text if r.status_code == 200 else "")
            time.sleep(SLEEP_SCOPUS)
            return r.status_code, (r.json() if r.status_code == 200 else None), quota
        if r.status_code in (401, 403):
            sys.exit(f"Scopus returned {r.status_code} on the default view. Body: {r.text[:300]}")
        time.sleep(2 ** attempt)
    cache.put(doi, 0, "")
    return 0, None, {}


def parse_affiliations(payload: dict, doi: str) -> list[dict]:
    """Default-view response: abstracts-retrieval-response/affiliation = dict or list of
    {affilname, affiliation-city, affiliation-country, @id}. Work-level, not per author."""
    try:
        root = payload["abstracts-retrieval-response"]
    except (KeyError, TypeError):
        return []
    rows = []
    for i, a in enumerate(aslist(root.get("affiliation"))):
        if not isinstance(a, dict):
            continue
        rows.append({"doi": doi, "idx": i,
                     "affilname": a.get("affilname") or "",
                     "city": a.get("affiliation-city") or "",
                     "country": a.get("affiliation-country") or "",
                     "afid": a.get("@id") or ""})
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--probe", action="store_true", help="fetch 3 DOIs only and print field availability")
    args = ap.parse_args()

    headers = scopus_headers()
    cache = Cache(CACHE_DIR / "abstract_full.sqlite")

    print(f"[1/3] OpenAlex sample: {args.n} works, seed {args.seed}, topic {TOPIC}")
    works = openalex_sample(3 if args.probe else args.n, args.seed)
    print(f"      got {len(works)} works")

    # OpenAlex raw strings for the same works
    oa_rows = [r for w in works for r in openalex_affil_rows(w)]

    print(f"[2/3] Scopus Abstract Retrieval (default view) for {len(works)} DOIs")
    sc_rows, sample_rows, last_quota = [], [], {}
    n_found = n_404 = n_err = n_with_src = 0
    for i, w in enumerate(works, 1):
        doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
        st, payload, quota = fetch_abstract(doi, headers, cache)
        if quota:
            last_quota = quota
        groups = parse_affiliations(payload, doi) if payload else []
        if st == 200:
            n_found += 1
            if any(g["city"] and g["country"] for g in groups):
                n_with_src += 1
        elif st == 404:
            n_404 += 1
        else:
            n_err += 1
        sc_rows.extend(groups)
        sample_rows.append({"openalex_id": w["id"].rsplit("/", 1)[-1], "doi": doi,
                            "year": w.get("publication_year"), "scopus_status": st,
                            "n_affiliations": len(groups),
                            "n_with_city_and_country": sum(1 for g in groups if g["city"] and g["country"])})
        if i % 100 == 0 or args.probe:
            print(f"      {i}/{len(works)}  found={n_found} 404={n_404} err={n_err} "
                  f"with_city+country={n_with_src}  quota={last_quota.get('X-RateLimit-Remaining', '?')}")

    if args.probe:
        print("\n=== PROBE RESULT ===")
        for g in sc_rows[:6]:
            print(json.dumps({k: (v[:80] if isinstance(v, str) else v) for k, v in g.items()}, ensure_ascii=False))
        print(f"\nworks found in Scopus: {n_found}/{len(works)}; with structured city+country: {n_with_src}")
        return

    print("[3/3] writing outputs")
    def dump(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    dump(HERE / "work_sample.csv", sample_rows,
         ["openalex_id", "doi", "year", "scopus_status", "n_affiliations", "n_with_city_and_country"])
    dump(HERE / "scopus_affils.csv", sc_rows,
         ["doi", "idx", "affilname", "city", "country", "afid"])
    dump(HERE / "openalex_affils.csv", oa_rows,
         ["doi", "openalex_id", "raw_affiliation_string", "institution_ids", "country_codes"])
    log = {"n_works": len(works), "seed": args.seed, "topic": TOPIC,
           "scopus_found": n_found, "scopus_404": n_404, "scopus_error": n_err,
           "works_with_city_and_country": n_with_src,
           "scopus_affil_rows": len(sc_rows), "openalex_affil_rows": len(oa_rows),
           "last_quota_headers": last_quota, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(log, open(HERE / "fetch_log.json", "w"), indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
