#!/usr/bin/env python3
"""
s2_geocode_compare.py  —  Scopus raw-affiliation mini-test, step 2 (geocode + compare)

Runs the manuscript's cascade (copied verbatim from scpark-network/pipeline/
02_geocode_and_enrich.py: clean_address_variants + Nominatim, 1 req/s, local cache)
on three string sets drawn from the SAME works:

  A  Scopus structured "affilname, city, country"  (fullest structured form Scopus returns)
  B  OpenAlex raw affiliation strings               (raw_affiliation_strings)
  C  Scopus structured "city, country"              (what a Scopus user would actually geocode)

Note: this key's Scopus service level does not expose the raw affiliation text
(ce:source-text / author-group), so the Scopus side is structured fields only.
Reports failure rates, the comma gradient, and the HTML-entity share for each,
plus a paired work-level comparison.

Only AGGREGATES go into results; the per-string files stay local.

Outputs:
  scopus_test/cache/geo_cache_scopus_test.json     resumable geocode cache
  scopus_test/scopus_test_results.json             aggregates for the manuscript
  scopus_test/scopus_test_table.md                 ready-to-paste comparison table

Run (after s1):   python s2_geocode_compare.py --max-per-set 3000 --seed 20260811
Time: ~1.1 s per Nominatim query; with 3,000 + 3,000 + ~800 unique strings and the
cascade's extra variants on failures, expect roughly 2.5-3.5 hours. Safe to interrupt
and re-run; the cache resumes.
"""
import argparse
import html
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE_FILE = HERE / "cache" / "geo_cache_scopus_test.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "micro-geo-innovation-mapper/0.1 (dddsss5419@gmail.com)"}
NOMINATIM_MIN_INTERVAL = 1.1

# ---------------------------------------------------------------- cascade (verbatim)
INSTITUTION_KEYWORDS = re.compile(
    r'\b(university|institute|college|hospital|school|center|centre|'
    r'corp|inc|ltd|laboratory|lab|academy|foundation|clinic|company)\b',
    re.IGNORECASE,
)


def clean_address_variants(address):
    variants = [address]
    no_footnote = re.sub(r'^\d+(?=[A-Za-z])', '', address).strip()
    if no_footnote and no_footnote != address:
        variants.append(no_footnote)
    no_email = re.sub(r'Electronic address:\s*\S+', '', no_footnote, flags=re.IGNORECASE)
    no_email = re.sub(r'\S+@\S+', '', no_email).strip(' .;,')
    if no_email and no_email != no_footnote:
        variants.append(no_email)
    parts = [p.strip() for p in no_email.split(',') if p.strip()]
    inst_idx = None
    for i, p in enumerate(parts):
        if INSTITUTION_KEYWORDS.search(p):
            inst_idx = i
    if inst_idx is not None and inst_idx > 0:
        variants.append(', '.join(parts[inst_idx:]))
    for n in (3, 2):
        if len(parts) > n:
            variants.append(', '.join(parts[-n:]))
    seen, uniq = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v); uniq.append(v)
    return uniq


def _nominatim_call(q, max_retries=3):
    params = {"q": q, "format": "jsonv2", "limit": 1}
    for attempt in range(max_retries):
        try:
            r = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json(), True
        except Exception as e:
            print(f"[retry {attempt+1}/{max_retries}] {q[:60]}: {e}")
            time.sleep(2 ** attempt)
    return None, False


def load_cache():
    if CACHE_FILE.exists():
        return json.load(open(CACHE_FILE, encoding="utf-8"))
    return {}


def save_cache(c):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    json.dump(c, open(CACHE_FILE, "w", encoding="utf-8"), ensure_ascii=False)


def geocode_address(address, cache):
    """Returns {'lat','lon','variant_idx'} or {'lat':None,...}. Cache key = original string."""
    if address in cache:
        return cache[address]
    for vi, v in enumerate(clean_address_variants(address)):
        results, ok = _nominatim_call(v)
        if not ok:
            time.sleep(NOMINATIM_MIN_INTERVAL)
            return {"lat": None, "lon": None, "variant_idx": None, "network_fail": True}
        time.sleep(NOMINATIM_MIN_INTERVAL)
        if results:
            rec = {"lat": float(results[0]["lat"]), "lon": float(results[0]["lon"]), "variant_idx": vi}
            cache[address] = rec; save_cache(cache); return rec
    rec = {"lat": None, "lon": None, "variant_idx": None}
    cache[address] = rec; save_cache(cache); return rec


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round((c - h) * 100, 1), round((c + h) * 100, 1))


HTML_ENT = re.compile(r"&#x|&amp;|&\w+;")


def comma_bin(s):
    n = s.count(",")
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3+"


def norm(s):
    return re.sub(r"\s+", " ", html.unescape(s)).strip().lower()


def summarise(name, strings, cache):
    res = {s: cache[s] for s in strings if s in cache}
    n = len(res)
    fails = [s for s, r in res.items() if r["lat"] is None]
    out = {"set": name, "n_unique_strings": n, "n_failed": len(fails),
           "fail_rate_pct": round(len(fails) / n * 100, 1) if n else None,
           "fail_rate_wilson95": wilson(len(fails), n),
           "html_entity_share_pct": round(sum(bool(HTML_ENT.search(s)) for s in res) / n * 100, 1) if n else None,
           "mean_length": round(sum(len(s) for s in res) / n, 1) if n else None,
           "mean_commas": round(sum(s.count(",") for s in res) / n, 2) if n else None,
           "by_comma_bin": {}, "rescued_by_variant_pct": None}
    bins = Counter(comma_bin(s) for s in res)
    fb = Counter(comma_bin(s) for s in fails)
    for b in ["0", "1", "2", "3+"]:
        out["by_comma_bin"][b] = {"n": bins[b], "fail_pct": round(fb[b] / bins[b] * 100, 1) if bins[b] else None}
    succ = [r for r in res.values() if r["lat"] is not None]
    if succ:
        out["rescued_by_variant_pct"] = round(sum(1 for r in succ if (r.get("variant_idx") or 0) > 0) / len(succ) * 100, 1)
    le1 = [s for s in res if s.count(",") <= 1]
    out["strings_le1_comma_pct"] = round(len(le1) / n * 100, 1) if n else None
    out["failures_from_le1_comma_pct"] = round(sum(1 for s in fails if s.count(",") <= 1) / len(fails) * 100, 1) if fails else None
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-set", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--dry", action="store_true", help="no geocoding; only report set sizes and A/B text identity")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sc = pd.read_csv(HERE / "scopus_affils.csv", dtype=str).fillna("")
    oa = pd.read_csv(HERE / "openalex_affils.csv", dtype=str).fillna("")
    sample = pd.read_csv(HERE / "work_sample.csv", dtype=str).fillna("")
    found = set(sample.loc[sample.scopus_status == "200", "doi"])
    sc = sc[sc.doi.isin(found)]
    oa = oa[oa.doi.isin(found)]                      # same works only
    struct_ok = sc[(sc.city != "") & (sc.country != "")]
    full_ok = struct_ok[struct_ok.affilname != ""]
    sc_src = full_ok.assign(source_text=full_ok.affilname + ", " + full_ok.city + ", " + full_ok.country)

    cov = {"works_in_both": len(found),
           "scopus_affiliation_rows": len(sc),
           "scopus_rows_with_city_and_country_pct": round(len(struct_ok) / len(sc) * 100, 1) if len(sc) else None,
           "scopus_rows_with_country_pct": round((sc.country != "").mean() * 100, 1) if len(sc) else None,
           "scopus_rows_with_afid_pct": round((sc.afid != "").mean() * 100, 1) if len(sc) else None,
           "openalex_raw_strings": len(oa),
           "openalex_strings_with_institution_pct": round((oa.institution_ids != "").mean() * 100, 1) if len(oa) else None,
           "note": "Scopus raw affiliation text not available at this API service level; Scopus sets are structured fields"}

    A = sorted(set(sc_src.source_text))
    B = sorted(set(oa.raw_affiliation_string))
    C = sorted({f"{c}, {k}" for c, k in zip(struct_ok.city, struct_ok.country)})
    rng.shuffle(A); rng.shuffle(B); rng.shuffle(C)
    A, B, C = A[:args.max_per_set], B[:args.max_per_set], C[:args.max_per_set]
    print(json.dumps(cov, indent=1))
    print(f"unique strings to geocode: A={len(A)} B={len(B)} C={len(C)}")
    if args.dry:
        return

    cache = load_cache()
    todo = [s for s in A + B + C if s not in cache]
    print(f"{len(todo)} not yet cached; at ~1.1 s/query plus cascade variants this is roughly "
          f"{len(todo) * 1.6 / 3600:.1f} h")
    t0 = time.time()
    for i, s in enumerate(todo, 1):
        geocode_address(s, cache)
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el/60:.0f} min elapsed, ~{(len(todo)-i)*el/i/60:.0f} min left")

    res = {"coverage": cov,
           "A_scopus_structured_full": summarise("A Scopus affilname, city, country", A, cache),
           "B_openalex_raw": summarise("B OpenAlex raw_affiliation_strings", B, cache),
           "C_scopus_structured": summarise("C Scopus city, country", C, cache),
           "design": {"topic": "T10361 2018-2024", "seed": args.seed, "max_per_set": args.max_per_set,
                      "cascade": "clean_address_variants (identical to manuscript)", "geocoder": "Nominatim public, 1 req/s",
                      "run_date": time.strftime("%Y-%m-%d")}}

    # paired work-level: all groups/strings succeed?
    def work_all_ok(df, col):
        d = df[df[col].isin(cache.keys())].copy()
        d["ok"] = d[col].map(lambda s: cache[s]["lat"] is not None)
        g = d.groupby("doi").ok.agg(["all", "size"])
        return g
    ga = work_all_ok(sc_src, "source_text"); gb = work_all_ok(oa, "raw_affiliation_string")
    both = ga.join(gb, lsuffix="_A", rsuffix="_B", how="inner")
    if len(both):
        res["paired_works"] = {"n_works": int(len(both)),
                               "all_strings_geocoded_A_pct": round(both["all_A"].mean() * 100, 1),
                               "all_strings_geocoded_B_pct": round(both["all_B"].mean() * 100, 1),
                               "A_ok_B_fail_pct": round(((both.all_A) & (~both.all_B)).mean() * 100, 1),
                               "B_ok_A_fail_pct": round(((~both.all_A) & (both.all_B)).mean() * 100, 1)}

    json.dump(res, open(HERE / "scopus_test_results.json", "w"), indent=1, ensure_ascii=False)

    # markdown table
    rows = [res["A_scopus_structured_full"], res["B_openalex_raw"], res["C_scopus_structured"]]
    md = ["| Set | Unique strings | Failure % [95% CI] | ≤1 comma: share of strings % / share of failures % | 0-comma fail % | 1-comma fail % | 2-comma fail % | 3+ fail % | HTML-entity % | Mean commas |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b = r["by_comma_bin"]
        md.append(f"| {r['set']} | {r['n_unique_strings']:,} | {r['fail_rate_pct']} {list(r['fail_rate_wilson95'])} | "
                  f"{r['strings_le1_comma_pct']} / {r['failures_from_le1_comma_pct']} | {b['0']['fail_pct']} | {b['1']['fail_pct']} | "
                  f"{b['2']['fail_pct']} | {b['3+']['fail_pct']} | {r['html_entity_share_pct']} | {r['mean_commas']} |")
    md.append("")
    md.append(f"Coverage: {json.dumps(cov)}")
    if "paired_works" in res:
        md.append(f"Paired works: {json.dumps(res['paired_works'])}")
    open(HERE / "scopus_test_table.md", "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
