#!/usr/bin/env python3
"""
s3_notfound_subset.py — optional follow-up to s2.

s2 compared OpenAlex raw strings only for works that Scopus also indexes (1,806/2,000).
This script geocodes the OpenAlex raw strings of the works Scopus does NOT index
(the ~194 remaining works) with the same cascade, so we can say whether the
OpenAlex-only part of the corpus has messier strings. Reuses the s2 cache.

    python s3_notfound_subset.py
"""
import json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_geocode_compare import load_cache, geocode_address, summarise, HERE

sample = pd.read_csv(HERE / "work_sample.csv", dtype=str).fillna("")
oa = pd.read_csv(HERE / "openalex_affils.csv", dtype=str).fillna("")
notfound = set(sample.loc[sample.scopus_status != "200", "doi"])
found = set(sample.loc[sample.scopus_status == "200", "doi"])
B_nf = sorted(set(oa[oa.doi.isin(notfound)].raw_affiliation_string))
B_f = sorted(set(oa[oa.doi.isin(found)].raw_affiliation_string))
print(f"works not in Scopus: {len(notfound)}; their unique OpenAlex strings: {len(B_nf)}")
cache = load_cache()
todo = [s for s in B_nf if s not in cache]
print(f"{len(todo)} to geocode (~{len(todo)*1.6/60:.0f} min)")
t0 = time.time()
for i, s in enumerate(todo, 1):
    geocode_address(s, cache)
    if i % 100 == 0:
        print(f"  {i}/{len(todo)}  {(time.time()-t0)/60:.0f} min")
res = {"B_openalex_works_NOT_in_scopus": summarise("B' OpenAlex raw strings, works not indexed by Scopus", B_nf, cache),
       "B_openalex_works_in_scopus_all_cached": summarise("B  OpenAlex raw strings, works indexed by Scopus (cached subset)", [s for s in B_f if s in cache], cache),
       "n_works_not_in_scopus": len(notfound), "n_works_in_scopus": len(found)}
json.dump(res, open(HERE / "scopus_test_notfound.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
