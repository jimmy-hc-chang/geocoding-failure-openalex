#!/usr/bin/env python3
"""
s0_probe_views.py — find out which Abstract Retrieval view / field combination this key
is entitled to, and whether any of them exposes the raw affiliation text (ce:source-text)
or the author-group block. Uses ONE well-known DOI. Prints a table; writes probe_views.json.

    python s0_probe_views.py
"""
import json, sys, time
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_fetch_scopus import scopus_headers, SCOPUS_ABS, UA

DOI = "10.1016/j.joi.2018.11.003"      # Abramo, D'Angelo & Felici 2019, JOI 13(1) — definitely in Scopus
COMBOS = [
    (None, None),
    (None, "author-group"),
    (None, "author-group,affiliation"),
    (None, "affiliation,author,dc:identifier"),
    ("META", None),
    ("META_ABS", None),
    ("META_ABS", "author-group"),
    ("FULL", None),
    ("FULL", "author-group"),
    ("ENTITLED", None),
]

def find_keys(o, targets, found=None, path=""):
    if found is None: found = {}
    if isinstance(o, dict):
        for k, v in o.items():
            if k in targets and k not in found: found[k] = path + "/" + k
            find_keys(v, targets, found, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]): find_keys(v, targets, found, path + f"[{i}]")
    return found

h = scopus_headers()
out = []
print(f"{'view':10} {'field':36} {'status':7} source-text  author-group  affiliation  city  @afid")
for view, field in COMBOS:
    params = {}
    if view: params["view"] = view
    if field: params["field"] = field
    try:
        r = requests.get(f"{SCOPUS_ABS}/{DOI}", params=params, headers=h, timeout=60)
        st = r.status_code
        keys = find_keys(r.json(), {"ce:source-text", "author-group", "affiliation", "city", "@afid", "affiliation-city"}) if st == 200 else {}
        err = r.json().get("service-error", {}).get("status", {}).get("statusText", "")[:60] if st != 200 else ""
    except Exception as e:
        st, keys, err = -1, {}, str(e)[:60]
    row = {"view": view, "field": field, "status": st,
           "has_source_text": "ce:source-text" in keys, "has_author_group": "author-group" in keys,
           "has_affiliation": "affiliation" in keys, "has_city": ("city" in keys) or ("affiliation-city" in keys),
           "has_afid": "@afid" in keys, "paths": keys, "error": err}
    out.append(row)
    print(f"{str(view):10} {str(field):36} {st:<7} {str(row['has_source_text']):12} {str(row['has_author_group']):13} "
          f"{str(row['has_affiliation']):12} {str(row['has_city']):5} {row['has_afid']}  {err}")
    time.sleep(0.4)
json.dump(out, open(Path(__file__).resolve().parent / "probe_views.json", "w"), indent=1)
print("\nwritten probe_views.json — paste this table back.")
