#!/usr/bin/env python3
"""
L2_held_structure_requery.py  (Reviewer 2 Major 6; Supplementary S6.2)

Input: hand_coding/地名標註_300_作者.xlsx (the author's place-name annotation; instructions in sheet 先看這頁).

For each failed string, the place names you annotated (column F, smallest to largest,
separated by ';') are queried against public Nominatim in two forms:
   Q_full     all annotated names joined by ', '     e.g. "Coimbatore, Tamil Nadu, India"
   Q_specific the most specific name + the largest   e.g. "Coimbatore, India"
Classification:
   parser-side     at least one query returns a point in the string's catalogue country
                   (the place exists in the gazetteer; the raw string's structure hid it)
   gazetteer-side  annotated names present, but neither query returns a point in that country
   no place name   column G = N  (structural absence: neither mechanism applies)
A point in the wrong country is counted separately (ambiguous names).

Output: L2_held_structure_results.csv, L2_held_structure_summary.json
Time:   ≤ 600 queries at 1.1 s → ~12 min. Ctrl+C safe.

    python L2_held_structure_requery.py
"""
import html
import json
import math
import time
from pathlib import Path

import pandas as pd
import re

from cascade import Cache, nominatim

import sys

HERE = Path(__file__).resolve().parent
X, SHEET, TAG = HERE.parent / "hand_coding" / "地名標註_300_作者.xlsx", "填這頁", "author"
cache = Cache(HERE / "cache_L2.json")
df = pd.read_excel(X, sheet_name=SHEET, dtype=str).fillna("")
df = df.rename(columns={"地名": "place_names_annotated", "有沒有地名_YN": "has_any_place_name", "備註": "notes"})
# "?" = the annotator was unsure whether a word is a place name (allowed values: Y / N / ?)
df["has_any_place_name"] = df.has_any_place_name.str.strip().str.upper().str.replace("？", "?")
df["uncertain"] = (df.has_any_place_name == "?") | df.place_names_annotated.str.contains(r"[?？]", regex=True)
missing = df[(df.has_any_place_name == "") | (df.has_any_place_name.isin(["Y", "?"]) & (df.place_names_annotated.str.strip() == ""))]
if len(missing):
    print(f"!! {len(missing)} rows not yet annotated or Y/? without a name (e.g. {list(missing.id[:5])}). Finish annotation first.")
    raise SystemExit(1)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round((c - h) * 100, 1), round((c + h) * 100, 1)]


rows = []
for r in df.itertuples(index=False):
    if r.has_any_place_name == "N":
        rows.append({**r._asdict(), "q_full": "", "q_specific": "", "hit_full_cc": "", "hit_spec_cc": "", "class": "no_place_name"})
        continue
    # split on the ';' separator only — not on the ';' that closes an HTML entity such as &#x00E9;
    names = [x.strip().rstrip("?？").strip() for x in re.split(r";(?=\s|$)", r.place_names_annotated.replace("；", ";")) if x.strip().rstrip("?？").strip()]
    q_full = ", ".join(names)
    q_spec = ", ".join([names[0], names[-1]]) if len(names) > 1 else names[0]
    a = nominatim(q_full, cache)
    b = nominatim(q_spec, cache) if q_spec != q_full else a
    cc = (r.country or "").upper()
    cc_a = a[3] if isinstance(a, list) else ""
    cc_b = b[3] if isinstance(b, list) else ""
    # second pass: the same names with HTML entities decoded (e.g. "Para&#x00ED;ba" -> "Paraíba").
    # A hit here means the place exists and the failure was text corruption (hygiene side), not gazetteer absence.
    dn = [html.unescape(x) for x in names]
    q_dec = ", ".join(dn)
    q_dec_s = ", ".join([dn[0], dn[-1]]) if len(dn) > 1 else dn[0]
    cc_d = cc_ds = ""
    if (q_dec != q_full) or (q_dec_s != q_spec):
        c1 = nominatim(q_dec, cache) if q_dec != q_full else a
        c2 = nominatim(q_dec_s, cache) if q_dec_s != q_spec else b
        cc_d = c1[3] if isinstance(c1, list) else ""
        cc_ds = c2[3] if isinstance(c2, list) else ""
    if "NETWORK_FAIL" in (a, b):
        cls = "network_fail"
    elif cc and (cc_a == cc or cc_b == cc):
        cls = "parser_side"
    elif cc and (cc_d == cc or cc_ds == cc):
        cls = "hygiene_side"                      # found only after decoding / de-spacing
    elif (cc_a or cc_b or cc_d or cc_ds) and cc:
        cls = "found_wrong_country"
    elif (cc_a or cc_b or cc_d or cc_ds) and not cc:
        cls = "parser_side_no_catalogue_country"
    else:
        cls = "gazetteer_side"                    # still not found: typo, truncation, or genuinely absent
    # an unsure name that the gazetteer cannot find is ambiguous: an unknown place, or not a place at all
    if r.uncertain and cls == "gazetteer_side":
        cls = "uncertain_name_not_found"
    rows.append({**r._asdict(), "q_full": q_full, "q_specific": q_spec, "hit_full_cc": cc_a, "hit_spec_cc": cc_b, "q_decoded": q_dec, "hit_decoded_cc": cc_d or cc_ds, "class": cls})
R = pd.DataFrame(rows)
R.to_csv(HERE / f"L2_held_structure_results_{TAG}.csv", index=False, encoding="utf-8-sig")
n = int((R["class"] != "network_fail").sum())
summ = {"annotation": TAG, "source_file": str(X.name), "n": n, "n_uncertain_rows": int(R.uncertain.sum()),
        "run_date": time.strftime("%Y-%m-%d"), "overall": {}, "by_region": {}, "by_comma": {}}
# bounds for the gazetteer-side share: ambiguous rows counted as not-a-place (lower) or as gazetteer-side (upper)
k_g = int((R["class"] == "gazetteer_side").sum())
k_u = int((R["class"] == "uncertain_name_not_found").sum())
summ["gazetteer_side_bounds_pct"] = {"lower_uncertain_as_no_place": round(k_g / n * 100, 1),
                                     "upper_uncertain_as_gazetteer": round((k_g + k_u) / n * 100, 1)}
for c, k in R["class"].value_counts().items():
    summ["overall"][c] = {"n": int(k), "pct": round(k / n * 100, 1), "wilson95": wilson(int(k), n)}
for grp, col in [("by_region", "region"), ("by_comma", "n_commas")]:
    for g, sub in R.groupby(R[col] if col == "region" else R[col].astype(int).clip(upper=2).map({0: "0", 1: "1", 2: ">=2"})):
        m = len(sub)
        summ[grp][str(g)] = {c: round(float((sub["class"] == c).mean() * 100), 1) for c in R["class"].unique()} | {"n": m}
json.dump(summ, open(HERE / f"L2_held_structure_summary_{TAG}.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
