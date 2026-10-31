#!/usr/bin/env python3
"""
L1_decode_requery.py  (Reviewer 2 minor 4) — replace the S4.6 upper bound with a measurement.

All 3,098 strings containing an HTML entity are decoded (html.unescape). The 1,143 that
FAILED in the original run are re-queried through the same cascade in decoded form.
(The successful ones are not re-queried: decoding cannot turn a success into a failure for
the purposes of the failure rate, and re-querying them would mix in gazetteer drift.)

Output: L1_decode_requery_results.csv, L1_decode_requery_summary.json
Time:   ~1,143 strings, up to 6 cascade variants each at 1.1 s → 30-90 min. Ctrl+C safe.

    python L1_decode_requery.py
"""
import json
import time
from pathlib import Path

import pandas as pd

from cascade import Cache, geocode_with_step

HERE = Path(__file__).resolve().parent
cache = Cache(HERE / "cache_L1.json")
df = pd.read_csv(HERE / "input_html_entity_strings.csv", dtype=str).fillna("")
df["was_success"] = df.was_success.str.lower() == "true"
todo = df[~df.was_success].copy()
print(f"{len(df)} entity strings; {len(todo)} failed originally → re-query decoded form")
rows, t0 = [], time.time()
for i, r in enumerate(todo.itertuples(index=False), 1):
    res = geocode_with_step(r.decoded, cache)
    rows.append({"Raw_Affiliation": r.Raw_Affiliation, "decoded": r.decoded, "country": r.country, **res})
    if i % 50 == 0:
        el = time.time() - t0
        print(f"  {i}/{len(todo)}  {el/60:.0f} min, ~{(len(todo)-i)*el/i/60:.0f} min left")
R = pd.DataFrame(rows)
R.to_csv(HERE / "L1_decode_requery_results.csv", index=False, encoding="utf-8-sig")
ok = R[R.success.notna()]
summ = {"n_entity_strings": int(len(df)), "n_failed_original": int(len(todo)),
        "n_requeried": int(len(ok)), "n_network_fail": int((R.step == "network_fail").sum()),
        "rescued_by_decoding": int((ok.success == True).sum()),  # noqa: E712
        "rescued_pct_of_failed_entity_strings": round(float((ok.success == True).mean() * 100), 1),  # noqa: E712
        "rescued_by_step": ok[ok.success == True].step.value_counts().to_dict(),  # noqa: E712
        "run_date": time.strftime("%Y-%m-%d")}
json.dump(summ, open(HERE / "L1_decode_requery_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print("\nCorpus failure rate after decoding would be (10,721 - rescued) / 80,176. Paste the summary back.")
