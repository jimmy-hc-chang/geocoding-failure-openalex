#!/usr/bin/env python3
"""
L4_drift_control.py — gazetteer-drift control for L1.

L1 re-queried decoded HTML-entity strings in 2026-10 and found 69% now succeed. Part of that
could be gazetteer growth since the 2026-07 run rather than decoding. This re-queries, unchanged,
a random 300 of the failed strings that contain NO entity: the share that now succeeds is the
drift rate to subtract from L1.

Output: L4_drift_control_summary.json      Time: ~15-25 min. Ctrl+C safe.
    python L4_drift_control.py
"""
import json, time
from pathlib import Path
import pandas as pd
from cascade import Cache, geocode_with_step
HERE = Path(__file__).resolve().parent
cache = Cache(HERE / "cache_L4.json")
s = pd.read_csv(HERE / "input_drift_control_300.csv", dtype=str).fillna("")
rows, t0 = [], time.time()
for i, r in enumerate(s.itertuples(index=False), 1):
    rows.append({"Raw_Affiliation": r.Raw_Affiliation, "country": r.country, **geocode_with_step(r.Raw_Affiliation, cache)})
    if i % 50 == 0:
        el = time.time() - t0; print(f"  {i}/{len(s)}  {el/60:.0f} min, ~{(len(s)-i)*el/i/60:.0f} min left")
R = pd.DataFrame(rows); R.to_csv(HERE / "L4_drift_control_results.csv", index=False, encoding="utf-8-sig")
ok = R[R.success.notna()]
summ = {"n": int(len(ok)), "now_succeed": int((ok.success == True).sum()), "drift_pct": round(float((ok.success == True).mean() * 100), 1),  # noqa: E712
        "by_step": ok[ok.success == True].step.value_counts().to_dict(), "run_date": time.strftime("%Y-%m-%d")}  # noqa: E712
json.dump(summ, open(HERE / "L4_drift_control_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
