#!/usr/bin/env python3
"""
L3_cascade_step_requery.py  (Reviewer 2 Major 3: "report discordance by ... cascade step")

The original cache stored coordinates only, not which cascade variant produced them.
This re-runs the cascade on a stratified sample of SUCCESSFUL strings with a catalogue
country (600 whose geocode falls in another country + 600 concordant), recording the step.
Discordance by step is then estimated with the stratum weights (2,442 discordant and
59,712 concordant in the full set).

Caveat recorded in the output: a re-query in 2026-10 is against a later gazetteer than the
July run, so a minority of strings may now resolve at a different step or place; the
script reports how often the re-query reproduces the original point within 1 km.

Output: L3_cascade_step_results.csv, L3_cascade_step_summary.json
Time:   1,200 strings × ~1.5 queries ≈ 35-45 min. Ctrl+C safe.

    python L3_cascade_step_requery.py
"""
import json
import math
import time
from pathlib import Path

import pandas as pd

from cascade import Cache, geocode_with_step

HERE = Path(__file__).resolve().parent
N_DISC, N_CONC = 2442, 59712
cache = Cache(HERE / "cache_L3.json")
s = pd.read_csv(HERE / "input_cascade_step_sample.csv", dtype={"country": str, "geo_cc": str})
s["disc"] = s.disc.astype(str).str.lower() == "true"


def hav(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


rows, t0 = [], time.time()
for i, r in enumerate(s.itertuples(index=False), 1):
    res = geocode_with_step(r.Raw_Affiliation, cache)
    same = None
    if res.get("success"):
        same = hav(r.orig_lat, r.orig_lon, res["lat"], res["lon"]) <= 1.0
    rows.append({**r._asdict(), **res, "reproduces_original_point": same})
    if i % 100 == 0:
        el = time.time() - t0
        print(f"  {i}/{len(s)}  {el/60:.0f} min, ~{(len(s)-i)*el/i/60:.0f} min left")
R = pd.DataFrame(rows)
R.to_csv(HERE / "L3_cascade_step_results.csv", index=False, encoding="utf-8-sig")
ok = R[R.success == True]  # noqa: E712
ok = ok.assign(w=ok.disc.map({True: N_DISC / 600, False: N_CONC / 600}))
by = ok.groupby("step").apply(lambda g: pd.Series({"n_sample": len(g), "weighted_share_of_successes_pct": g.w.sum() / ok.w.sum() * 100,
                                                 "discordance_pct": (g.w * g.disc).sum() / g.w.sum() * 100}))
summ = {"n_sample": int(len(R)), "requery_success_pct": round(float((R.success == True).mean() * 100), 1),  # noqa: E712
        "reproduces_original_point_within_1km_pct": round(float(ok.reproduces_original_point.mean() * 100), 1),
        "by_step": {k: {kk: round(float(vv), 2) for kk, vv in v.items()} for k, v in by.to_dict("index").items()},
        "run_date": time.strftime("%Y-%m-%d"),
        "caveat": "re-query against the 2026-10 gazetteer; original run was 2026-07"}
json.dump(summ, open(HERE / "L3_cascade_step_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
