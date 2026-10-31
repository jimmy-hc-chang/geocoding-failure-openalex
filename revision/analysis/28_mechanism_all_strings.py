#!/usr/bin/env python3
"""
28_mechanism_all_strings.py

Mechanism statistics on ALL unique strings (main-text Section 3.3, Abstract,
Section 3.5, Supplementary Tables S9 and S14): failure by comma count, the share
of strings and of failures with at most one comma, and failure with and without
unescaped HTML entities; plus the Latin American entity share (strings with a
region) and the composition distortion of the geocoded sample (strings with a
region, unrounded). Run for silicon carbide in full_run/ and, with
CORPUS=cart, in cart_run/ (gzipped inputs).
"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_rule import assign_country, REGION  # noqa: E402

gz = ".gz" if os.environ.get("CORPUS") == "cart" else ""
af = pd.read_csv("affil_full.csv" + gz, usecols=["Raw_Affiliation", "Institution_Countries"])
af = af[af.Raw_Affiliation.notna()]
g = pd.read_csv("geocoded.csv" + gz).merge(assign_country(af).rename("country"),
                                             left_on="Raw_Affiliation", right_index=True, how="left")
g["fail"] = g.Latitude.isna()
g["nc"] = g.Raw_Affiliation.str.count(",")
g["ent"] = g.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False)
g["region"] = g.country.map(REGION)
le1 = g.nc <= 1
out = {
    "n_strings": len(g), "fail_pct": round(g.fail.mean() * 100, 2),
    "fail_by_commas_pct": {str(k): round(v * 100, 2) for k, v in g.groupby(g.nc.clip(upper=4)).fail.mean().items()},
    "le1_comma_share_pct": round(le1.mean() * 100, 2),
    "le1_comma_share_of_failures_pct": round(g[le1].fail.sum() / g.fail.sum() * 100, 2),
    "html_entity_share_pct": round(g.ent.mean() * 100, 2),
    "fail_with_entity_pct": round(g[g.ent].fail.mean() * 100, 2),
    "fail_without_entity_pct": round(g[~g.ent].fail.mean() * 100, 2),
    "failed_entity_strings": int(g[g.ent].fail.sum()),
}
dr = g[g.region.notna()]
out["entity_share_by_region_pct"] = (dr.groupby("region").ent.mean() * 100).round(2).to_dict()
true = dr.region.value_counts(normalize=True); obs = dr[~dr.fail].region.value_counts(normalize=True)
out["composition_distortion_pct"] = ((obs / true - 1) * 100).round(2).to_dict()
name = "mechanism_all_strings_cart.json" if gz else "mechanism_all_strings.json"
json.dump(out, open(name, "w"), indent=1); print(json.dumps(out, indent=1)[:1500])
