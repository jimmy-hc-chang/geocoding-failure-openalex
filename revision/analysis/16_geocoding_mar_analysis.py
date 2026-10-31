#!/usr/bin/env python3
"""
16_geocoding_mar_analysis.py

Primary analysis for the manuscript:
  "The geography of geocoding failure: non-random missingness in
   open-infrastructure spatial scientometrics, and the limits of its
   correction by inverse-probability weighting"

Run from full_run/ (cart_run/ must be a sibling directory):
    cd full_run
    python ../analysis/16_geocoding_mar_analysis.py

Inputs : affil_full.csv, geocoded.csv
Outputs: geo_mar_results.json, country_failure_table.csv,
         or_geocode.csv, or_resolution.csv, ipw_metrics.csv

This script produces no figures. Every figure is drawn by 20_make_figures.py
from these outputs and from the raw deposits.

Dependencies: pandas, numpy, scipy, statsmodels, reverse_geocoder
"""
import itertools
import json
from collections import defaultdict

import numpy as np
from country_rule import assign_country, tie_diagnostics, TIE_RULE
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

RNG_NOTE = "No stochastic steps; all estimates deterministic given inputs."

# ---------------------------------------------------------------- classifications
HIGH_INCOME = set("""US JP DE FR IT GB KR TW CA AU CH DK BE PL SE AT NL SA SG CL PT HK CZ NO IL SK AE FI HU GR QA MO IE HR EE NZ SI CY LV KW LT MT OM CR BN LU IS UY PR PA BH RO BG RU GY""".split())
GLOBAL_NORTH = set("""US CA JP DE FR IT GB KR TW AU NZ CH DK BE PL SE AT NL RO CZ NO SK FI HU GR IE HR EE SI CY LV LT MT LU IS PT ES BG RU UA BY MD RS ME MK BA AL IL SM AD MC LI GI""".split())
LATIN = set("""US CA GB AU NZ IE DE FR IT ES PT NL BE CH AT DK SE NO FI IS PL CZ SK HU RO HR SI EE LV LT MT LU CY TR BR MX CL CO AR EC PY PE UY CR PA CU VE DO GT HN NI SV BO ID MY VN PH SG NG ZA KE GH TZ MZ BW ZW CM SN BJ BF TD SS FJ MU AZ UZ TM AL BA ME MD TN DZ MA HT AO GQ CV ST TL PG SB VU WS TO KI NR TV FM MH PW BN""".split())

REGION = {}
for cc in "US CA".split(): REGION[cc] = "Northern America"
for cc in "MX GT HN SV NI CR PA CU DO HT JM TT BB BS PR BR AR CL CO PE EC VE UY PY BO GY SR".split(): REGION[cc] = "Latin America & Caribbean"
for cc in "DE FR IT GB ES PT NL BE CH AT DK SE NO FI IS IE LU MT CY GR".split(): REGION[cc] = "Western Europe"
for cc in "PL CZ SK HU RO BG HR SI EE LV LT RS ME MK BA AL UA BY MD RU KZ UZ TJ TM KG AZ AM GE XK".split(): REGION[cc] = "Eastern Europe & Central Asia"
for cc in "CN JP KR TW HK MO MN KP".split(): REGION[cc] = "East Asia"
for cc in "IN PK BD LK NP AF BT MV".split(): REGION[cc] = "South Asia"
for cc in "SG MY TH VN ID PH BN KH LA MM TL".split(): REGION[cc] = "Southeast Asia"
for cc in "IR IQ SA AE QA KW OM BH YE JO LB SY IL PS TR".split(): REGION[cc] = "Middle East"
for cc in "EG DZ TN MA LY SD MR".split(): REGION[cc] = "North Africa"
for cc in "NG ZA KE GH TZ MZ BW ZW CM SN BJ BF TD SS ET MU UG RW ZM MW AO CD CG GA CI ML NE SO SZ TG MG".split(): REGION[cc] = "Sub-Saharan Africa"
for cc in "AU NZ FJ PG".split(): REGION[cc] = "Oceania"

out = {}

# ---------------------------------------------------------------- 1. build string-level frame
af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Raw_Affiliation", "Institution_Countries", "Standardized_Institutions"])
af = af[af.Raw_Affiliation.notna()].copy()
out["tie_rule"] = TIE_RULE
out["ties"] = tie_diagnostics(af)

country = assign_country(af)

n_papers = af.groupby("Raw_Affiliation")["Paper_ID"].nunique().rename("n_papers")
af["has_std"] = af.Standardized_Institutions.notna()
resolved = af.groupby("Raw_Affiliation")["has_std"].mean().rename("resolved_share")
base = pd.concat([country.rename("country"), n_papers, resolved], axis=1).reset_index()
base = base.rename(columns={"index": "Raw_Affiliation"})

geo = pd.read_csv("geocoded.csv")
m = geo.merge(base, on="Raw_Affiliation", how="left")
m["success"] = m.Latitude.notna().astype(int)
m["fail"] = 1 - m.success
m["log_papers"] = np.log(m.n_papers)
m["strlen"] = m.Raw_Affiliation.str.len()
m["n_commas"] = m.Raw_Affiliation.str.count(",")
m["html_ent"] = m.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int)
m["global_south"] = m.country.map(lambda c: 0 if c in GLOBAL_NORTH else (1 if pd.notna(c) else np.nan))
m["non_latin"] = m.country.map(lambda c: 0 if c in LATIN else (1 if pd.notna(c) else np.nan))
m["region"] = m.country.map(REGION)

out["n_strings"] = len(m)
out["n_success"] = int(m.success.sum())
out["fail_rate_overall"] = round(m.fail.mean(), 4)
out["country_known_pct"] = round(m.country.notna().mean() * 100, 1)
out["country_known_failed_pct"] = round(m.loc[m.fail == 1, "country"].notna().mean() * 100, 1)

d = m[m.region.notna()].copy()

# ---------------------------------------------------------------- 2. MCAR test + country table
ct = pd.crosstab(d.country, d.fail)
chi2, p, dof, _ = stats.chi2_contingency(ct)
out["chi2"] = round(chi2, 1); out["chi2_dof"] = int(dof)
out["cramer_v"] = round(float(np.sqrt(chi2 / len(d))), 3)

tbl = d.groupby("country").agg(n=("fail", "size"), fail_rate=("fail", "mean")).sort_values("fail_rate", ascending=False)
tbl.to_csv("country_failure_table.csv")

# ---------------------------------------------------------------- 3. logits
def fit(formula, data, dep_note=""):
    r = smf.logit(formula, data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": data.country})
    ci = np.exp(r.conf_int())
    o = pd.DataFrame({"OR": np.exp(r.params), "lo": ci[0], "hi": ci[1], "p": r.pvalues}).round(4)
    o.index = [i.split("[T.")[-1].rstrip("]") for i in o.index]
    return o

M1 = fit("fail ~ global_south + non_latin + log_papers", d)
M2 = fit('fail ~ C(region, Treatment(reference="Northern America")) + log_papers', d)
M2b = fit('fail ~ C(region, Treatment(reference="Northern America")) + log_papers', d[d.html_ent == 0])
M3 = fit("fail ~ global_south + non_latin + log_papers + strlen + n_commas", d)
M2.to_csv("or_geocode.csv"); out["M1"] = M1.to_dict("index"); out["M3"] = M3.to_dict("index")
out["M2"] = M2.to_dict("index"); out["M2_noent"] = M2b.to_dict("index")

# comma / entity mechanisms
low = d[d.n_commas <= 1]
out["strings_0or1_comma_pct"] = round(len(low) / len(d) * 100, 1)
out["failures_from_0or1_comma_pct"] = round(low.fail.sum() / d.fail.sum() * 100, 1)
out["html_ent_pct"] = round(d.html_ent.mean() * 100, 2)
out["fail_html_ent"] = round(d[d.html_ent == 1].fail.mean() * 100, 1)
out["fail_no_ent"] = round(d[d.html_ent == 0].fail.mean() * 100, 1)

# ---------------------------------------------------------------- 4. IPW
prop = smf.logit("success ~ C(region) + log_papers + html_ent + n_commas + strlen", d).fit(disp=0)
d["phat"] = prop.predict(d)
d["w"] = np.where(d.success == 1, 1 / d.phat, 0.0)
cap = d.loc[d.success == 1, "w"].quantile(0.99)
d["w_t"] = d.w.clip(upper=cap)
out["ipw_weight_cap"] = round(float(cap), 3)

true_comp = d.region.value_counts(normalize=True)
geo_comp = d[d.success == 1].region.value_counts(normalize=True)
ipw_comp = d[d.success == 1].groupby("region").w_t.sum() / d.loc[d.success == 1, "w_t"].sum()
out["composition"] = {r: [round(true_comp[r] * 100, 2), round(geo_comp[r] * 100, 2), round(ipw_comp[r] * 100, 2)] for r in true_comp.index}

# pairs
ok = d[d.success == 1]
by_paper = defaultdict(list)
sub = af[["Paper_ID", "Raw_Affiliation"]].drop_duplicates().merge(
    ok[["Raw_Affiliation", "Latitude", "Longitude", "region", "country", "w_t"]], on="Raw_Affiliation")
for row in sub.itertuples(index=False):
    by_paper[row.Paper_ID].append((row.Latitude, row.Longitude, row.region, row.country, row.w_t))

def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

rows = []
for pid, lst in by_paper.items():
    if len(lst) < 2 or len(lst) > 60:
        continue
    for a, b in itertools.combinations(lst, 2):
        rows.append((a[0], a[1], b[0], b[1], a[2], b[2], a[3], b[3], a[4] * b[4]))
P = pd.DataFrame(rows, columns=["lat1", "lon1", "lat2", "lon2", "r1", "r2", "c1", "c2", "w"])
P["dist"] = hav(P.lat1, P.lon1, P.lat2, P.lon2)
P["intl"] = (P.c1 != P.c2).astype(int)
out["n_pairs"] = len(P)
out["n_papers_pairs"] = int(sum(1 for v in by_paper.values() if 2 <= len(v) <= 60))

def wq(x, w, q):
    i = np.argsort(x)
    x, w = np.asarray(x)[i], np.asarray(w)[i]
    return float(np.interp(q, np.cumsum(w) / w.sum(), x))

ipw = {
    "mean_dist_raw": P.dist.mean(), "mean_dist_ipw": np.average(P.dist, weights=P.w),
    "median_raw": P.dist.median(), "median_ipw": wq(P.dist, P.w, 0.5),
    "intl_share_raw": P.intl.mean(), "intl_share_ipw": np.average(P.intl, weights=P.w),
}
for reg in ["Latin America & Caribbean", "North Africa", "Middle East", "Sub-Saharan Africa"]:
    inv = ((P.r1 == reg) | (P.r2 == reg)).astype(float)
    ipw[f"invshare_raw::{reg}"] = inv.mean()
    ipw[f"invshare_ipw::{reg}"] = np.average(inv, weights=P.w)
intl = P[P.intl == 1]
invLA = ((intl.r1 == "Latin America & Caribbean") | (intl.r2 == "Latin America & Caribbean")).astype(float)
ipw["LA_share_of_intl_raw"] = invLA.mean()
ipw["LA_share_of_intl_ipw"] = np.average(invLA, weights=intl.w)
pd.Series(ipw).round(4).to_csv("ipw_metrics.csv")
out["ipw_metrics"] = {k: round(float(v), 4) for k, v in ipw.items()}

# ---------------------------------------------------------------- 5. entity-resolution layer
import reverse_geocoder as rg
mm = m.copy()
mm["unresolved"] = (mm.resolved_share == 0).astype(int)
out["unresolved_overall_pct"] = round(mm.unresolved.mean() * 100, 2)
out["geocode_fail_by_resolution"] = [round(mm[mm.unresolved == 1].fail.mean() * 100, 1),
                                     round(mm[mm.unresolved == 0].fail.mean() * 100, 1)]
okm = mm[mm.success == 1].copy()
res = rg.search(list(zip(okm.Latitude, okm.Longitude)), mode=1)
okm["geo_cc"] = [r["cc"] for r in res]
okm["geo_region"] = okm.geo_cc.map(REGION)
both = okm[okm.country.notna()]
out["reverse_geo_agreement_pct"] = round((both.geo_cc == both.country).mean() * 100, 1)
okm["log_papers"] = np.log(okm.n_papers)
s2 = okm[okm.geo_region.notna()].drop(columns=["region"]).rename(columns={"geo_region": "region"}).copy()
s2["country"] = s2.geo_cc
Mr = fit('unresolved ~ C(region, Treatment(reference="Northern America")) + log_papers', s2)
Mr.to_csv("or_resolution.csv")
out["resolution_model"] = Mr.to_dict("index")
out["unresolved_by_region"] = {r: [int(v.n), round(v.ur * 100, 2)] for r, v in
                               s2.groupby("region").agg(n=("unresolved", "size"), ur=("unresolved", "mean")).iterrows()}

json.dump(out, open("geo_mar_results.json", "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ["n_strings", "n_success", "fail_rate_overall", "chi2", "cramer_v",
                                      "strings_0or1_comma_pct", "failures_from_0or1_comma_pct",
                                      "unresolved_overall_pct"]}, indent=1))
print("done ->", "geo_mar_results.json")
