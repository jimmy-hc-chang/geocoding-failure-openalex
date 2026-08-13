#!/usr/bin/env python3
"""
17_reviewer_diagnostics.py

Reproduces the revision-round analyses added in response to peer review:
  (1) CAR-T second-field mechanism replication           -> cart_replication.json
  (2) IPW weight diagnostics (ESS, positivity, trimming) -> reviewer_diagnostics.json
  (3) Within-paper selection-dependence (kappa) test     -> reviewer_diagnostics.json
  (4) Cluster-bootstrap CIs for corrected estimates      -> reviewer_diagnostics.json
  (5) Wilson CIs for country failure rates               -> reviewer_diagnostics.json
  (6) Unknown-country bounds + MNAR tipping-point        -> reviewer_diagnostics.json
  (7) India unresolved-string characterization           -> reviewer_diagnostics.json

Run AFTER 16_geocoding_mar_analysis.py, from data/full_run/ (data/cart_run
must be a sibling directory). Bootstrap seed fixed at 20260811.
"""
import itertools
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

SEED = 20260811
B_REPS = 300
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# region map identical to 16_geocoding_mar_analysis.py
from importlib import util as _u
_spec = _u.spec_from_file_location("m16", "16_geocoding_mar_analysis.py")
# NOTE: importing would re-run the module; replicate the REGION constant instead.
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

LA = "Latin America & Caribbean"
out = {}


def build_strings(dirpath):
    af = pd.read_csv(f"{dirpath}/affil_full.csv",
                     usecols=["Paper_ID", "Raw_Affiliation", "Institution_Countries"])
    af = af[af.Raw_Affiliation.notna()]
    cc = af[["Raw_Affiliation", "Institution_Countries"]].dropna()
    cc = cc.assign(country=cc.Institution_Countries.str.split(",")).explode("country")
    cc["country"] = cc.country.str.strip()
    vc = (cc.groupby(["Raw_Affiliation", "country"]).size().reset_index(name="n")
            .sort_values("n", ascending=False).drop_duplicates("Raw_Affiliation"))
    country = vc.set_index("Raw_Affiliation")["country"]
    n_papers = af.groupby("Raw_Affiliation")["Paper_ID"].nunique()
    geo = pd.read_csv(f"{dirpath}/geocoded.csv")
    m = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True, how="left")
    m = m.merge(n_papers.rename("n_papers"), left_on="Raw_Affiliation", right_index=True, how="left")
    m["success"] = m.Latitude.notna().astype(int)
    m["fail"] = 1 - m.success
    m["log_papers"] = np.log(m.n_papers)
    m["n_commas"] = m.Raw_Affiliation.str.count(",")
    m["strlen"] = m.Raw_Affiliation.str.len()
    m["html_ent"] = m.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int)
    m["region"] = m.country.map(REGION)
    return m, af


# ------------------------------------------------ (1) CAR-T replication
cart, _ = build_strings("../cart_run")
dc = cart[cart.region.notna()].copy()
rep = {"n_strings": len(cart), "fail_rate": round(cart.fail.mean(), 4),
       "country_known_failed_pct": round(cart.loc[cart.fail == 1, "country"].notna().mean() * 100, 1)}
low = dc[dc.n_commas <= 1]
rep["strings_0or1_comma_pct"] = round(len(low) / len(dc) * 100, 1)
rep["failures_from_0or1_comma_pct"] = round(low.fail.sum() / dc.fail.sum() * 100, 1)
rep["fail_html_ent"] = round(dc[dc.html_ent == 1].fail.mean() * 100, 1)
rep["fail_no_ent"] = round(dc[dc.html_ent == 0].fail.mean() * 100, 1)
r = smf.logit('fail ~ C(region, Treatment(reference="Northern America")) + log_papers',
              dc).fit(disp=0, cov_type="cluster", cov_kwds={"groups": dc.country})
ci = np.exp(r.conf_int())
rep["region_OR"] = {k.split("[T.")[-1].rstrip("]"): [round(np.exp(r.params[k]), 3),
                                                     round(ci.loc[k, 0], 3), round(ci.loc[k, 1], 3)]
                    for k in r.params.index if "region" in k}
json.dump(rep, open("cart_replication.json", "w"), indent=1)
out["cart"] = rep

# ------------------------------------------------ main corpus
m, af_full = build_strings(".")
d = m[m.region.notna()].copy()
prop = smf.logit("success ~ C(region) + log_papers + html_ent + n_commas + strlen", d).fit(disp=0)
d["phat"] = prop.predict(d)
d["w"] = np.where(d.success == 1, 1 / d.phat, 0.0)
cap = d.loc[d.success == 1, "w"].quantile(0.99)
d["w_t"] = d.w.clip(upper=cap)

# ------------------------------------------------ (2) weight diagnostics
ok = d[d.success == 1]
out["w_mean"] = round(ok.w_t.mean(), 3); out["w_median"] = round(ok.w_t.median(), 3)
out["w_cap"] = round(float(cap), 3); out["w_untrimmed_max"] = round(ok.w.max(), 3)
out["ESS"] = int(ok.w_t.sum() ** 2 / (ok.w_t ** 2).sum()); out["n_success"] = len(ok)
out["min_phat_by_region"] = {r_: [round(v["min"], 3), round(v["mean"], 3)]
                             for r_, v in d.groupby("region").phat.agg(["min", "mean"]).iterrows()}
out["share_phat_below_0.2_pct"] = round((d.phat < 0.2).mean() * 100, 2)

# ------------------------------------------------ (5) Wilson CIs
tbl = d.groupby("country").agg(n=("fail", "size"), f=("fail", "sum"))
wil = {}
for c in ["AR", "NP", "CO", "UZ", "FR", "SG", "US", "JP", "TW", "CN", "MO", "EE"]:
    n, f = int(tbl.loc[c, "n"]), int(tbl.loc[c, "f"])
    lo, hi = proportion_confint(f, n, method="wilson")
    wil[c] = [n, round(f / n * 100, 1), round(lo * 100, 1), round(hi * 100, 1)]
out["wilson"] = wil
out["fold_conservative_AR_over_US"] = round((wil["AR"][2] / 100) / (wil["US"][3] / 100), 1)

# ------------------------------------------------ (6a) unknown-country bounds
unk = m[m.country.isna()]
out["unknown_country"] = {"n": len(unk), "share_pct": round(len(unk) / len(m) * 100, 1),
                          "fail_rate": round(unk.fail.mean(), 4),
                          "known_fail_rate": round(m[m.country.notna()].fail.mean(), 4)}
la_known = (m.country.map(REGION) == LA).sum()
out["LA_share_bounds_pct"] = [round(la_known / len(m) * 100, 2),
                              round((la_known + len(unk)) / len(m) * 100, 2)]

# ------------------------------------------------ pairs, kappa, bootstrap, tipping
pairs_af = af_full[["Paper_ID", "Raw_Affiliation"]].drop_duplicates().merge(
    d[["Raw_Affiliation", "success", "phat", "region", "country", "w_t", "Latitude", "Longitude"]],
    on="Raw_Affiliation")

# (3) kappa: within-work joint-success dependence
obs_ss = exp_ss = obs_ss_la = exp_ss_la = obs_ff = exp_ff = npairs = 0
for pid, grp in pairs_af.groupby("Paper_ID"):
    if not (2 <= len(grp) <= 60):
        continue
    s, p, reg = grp.success.values, grp.phat.values, grp.region.values
    for i, j in itertools.combinations(range(len(s)), 2):
        npairs += 1
        obs_ss += s[i] * s[j]; exp_ss += p[i] * p[j]
        obs_ff += (1 - s[i]) * (1 - s[j]); exp_ff += (1 - p[i]) * (1 - p[j])
        if LA in (reg[i], reg[j]):
            obs_ss_la += s[i] * s[j]; exp_ss_la += p[i] * p[j]
kappa_all, kappa_la = obs_ss / exp_ss, obs_ss_la / exp_ss_la
out["kappa"] = {"bothfail_obs_over_exp": round(obs_ff / exp_ff, 2),
                "bothsuccess_all": round(kappa_all, 4), "bothsuccess_LA": round(kappa_la, 4)}

# successful-endpoint pairs
ok_af = pairs_af[pairs_af.success == 1]
sizes = ok_af.groupby("Paper_ID").size()
keep = sizes[(sizes >= 2) & (sizes <= 60)].index
rows = []
for pid, grp in ok_af[ok_af.Paper_ID.isin(keep)].groupby("Paper_ID"):
    arr = grp[["region", "country", "w_t", "Latitude", "Longitude"]].values
    for a, b in itertools.combinations(arr, 2):
        rows.append((pid, a[0], b[0], a[1], b[1], a[2] * b[2], a[3], a[4], b[3], b[4]))
P = pd.DataFrame(rows, columns=["pid", "r1", "r2", "c1", "c2", "w", "lat1", "lon1", "lat2", "lon2"])

def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

P["dist"] = hav(P.lat1, P.lon1, P.lat2, P.lon2)
P["intl"] = (P.c1 != P.c2).astype(int)
P["la"] = ((P.r1 == LA) | (P.r2 == LA)).astype(int)

# kappa-adjusted LA share
w_adj = P.w / np.where(P.la == 1, kappa_la, kappa_all)
out["LA_pairshare"] = {"raw": round(P.la.mean() * 100, 3),
                       "ipw": round(np.average(P.la, weights=P.w) * 100, 3),
                       "ipw_kappa_adj": round(np.average(P.la, weights=w_adj) * 100, 3)}

# (4) cluster bootstrap by work
papers = P.pid.unique()
groups = P.groupby("pid").indices
stats = []
for _ in range(B_REPS):
    samp = np.random.choice(papers, size=len(papers), replace=True)
    idx = np.concatenate([groups[p] for p in samp])
    sub = P.iloc[idx]
    stats.append([sub.la.mean() * 100, np.average(sub.la, weights=sub.w) * 100,
                  sub.dist.mean(), np.average(sub.dist, weights=sub.w),
                  sub.intl.mean() * 100, np.average(sub.intl, weights=sub.w) * 100])
S = np.array(stats)
for i, nm in enumerate(["LA_raw", "LA_ipw", "dist_raw", "dist_ipw", "intl_raw", "intl_ipw"]):
    out[f"boot_{nm}"] = [round(np.percentile(S[:, i], 2.5), 3), round(np.percentile(S[:, i], 97.5), 3)]
rel = (S[:, 1] - S[:, 0]) / S[:, 1] * 100
out["boot_LA_understatement_pct"] = [round(np.percentile(rel, 2.5), 1), round(np.percentile(rel, 97.5), 1)]

# (6b) MNAR tipping point: attenuate LA excess weights by lambda
tip = {}
for lam in [0.75, 0.5, 0.25, 0.0]:
    adj = d.copy()
    adj["w_l"] = np.where(adj.region == LA, 1 + lam * (adj.w_t - 1), adj.w_t)
    m2 = ok_af[ok_af.Paper_ID.isin(keep)].merge(adj[["Raw_Affiliation", "w_l"]], on="Raw_Affiliation")
    rows2 = []
    for pid, grp in m2.groupby("Paper_ID"):
        arr = grp[["region", "w_l"]].values
        for a, b in itertools.combinations(arr, 2):
            rows2.append((1 if LA in (a[0], b[0]) else 0, a[1] * b[1]))
    Q = pd.DataFrame(rows2, columns=["la", "w"])
    tip[lam] = round(np.average(Q.la, weights=Q.w) * 100, 3)
out["tipping_LA_share_by_lambda"] = tip

# ------------------------------------------------ (7) India unresolved characterization
try:
    ror = pd.read_csv("strings_ror_full.csv")  # produced by earlier audit if present
    ind = ror[(ror.geo_cc == "IN") & (ror.unresolved == 1)]
    out["india_unresolved"] = {
        "n": len(ind),
        "univ_keyword_pct": round(ind.Raw_Affiliation.str.contains(
            r"\b(?:University|Institute|College|IIT|NIT|IIIT|Academy|Laborator)\w*",
            case=False, regex=True).mean() * 100, 1),
        "company_marker_pct": round(ind.Raw_Affiliation.str.contains(
            r"\b(?:Pvt|Ltd|Limited|Inc|Corp|Company|Technologies|Industries|Solutions|Consultancy|Services)\b",
            case=False, regex=True).mean() * 100, 1)}
except FileNotFoundError:
    out["india_unresolved"] = "run entity-resolution audit first (16_geocoding_mar_analysis.py)"

json.dump(out, open("reviewer_diagnostics.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
