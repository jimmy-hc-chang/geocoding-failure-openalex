#!/usr/bin/env python3
"""
18_minor_revision_checks.py

Reproduces the second-revision analyses (JoI minor-revision round):
  (B-i)   Propensity-model calibration by decile
  (B-ii)  Richer propensity specifications (region x entity, region x comma)
  (B-iii) Trimming-threshold sweep (95/97.5/99/untrimmed) + stabilized weights
  (C)     Mechanism decomposition: structure-side vs place-level failures
  (5a)    Reverse-geocode reclamation of catalogue-country-missing strings
  (5b)    Augmented-sample corrected estimate, 2,000-replicate cluster bootstrap

Run from data/full_run/ after 16_geocoding_mar_analysis.py and
17_reviewer_diagnostics.py. Seed 20260811. Outputs: minor_revision_checks.json
"""
import itertools
import json

import numpy as np
import pandas as pd
import reverse_geocoder as rg
import statsmodels.formula.api as smf

np.random.seed(20260811)
LA = "Latin America & Caribbean"
out = {}

REGION = {}
for cc in "US CA".split(): REGION[cc] = "Northern America"
for cc in "MX GT HN SV NI CR PA CU DO HT JM TT BB BS PR BR AR CL CO PE EC VE UY PY BO GY SR".split(): REGION[cc] = LA
for cc in "DE FR IT GB ES PT NL BE CH AT DK SE NO FI IS IE LU MT CY GR".split(): REGION[cc] = "Western Europe"
for cc in "PL CZ SK HU RO BG HR SI EE LV LT RS ME MK BA AL UA BY MD RU KZ UZ TJ TM KG AZ AM GE XK".split(): REGION[cc] = "Eastern Europe & Central Asia"
for cc in "CN JP KR TW HK MO MN KP".split(): REGION[cc] = "East Asia"
for cc in "IN PK BD LK NP AF BT MV".split(): REGION[cc] = "South Asia"
for cc in "SG MY TH VN ID PH BN KH LA MM TL".split(): REGION[cc] = "Southeast Asia"
for cc in "IR IQ SA AE QA KW OM BH YE JO LB SY IL PS TR".split(): REGION[cc] = "Middle East"
for cc in "EG DZ TN MA LY SD MR".split(): REGION[cc] = "North Africa"
for cc in "NG ZA KE GH TZ MZ BW ZW CM SN BJ BF TD SS ET MU UG RW ZM MW AO CD CG GA CI ML NE SO SZ TG MG".split(): REGION[cc] = "Sub-Saharan Africa"
for cc in "AU NZ FJ PG".split(): REGION[cc] = "Oceania"

# ---------------------------------------------------------------- base frame
af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Raw_Affiliation", "Institution_Countries"])
af = af[af.Raw_Affiliation.notna()]
cc = af[["Raw_Affiliation", "Institution_Countries"]].dropna()
cc = cc.assign(country=cc.Institution_Countries.str.split(",")).explode("country")
cc["country"] = cc.country.str.strip()
vc = (cc.groupby(["Raw_Affiliation", "country"]).size().reset_index(name="n")
        .sort_values("n", ascending=False).drop_duplicates("Raw_Affiliation"))
country = vc.set_index("Raw_Affiliation")["country"]
n_papers = af.groupby("Raw_Affiliation")["Paper_ID"].nunique()
geo = pd.read_csv("geocoded.csv")
m = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True, how="left")
m = m.merge(n_papers.rename("n_papers"), left_on="Raw_Affiliation", right_index=True, how="left")
m["success"] = m.Latitude.notna().astype(int)
m["log_papers"] = np.log(m.n_papers)
m["n_commas"] = m.Raw_Affiliation.str.count(",")
m["strlen"] = m.Raw_Affiliation.str.len()
m["html_ent"] = m.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int)
m["region"] = m.country.map(REGION)
d = m[m.region.notna()].copy()

pairs_af = af[["Paper_ID", "Raw_Affiliation"]].drop_duplicates()

def la_pair_share(dd, wcol):
    sub = pairs_af.merge(dd.loc[dd.success == 1, ["Raw_Affiliation", "region", wcol]], on="Raw_Affiliation")
    sizes = sub.groupby("Paper_ID").size()
    keep = sizes[(sizes >= 2) & (sizes <= 60)].index
    tot = la = 0.0
    for pid, grp in sub[sub.Paper_ID.isin(keep)].groupby("Paper_ID"):
        arr = grp[["region", wcol]].values
        for a, b in itertools.combinations(arr, 2):
            w = a[1] * b[1]; tot += w
            if LA in (a[0], b[0]): la += w
    return round(la / tot * 100, 3)

MAIN = "success ~ C(region) + log_papers + html_ent + n_commas + strlen"

# ---------------------------------------------------------------- B(i) calibration
p0 = smf.logit(MAIN, d).fit(disp=0)
d["phat"] = p0.predict(d).clip(1e-3)
d["dec"] = pd.qcut(d.phat, 10, labels=False, duplicates="drop")
cal = d.groupby("dec").agg(pred=("phat", "mean"), obs=("success", "mean"))
out["calibration_max_abs_gap"] = round((cal.pred - cal.obs).abs().max(), 4)

# ---------------------------------------------------------------- B(ii) richer specs
for name, f in {
        "main_effects": MAIN,
        "region_x_ent": "success ~ C(region)*html_ent + log_papers + n_commas + strlen",
        "region_x_ent_commas": "success ~ C(region)*html_ent + C(region)*n_commas + log_papers + strlen"}.items():
    ph = smf.logit(f, d).fit(disp=0, maxiter=200).predict(d).clip(1e-3)
    w = np.where(d.success == 1, 1 / ph, 0)
    d["w_spec"] = np.clip(w, None, np.quantile(w[d.success == 1], 0.99))
    out[f"LA_share_{name}"] = la_pair_share(d, "w_spec")

# ---------------------------------------------------------------- B(iii) trimming sweep
raw_w = np.where(d.success == 1, 1 / d.phat, 0)
for q, tag in [(0.95, "t95"), (0.975, "t975"), (0.99, "t99"), (None, "untrimmed")]:
    w = raw_w.copy()
    if q:
        w = np.clip(w, None, np.quantile(w[d.success == 1], q))
    d["w_sw"] = w
    out[f"LA_share_{tag}"] = la_pair_share(d, "w_sw")
d["w_stab"] = np.where(d.success == 1, d.success.mean() / d.phat, 0)
d["w_stab"] = np.clip(d.w_stab, None, np.quantile(d.w_stab[d.success == 1], 0.99))
out["LA_share_stabilized"] = la_pair_share(d, "w_stab")

# ---------------------------------------------------------------- C decomposition
fails = m[m.success == 0]
out["fail_le1_comma_pct"] = round((fails.n_commas <= 1).mean() * 100, 1)   # no place-level fallback existed
out["fail_ge2_comma_pct"] = round((fails.n_commas >= 2).mean() * 100, 1)   # place-level tail query failed too
laf = fails[fails.region == LA]
out["LA_fail_ge2_comma_pct"] = round((laf.n_commas >= 2).mean() * 100, 1)

# ---------------------------------------------------------------- 5a reclaim + 5b augmented bootstrap
unk = m[m.country.isna()]
unk_ok = unk[unk.success == 1].copy()
res = rg.search(list(zip(unk_ok.Latitude, unk_ok.Longitude)), mode=1)
unk_ok["region"] = pd.Series([r["cc"] for r in res], index=unk_ok.index).map(REGION)
out["unk_reclaimed_n"] = int(unk_ok.region.notna().sum())
out["unk_doublemissing_n"] = int((unk.success == 0).sum())

aug = pd.concat([d[["Raw_Affiliation", "success", "region", "log_papers", "html_ent", "n_commas", "strlen"]],
                 unk_ok.loc[unk_ok.region.notna(),
                            ["Raw_Affiliation", "success", "region", "log_papers", "html_ent", "n_commas", "strlen"]]])
p = smf.logit(MAIN, aug).fit(disp=0)
aug["ph"] = p.predict(aug).clip(1e-3)
aug["w"] = np.where(aug.success == 1, 1 / aug.ph, 0)
aug["w"] = aug.w.clip(upper=np.quantile(aug.loc[aug.success == 1, "w"], 0.99))

sub = pairs_af.merge(aug.loc[aug.success == 1, ["Raw_Affiliation", "region", "w"]], on="Raw_Affiliation")
sizes = sub.groupby("Paper_ID").size()
keep = sizes[(sizes >= 2) & (sizes <= 60)].index
rows = []
for pid, grp in sub[sub.Paper_ID.isin(keep)].groupby("Paper_ID"):
    arr = grp[["region", "w"]].values
    for a, b in itertools.combinations(arr, 2):
        rows.append((pid, 1 if LA in (a[0], b[0]) else 0, a[1] * b[1]))
Q = pd.DataFrame(rows, columns=["pid", "la", "w"])
out["LA_share_raw_augmented"] = round(Q.la.mean() * 100, 3)
out["LA_share_ipw_augmented"] = round(np.average(Q.la, weights=Q.w) * 100, 3)
out["pairs_augmented"] = len(Q)

papers = Q.pid.unique(); groups = Q.groupby("pid").indices
la_arr, w_arr = Q.la.values, Q.w.values
S = []
for _ in range(2000):
    samp = np.random.choice(papers, size=len(papers), replace=True)
    idx = np.concatenate([groups[p] for p in samp])
    l, w = la_arr[idx], w_arr[idx]
    S.append([l.mean() * 100, np.average(l, weights=w) * 100])
S = np.array(S)
rel = (S[:, 1] - S[:, 0]) / S[:, 1] * 100
out["boot2000_understatement_CI"] = [round(np.percentile(rel, 2.5), 1), round(np.percentile(rel, 97.5), 1)]
out["boot2000_diff_CI"] = [round(np.percentile(S[:, 1] - S[:, 0], 2.5), 3),
                           round(np.percentile(S[:, 1] - S[:, 0], 97.5), 3)]

json.dump(out, open("minor_revision_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
