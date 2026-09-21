#!/usr/bin/env python3
"""
19_presubmission_pass.py

Country-level specification and the primary bootstrap reported in Table 3:
  (4) Country fixed-effects propensity specification (cells with fewer than
      50 strings pooled to their region), with per-country calibration for
      AR, CO, FR and NP -> corrected Latin-America pair share
  (5) Primary cluster bootstrap at 2,000 replicates with the propensity model
      re-fit inside every replicate (paper-level resampling; strings enter
      each replicate's fit with frequency weights equal to the number of
      sampled works containing them)

Run from full_run/. Seed 20260811. Output: presubmission_pass.json
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import sparse

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

# ---------------------------------------------------------------- frame
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
d = m[m.region.notna()].reset_index(drop=True)

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

# ---------------------------------------------------------------- (4) country-FE propensity
csize = d.country.value_counts()
d["cgrp"] = np.where(d.country.map(csize) >= 50, d.country, "RARE_" + d.region)
pc = smf.logit("success ~ C(cgrp) + log_papers + html_ent + n_commas + strlen", d).fit(disp=0, maxiter=300)
d["ph_c"] = pc.predict(d).clip(1e-3)
w = np.where(d.success == 1, 1 / d.ph_c, 0)
d["w_c"] = np.clip(w, None, np.quantile(w[d.success == 1], 0.99))
out["LA_share_country_FE"] = la_pair_share(d, "w_c")
out["n_country_FE_cells"] = int(d.cgrp.nunique())
for c in ["AR", "CO", "FR", "NP"]:
    s = d[d.country == c]
    out[f"calib_{c}"] = {"n": len(s), "pred_success": round(s.ph_c.mean(), 3),
                         "obs_success": round(s.success.mean(), 3)}

# ---------------------------------------------------------------- (5) 2,000-rep refit bootstrap
MAIN = "success ~ C(region) + log_papers + html_ent + n_commas + strlen"
y, X = patsy.dmatrices(MAIN, d, return_type="dataframe")
y = np.asarray(y).ravel(); X = np.asarray(X)

# membership: paper x string (only strings in d)
sub = pairs_af.merge(d[["Raw_Affiliation"]].reset_index().rename(columns={"index": "sidx"}), on="Raw_Affiliation")
papers_all = sub.Paper_ID.unique()
pidx = {p: i for i, p in enumerate(papers_all)}
A = sparse.csr_matrix((np.ones(len(sub)), (sub.Paper_ID.map(pidx).values, sub.sidx.values)),
                      shape=(len(papers_all), len(d)))

# pair structures (successful endpoints, papers with 2..60 geocoded strings)
ok = d[d.success == 1].reset_index().rename(columns={"index": "didx"})
ok["oidx"] = np.arange(len(ok))
sub_ok = pairs_af.merge(ok[["Raw_Affiliation", "oidx", "didx", "region"]], on="Raw_Affiliation")
sizes = sub_ok.groupby("Paper_ID").size()
keep = set(sizes[(sizes >= 2) & (sizes <= 60)].index)
rows = []
for pid, grp in sub_ok[sub_ok.Paper_ID.isin(keep)].groupby("Paper_ID"):
    arr = grp[["oidx", "region"]].values
    for a, b in itertools.combinations(arr, 2):
        rows.append((pidx[pid], a[0], b[0], 1 if LA in (a[1], b[1]) else 0))
PP = pd.DataFrame(rows, columns=["p", "i1", "i2", "la"])
p_arr, i1, i2, la = PP.p.values, PP.i1.values, PP.i2.values, PP.la.values.astype(float)
ok_didx = ok.didx.values          # map success-string order -> row of d/X
succ_mask = d.success.values == 1

B = 2000
beta0 = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=100).params
stats = np.full((B, 2), np.nan)
t0 = time.time()
for b in range(B):
    samp = np.random.randint(0, len(papers_all), len(papers_all))
    pmult = np.bincount(samp, minlength=len(papers_all)).astype(float)
    fw = np.asarray(A.T @ pmult).ravel()          # per-string frequency weight
    use = fw > 0
    glm = sm.GLM(y[use], X[use], family=sm.families.Binomial(), freq_weights=fw[use])
    try:
        res = glm.fit(maxiter=30, start_params=beta0)
        beta = res.params
    except Exception:
        continue
    ph = 1 / (1 + np.exp(-(X @ beta)))
    ph = np.clip(ph, 1e-3, None)
    wfull = np.where(succ_mask, 1 / ph, 0)
    ws = wfull[ok_didx]
    ws = np.clip(ws, None, np.quantile(ws, 0.99))
    k = pmult[p_arr]
    pw = ws[i1] * ws[i2] * k
    denom_r = k.sum(); denom_w = pw.sum()
    if denom_r == 0 or denom_w == 0:
        continue
    stats[b, 0] = (la * k).sum() / denom_r * 100
    stats[b, 1] = (la * pw).sum() / denom_w * 100
    if b % 200 == 0:
        print(f"rep {b} elapsed {time.time()-t0:.0f}s", flush=True)

S = stats[~np.isnan(stats).any(axis=1)]
out["boot_reps_effective"] = int(len(S))
out["boot2000_refit_LA_raw"] = [round(np.percentile(S[:, 0], 2.5), 3), round(np.percentile(S[:, 0], 97.5), 3)]
out["boot2000_refit_LA_ipw"] = [round(np.percentile(S[:, 1], 2.5), 3), round(np.percentile(S[:, 1], 97.5), 3)]
rel = (S[:, 1] - S[:, 0]) / S[:, 1] * 100
out["boot2000_refit_understatement"] = [round(np.percentile(rel, 2.5), 1), round(np.percentile(rel, 97.5), 1)]
diff = S[:, 1] - S[:, 0]
out["boot2000_refit_diff"] = [round(np.percentile(diff, 2.5), 3), round(np.percentile(diff, 97.5), 3)]
out["elapsed_s"] = round(time.time() - t0, 1)

json.dump(out, open("presubmission_pass.json", "w"), indent=1)
print(json.dumps(out, indent=1))
