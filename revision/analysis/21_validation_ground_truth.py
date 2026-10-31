#!/usr/bin/env python3
"""
21_validation_ground_truth.py  (revision, Reviewer 2 Major 1)

Regional involvement in within-paper co-affiliation pairs is the one downstream
indicator for which a full-data reference exists: the catalogue records region for
strings whose geocode failed, so pairs can be formed among ALL strings with a known
region, not only the geocoded ones. This script compares, for every region:

    observed       complete-case pairs (both endpoints geocoded)
    ipw_region     IPW, region-level propensity model (manuscript main specification)
    ipw_kappa      ipw_region / kappa_r  (within-work dependence of joint success, per region)
    ipw_countryFE  IPW, country fixed-effects propensity (cells < 50 strings pooled to region)
    full           full-data reference (all strings with known region, geocoded or not)

and reports the true understatement (observed vs full), the estimated understatement
(observed vs each IPW variant), and the share of the true gap each variant recovers.

Pair construction follows the manuscript: within each work, all pairs of distinct strings;
works with 2..60 strings in the relevant set. Two filter variants for the full-data
reference are reported (size counted on geocoded strings vs on all strings).

Cluster bootstrap by work (B reps, weights held fixed) gives intervals for observed,
full and the IPW variants.

Run from full_run/:  TIE_RULE=exclude python ../analysis/21_validation_ground_truth.py
Output: validation_ground_truth.json, validation_ground_truth.csv
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_rule import assign_country, TIE_RULE  # noqa: E402

B = int(os.environ.get("B_REPS", 500))
DROP_SAME_INST = os.environ.get("DROP_SAME_INST", "0") == "1"
SEED = 20260811
rng = np.random.default_rng(SEED)

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
REGIONS = sorted(set(REGION.values()))

# ---------------------------------------------------------------- frame (identical to 16)
af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Raw_Affiliation", "Institution_Countries", "Standardized_Institutions"])
af = af[af.Raw_Affiliation.notna()]
country = assign_country(af)
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
_inst = af.dropna(subset=["Standardized_Institutions"]).groupby("Raw_Affiliation").Standardized_Institutions.agg(lambda z: z.value_counts().index[0])
d["inst"] = d.Raw_Affiliation.map(_inst)

MAIN = "success ~ C(region) + log_papers + html_ent + n_commas + strlen"


def trimmed_w(ph, success, q=0.99):
    w = np.where(success == 1, 1 / np.clip(ph, 1e-3, None), 0.0)
    cap = np.quantile(w[success == 1], q)
    return np.clip(w, None, cap)


d["w_reg"] = trimmed_w(smf.logit(MAIN, d).fit(disp=0).predict(d).values, d.success.values)
csize = d.country.value_counts()
d["cgrp"] = np.where(d.country.map(csize) >= 50, d.country, "RARE_" + d.region)
pc = smf.logit("success ~ C(cgrp) + log_papers + html_ent + n_commas + strlen", d).fit(disp=0, maxiter=300)
d["w_cfe"] = trimmed_w(pc.predict(d).values, d.success.values)
d["phat_reg"] = smf.logit(MAIN, d).fit(disp=0).predict(d)

# ---------------------------------------------------------------- pair arrays
mem = af[["Paper_ID", "Raw_Affiliation"]].drop_duplicates().merge(
    d[["Raw_Affiliation"]].reset_index().rename(columns={"index": "sidx"}), on="Raw_Affiliation")
reg_code = {r: i for i, r in enumerate(REGIONS)}
d["rcode"] = d.region.map(reg_code).astype(int)
succ = d.success.values
inst_arr = d.inst.values
rcode = d.rcode.values


def build_pairs(size_on):
    """size_on: 'geocoded' -> 2..60 counted among geocoded strings of the work;
                'all'      -> 2..60 counted among all strings with region."""
    P = []
    for pid, idx in mem.groupby("Paper_ID").sidx:
        idx = idx.values
        n_geo = int(succ[idx].sum())
        n_all = len(idx)
        n = n_geo if size_on == "geocoded" else n_all
        if n < 2 or n > 60 or n_all < 2:
            continue
        for a, b in itertools.combinations(idx, 2):
            if DROP_SAME_INST and isinstance(inst_arr[a], str) and inst_arr[a] == inst_arr[b]:
                continue
            P.append((pid, a, b))
    P = pd.DataFrame(P, columns=["pid", "a", "b"])
    return P


def region_shares(P, wa=None):
    """share of pairs involving each region; wa = per-string weight array or None."""
    a, b = P.a.values, P.b.values
    w = np.ones(len(P)) if wa is None else wa[a] * wa[b]
    out = {}
    for r, k in reg_code.items():
        inv = ((rcode[a] == k) | (rcode[b] == k)).astype(float)
        out[r] = float((inv * w).sum() / w.sum() * 100)
    return out


res = {"tie_rule": TIE_RULE, "B": B, "drop_same_institution_pairs": DROP_SAME_INST, "n_strings_with_region": int(len(d))}
for size_on in ["geocoded", "all"]:
    Pall = build_pairs(size_on)
    both = succ[Pall.a.values] & succ[Pall.b.values]
    Pobs = Pall[both == 1].reset_index(drop=True)
    # when size_on == geocoded, observed pairs coincide with manuscript pairs
    full = region_shares(Pall)
    obs = region_shares(Pobs)
    ipw_r = region_shares(Pobs, d.w_reg.values)
    ipw_c = region_shares(Pobs, d.w_cfe.values)
    # kappa per region: observed joint success / expected (product of phat), pairs involving region
    ph = d.phat_reg.values
    a, b = Pall.a.values, Pall.b.values
    kap = {}
    for r, k in reg_code.items():
        inv = (rcode[a] == k) | (rcode[b] == k)
        kap[r] = float(both[inv].sum() / (ph[a][inv] * ph[b][inv]).sum()) if inv.any() else np.nan
    kall = float(both.sum() / (ph[a] * ph[b]).sum())
    ipw_k = {}
    for r, k in reg_code.items():
        # manuscript's kappa adjustment: divide weight of pairs involving r by kappa_r, others by kappa_all
        inv = ((rcode[Pobs.a.values] == k) | (rcode[Pobs.b.values] == k))
        w = d.w_reg.values[Pobs.a.values] * d.w_reg.values[Pobs.b.values]
        wadj = w / np.where(inv, kap[r], kall)
        ipw_k[r] = float((inv * wadj).sum() / wadj.sum() * 100)

    # bootstrap by work
    pids = Pall.pid.unique()
    gi_all = Pall.groupby("pid").indices
    obs_mask = both == 1
    S = {k: [] for k in ["obs", "full", "ipw_r", "ipw_c"]}
    wr = d.w_reg.values[a] * d.w_reg.values[b]
    wc = d.w_cfe.values[a] * d.w_cfe.values[b]
    INV = np.stack([((rcode[a] == k) | (rcode[b] == k)) for k in range(len(REGIONS))], axis=1).astype(float)
    for _ in range(B):
        s = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([gi_all[p] for p in s])
        I = INV[idx]; om = obs_mask[idx]
        S["full"].append(I.mean(0) * 100)
        S["obs"].append(I[om].mean(0) * 100)
        S["ipw_r"].append((I[om] * wr[idx][om, None]).sum(0) / wr[idx][om].sum() * 100)
        S["ipw_c"].append((I[om] * wc[idx][om, None]).sum(0) / wc[idx][om].sum() * 100)
    S = {k: np.array(v) for k, v in S.items()}

    rows = []
    for r, k in reg_code.items():
        true_gap = full[r] - obs[r]
        row = {"region": r, "observed": obs[r], "ipw_region": ipw_r[r], "ipw_kappa": ipw_k[r],
               "ipw_countryFE": ipw_c[r], "full_data": full[r],
               "true_understatement_pct": (obs[r] - full[r]) / full[r] * 100,
               "est_understatement_region_pct": (obs[r] - ipw_r[r]) / ipw_r[r] * 100,
               "est_understatement_kappa_pct": (obs[r] - ipw_k[r]) / ipw_k[r] * 100,
               "est_understatement_cfe_pct": (obs[r] - ipw_c[r]) / ipw_c[r] * 100,
               "recovered_region": (ipw_r[r] - obs[r]) / true_gap if abs(true_gap) > 1e-9 else np.nan,
               "recovered_kappa": (ipw_k[r] - obs[r]) / true_gap if abs(true_gap) > 1e-9 else np.nan,
               "recovered_cfe": (ipw_c[r] - obs[r]) / true_gap if abs(true_gap) > 1e-9 else np.nan,
               "kappa_r": kap[r],
               "ci_observed": list(np.percentile(S["obs"][:, k], [2.5, 97.5])),
               "ci_full": list(np.percentile(S["full"][:, k], [2.5, 97.5])),
               "ci_ipw_region": list(np.percentile(S["ipw_r"][:, k], [2.5, 97.5])),
               "ci_ipw_cfe": list(np.percentile(S["ipw_c"][:, k], [2.5, 97.5])),
               "ci_true_gap_pp": list(np.percentile(S["full"][:, k] - S["obs"][:, k], [2.5, 97.5])),
               "ci_ipw_region_minus_full_pp": list(np.percentile(S["ipw_r"][:, k] - S["full"][:, k], [2.5, 97.5])),
               "ci_ipw_cfe_minus_full_pp": list(np.percentile(S["ipw_c"][:, k] - S["full"][:, k], [2.5, 97.5]))}
        rows.append(row)
    df = pd.DataFrame(rows)
    res[f"size_on_{size_on}"] = {"n_pairs_full": int(len(Pall)), "n_pairs_observed": int(len(Pobs)),
                                 "n_works": int(Pall.pid.nunique()), "kappa_all": kall,
                                 "regions": json.loads(df.to_json(orient="records"))}
    df.round(4).to_csv(f"validation_ground_truth_{size_on}" + ("_nosameinst" if DROP_SAME_INST else "") + ".csv", index=False)
    print(f"\n=== size filter on {size_on} strings: {len(Pall):,} full pairs, {len(Pobs):,} observed ===")
    show = df[["region", "observed", "ipw_region", "ipw_kappa", "ipw_countryFE", "full_data",
               "true_understatement_pct", "recovered_region", "recovered_kappa", "recovered_cfe"]].round(3)
    print(show.to_string(index=False))

json.dump(res, open("validation_ground_truth" + ("_nosameinst" if DROP_SAME_INST else "") + ".json", "w"), indent=1, default=float)
print("\nwritten validation_ground_truth.json")
