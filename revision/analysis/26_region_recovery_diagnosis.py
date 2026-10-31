#!/usr/bin/env python3
"""
26_region_recovery_diagnosis.py  (revision, follow-up to Reviewer 2 Major 1)

Why does IPW recover the full-data pair share for Latin America but not for the Middle East?
Four candidate explanations, each tested for every region:

  H1 string-level mis-calibration   the propensity model mis-predicts success for the region's strings
                                     (observed vs mean predicted success; string-level composition recovery)
  H2 whole-work loss                 the region's missing pairs sit in works where fewer than two strings
                                     were geocoded, so no observed pair exists to up-weight
  H3 within-work dependence          joint failure of the region's endpoints is more frequent than the
                                     product of marginal propensities (both-fail obs/exp)
  H4 compositional denominator       a pair SHARE moves only if the region's weighted pair mass grows faster
                                     than everyone else's; if other regions are up-weighted as much, the share
                                     does not move even when the region's own mass is recovered

Decomposition used for H2/H4, per region r (same pair construction as script 21, size filter on all strings):
  full_r      pairs involving r among all strings with a region
  obs_r       pairs involving r with both endpoints geocoded
  ipw_r       IPW mass of obs_r (sum of w_a * w_b)
  numerator recovery   (ipw_r - obs_r) / (full_r - obs_r)
  denominator ratio    ipw_total / full_total, region ratio ipw_r / full_r
  share recovery       as in script 21

Run from full_run/:  TIE_RULE=exclude python ../analysis/26_region_recovery_diagnosis.py
Output: region_recovery_diagnosis.json, region_recovery_diagnosis.csv
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_rule import assign_country, TIE_RULE, REGION  # noqa: E402

af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Raw_Affiliation", "Institution_Countries"])
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
prop = smf.logit("success ~ C(region) + log_papers + html_ent + n_commas + strlen", d).fit(disp=0)
d["ph"] = prop.predict(d)
w = np.where(d.success == 1, 1 / d.ph.clip(1e-3), 0.0)
d["w"] = np.clip(w, None, np.quantile(w[d.success == 1], 0.99))
REGIONS = sorted(d.region.unique())
out = {"tie_rule": TIE_RULE}

# ---------------------------------------------------------------- H1 string level
h1 = {}
tot_w = d.w.sum()
for r in REGIONS:
    g = d[d.region == r]
    h1[r] = {"n": int(len(g)), "fail_pct": round(float(1 - g.success.mean()) * 100, 1),
             "obs_success": round(float(g.success.mean()), 4), "mean_pred_success": round(float(g.ph.mean()), 4),
             "string_share_full_pct": round(float(len(g) / len(d) * 100), 3),
             "string_share_obs_pct": round(float(g.success.sum() / d.success.sum() * 100), 3),
             "string_share_ipw_pct": round(float(g.w.sum() / tot_w * 100), 3)}
    gap = h1[r]["string_share_full_pct"] - h1[r]["string_share_obs_pct"]
    h1[r]["string_level_recovery"] = round(float((h1[r]["string_share_ipw_pct"] - h1[r]["string_share_obs_pct"]) / gap), 3) if abs(gap) > 1e-6 else None
# calibration within region by decile of phat
cal = {}
for r in REGIONS:
    g = d[d.region == r].copy()
    if len(g) < 200:
        continue
    g["dec"] = pd.qcut(g.ph, 5, labels=False, duplicates="drop")
    c = g.groupby("dec").agg(pred=("ph", "mean"), obs=("success", "mean"))
    cal[r] = round(float((c.pred - c.obs).abs().max()), 4)
out["H1_string_level"] = h1
out["H1_max_abs_calibration_gap_by_quintile"] = cal

# ---------------------------------------------------------------- pairs
mem = af[["Paper_ID", "Raw_Affiliation"]].drop_duplicates().merge(
    d[["Raw_Affiliation"]].reset_index().rename(columns={"index": "i"}), on="Raw_Affiliation")
succ = d.success.values
reg = d.region.values
cty = d.country.values
wv = d.w.values
ph = d.ph.values
rows = []
for pid, idx in mem.groupby("Paper_ID").i:
    idx = idx.values
    n = len(idx)
    if n < 2 or n > 60:
        continue
    ns = int(succ[idx].sum())
    for a, b in itertools.combinations(idx, 2):
        rows.append((pid, a, b, ns))
P = pd.DataFrame(rows, columns=["pid", "a", "b", "n_succ_work"])
a, b = P.a.values, P.b.values
P["both_ok"] = succ[a] & succ[b]
P["both_fail"] = (1 - succ[a]) & (1 - succ[b])
P["wmass"] = np.where(P.both_ok == 1, wv[a] * wv[b], 0.0)
P["exp_ok"] = ph[a] * ph[b]
P["exp_fail"] = (1 - ph[a]) * (1 - ph[b])
P["work_lost"] = P.n_succ_work < 2
full_total = len(P)
ipw_total = P.wmass.sum()
obs_total = int(P.both_ok.sum())

rows2 = []
for r in REGIONS:
    inv = (reg[a] == r) | (reg[b] == r)
    S = P[inv]
    full_r = len(S)
    obs_r = int(S.both_ok.sum())
    ipw_r = float(S.wmass.sum())
    missing = S[S.both_ok == 0]
    dom = (reg[S.a.values] == r) & (reg[S.b.values] == r)
    row = {"region": r, "full_pairs": full_r, "obs_pairs": obs_r, "ipw_mass": round(ipw_r, 1),
           "share_full": full_r / full_total * 100, "share_obs": obs_r / obs_total * 100, "share_ipw": ipw_r / ipw_total * 100,
           # H4: own-mass recovery vs share recovery
           "numerator_recovery": (ipw_r - obs_r) / (full_r - obs_r) if full_r > obs_r else np.nan,
           "region_mass_ratio_ipw_over_full": ipw_r / full_r,
           "total_mass_ratio_ipw_over_full": ipw_total / full_total,
           # H2: where the missing pairs are
           "missing_pairs": int(len(missing)),
           "missing_in_works_with_lt2_geocoded_pct": float(missing.work_lost.mean() * 100) if len(missing) else np.nan,
           "missing_both_endpoints_failed_pct": float(missing.both_fail.mean() * 100) if len(missing) else np.nan,
           # H3: dependence
           "both_fail_obs_over_exp": float(S.both_fail.sum() / S.exp_fail.sum()),
           "both_ok_obs_over_exp": float(S.both_ok.sum() / S.exp_ok.sum()),
           # composition of the region's pairs
           "within_region_pair_pct": float(dom.mean() * 100)}
    row["share_recovery"] = (row["share_ipw"] - row["share_obs"]) / (row["share_full"] - row["share_obs"]) \
        if abs(row["share_full"] - row["share_obs"]) > 1e-9 else np.nan
    rows2.append(row)
D = pd.DataFrame(rows2)
D.round(4).to_csv("region_recovery_diagnosis.csv", index=False)
out["pair_level"] = json.loads(D.round(4).to_json(orient="records"))
out["totals"] = {"full_pairs": full_total, "obs_pairs": obs_total, "ipw_mass": round(float(ipw_total), 1),
                 "total_mass_ratio_ipw_over_full": round(float(ipw_total / full_total), 4),
                 "obs_over_full": round(obs_total / full_total, 4)}

# ---------------------------------------------------------------- H5 pair-degree heterogeneity and H6 trimming
# String weights are calibrated to recover the NUMBER OF STRINGS. A pair share also depends on how many pairs
# each string enters. If failed strings enter more pairs than successful strings with the same propensity,
# up-weighting the successful ones recovers the strings but not their pairs.
deg = np.bincount(np.concatenate([a, b]), minlength=len(d))
d["deg"] = deg
d["w_untrim"] = np.where(d.success == 1, 1 / d.ph.clip(1e-3), 0.0)
h5 = {}
for r in REGIONS:
    g = d[d.region == r]
    f_, s_ = g[g.success == 0], g[g.success == 1]
    h5[r] = {"mean_pairs_per_failed_string": round(float(f_.deg.mean()), 2),
             "mean_pairs_per_successful_string": round(float(s_.deg.mean()), 2),
             "degree_ratio_failed_over_success": round(float(f_.deg.mean() / s_.deg.mean()), 3),
             # what string-weights promise vs what pairs need
             "weighted_success_degree_over_full_degree": round(float((s_.w * s_.deg).sum() / g.deg.sum()), 3),
             "trimmed_weight_mass_lost_pct": round(float((s_.w_untrim.sum() - s_.w.sum()) / s_.w_untrim.sum() * 100), 2)}
out["H5_pair_degree"] = h5
# within-region: does failure concentrate in high-degree strings beyond what the propensity predicts?
h5b = {}
for r in REGIONS:
    g = d[(d.region == r) & (d.deg > 0)].assign(ldeg=lambda z: np.log(z.deg))
    if g.success.nunique() < 2 or len(g) < 200:
        continue
    rr = smf.logit("success ~ ldeg + np.log(ph / (1 - ph))", g.assign(ph=g.ph.clip(1e-4, 1 - 1e-4))).fit(disp=0)
    h5b[r] = {"OR_log_degree_given_propensity": round(float(np.exp(rr.params["ldeg"])), 3),
              "p": round(float(rr.pvalues["ldeg"]), 4)}
out["H5_degree_beyond_propensity"] = h5b

# ---------------------------------------------------------------- Middle East by country and by pair type
ME = "Middle East"
inv = (reg[a] == ME) | (reg[b] == ME)
S = P[inv].copy()
S["me_cc"] = np.where(reg[S.a.values] == ME, cty[S.a.values], cty[S.b.values])
S["type"] = np.where((reg[S.a.values] == ME) & (reg[S.b.values] == ME), "within ME", "ME with other region")
bc = S.groupby("me_cc").agg(full=("both_ok", "size"), obs=("both_ok", "sum"), ipw=("wmass", "sum"),
                            work_lost=("work_lost", "mean"))
bc["obs_over_full"] = bc.obs / bc.full
bc["ipw_over_full"] = bc.ipw / bc.full
bc = bc.sort_values("full", ascending=False).head(10)
out["ME_by_country"] = json.loads(bc.round(4).to_json(orient="index"))
bt = S.groupby("type").agg(full=("both_ok", "size"), obs=("both_ok", "sum"), ipw=("wmass", "sum"),
                           work_lost=("work_lost", "mean"))
bt["obs_over_full"] = bt.obs / bt.full
bt["ipw_over_full"] = bt.ipw / bt.full
out["ME_by_pair_type"] = json.loads(bt.round(4).to_json(orient="index"))
# ME string-level failure by country (for context)
mc = d[d.region == ME].groupby("country").agg(n=("success", "size"), fail=("success", lambda z: 1 - z.mean()),
                                             pred_fail=("ph", lambda z: 1 - z.mean()))
out["ME_country_string_level"] = json.loads(mc.sort_values("n", ascending=False).head(10).round(4).to_json(orient="index"))

json.dump(out, open("region_recovery_diagnosis.json", "w"), indent=1, default=float)
cols = ["region", "share_obs", "share_ipw", "share_full", "share_recovery", "numerator_recovery",
        "region_mass_ratio_ipw_over_full", "missing_in_works_with_lt2_geocoded_pct", "both_fail_obs_over_exp",
        "within_region_pair_pct"]
print(D[cols].round(3).to_string(index=False))
print("\ntotal mass ratio ipw/full:", round(ipw_total / full_total, 4), " obs/full:", round(obs_total / full_total, 4))
print("\nH1 string level:", {r: (v["fail_pct"], v["obs_success"], v["mean_pred_success"], v["string_level_recovery"]) for r, v in h1.items()})
print("calibration gap:", cal)
print("\nME by pair type:\n", bt.round(3).to_string())
print("\nME by country:\n", bc.round(3).to_string())
print("\nME strings by country:\n", mc.sort_values("n", ascending=False).head(8).round(3).to_string())
print("\nH5 pair degree:")
for r, v in h5.items(): print(f"  {r:32} fail/success degree {v['degree_ratio_failed_over_success']:.2f}  weighted/full degree {v['weighted_success_degree_over_full_degree']:.3f}  trimmed {v['trimmed_weight_mass_lost_pct']}%  ", h5b.get(r))


# ================================================================ follow-up tests (added 2026-10-22)
# T1 richer propensity on observables; T2 hub strings; T3 collaboration degree (covariate / estimand-weighted fit);
# T4 degree-weighted calibration (pair-weighted observed vs predicted success).
import statsmodels.api as sm  # noqa: E402

A_, B_ = a, b
both_ = (succ[A_] & succ[B_]).astype(bool)
full_s = {r: ((reg[A_] == r) | (reg[B_] == r)).mean() * 100 for r in REGIONS}
obs_s = {r: ((reg[A_] == r) | (reg[B_] == r))[both_].mean() * 100 for r in REGIONS}


def recovery(ph_, mask=None):
    w_ = np.where(succ == 1, 1 / np.clip(ph_, 1e-3, None), 0.0)
    w_ = np.clip(w_, None, np.quantile(w_[succ == 1], 0.99))
    A2, B2, bo2 = (A_, B_, both_) if mask is None else (A_[mask], B_[mask], both_[mask])
    pw_ = np.where(bo2, w_[A2] * w_[B2], 0.0)
    res_, resid_ = {}, {}
    for r in REGIONS:
        inv_ = (reg[A2] == r) | (reg[B2] == r)
        f_, o_ = inv_.mean() * 100, inv_[bo2].mean() * 100
        s_ = (inv_ * pw_).sum() / pw_.sum() * 100
        res_[r] = round(float((s_ - o_) / (f_ - o_)), 3) if abs(f_ - o_) > 1e-9 else None
        resid_[r] = round(float(s_ - f_), 3)
    return res_, resid_


d["ldeg"] = np.log1p(d.deg)
d["cc5"] = pd.cut(d.n_commas, [-1, 0, 1, 2, 3, 999], labels=["c0", "c1", "c2", "c3", "c4"]).astype(str)
F0 = "success ~ C(region) + log_papers + html_ent + n_commas + strlen"
specs = {"main": F0,
         "categorical_comma": "success ~ C(region) + log_papers + html_ent + C(cc5) + strlen",
         "region_x_cat_comma": "success ~ C(region)*C(cc5) + log_papers + html_ent + strlen",
         "region_x_cat_comma_x_html": "success ~ C(region)*C(cc5) + C(region)*html_ent + log_papers + strlen",
         "region_x_log_degree": F0 + " + C(region):ldeg"}
T1 = {}
for nm, f in specs.items():
    fit_ = smf.logit(f, d).fit(disp=0, maxiter=300)
    rec_, resid_ = recovery(fit_.predict(d).values)
    T1[nm] = {"aic": round(float(fit_.aic), 1), "recovery": rec_, "residual_pp": resid_}
dd = d[d.deg > 0]
fw_ = smf.glm(F0, dd, family=sm.families.Binomial(), freq_weights=dd.deg).fit()
rec_, resid_ = recovery(fw_.predict(d).values)
T1["degree_weighted_fit"] = {"recovery": rec_, "residual_pp": resid_}
out["T1_propensity_variants"] = T1

# T2 hub strings: share of missing pairs from the top-5 failed strings; recovery after removing them
T2 = {}
base_rec, _ = recovery(d.ph.values)
for r in REGIONS:
    inv_ = (reg[A_] == r) | (reg[B_] == r)
    miss_ = inv_ & ~both_
    f_idx = np.where((reg == r) & (succ == 0))[0]
    cnt = pd.Series(np.concatenate([A_[miss_], B_[miss_]])).value_counts()
    cnt = cnt[cnt.index.isin(f_idx)]
    top5 = list(cnt.head(5).index)
    keep = ~(np.isin(A_, top5) | np.isin(B_, top5))
    rr_, _ = recovery(d.ph.values, keep)
    T2[r] = {"top5_failed_share_of_missing_pct": round(float(cnt.head(5).sum() / miss_.sum() * 100), 1),
             "top5_strings": [d.Raw_Affiliation.iloc[i][:90] for i in top5],
             "recovery": base_rec[r], "recovery_without_top5": rr_[r]}
out["T2_hubs"] = T2

# T4 pair-weighted calibration
T4 = {}
for r, g in d.groupby("region"):
    T4[r] = {"string_obs": round(float(g.success.mean()), 4), "string_pred": round(float(g.ph.mean()), 4),
             "pair_weighted_obs": round(float((g.deg * g.success).sum() / g.deg.sum()), 4),
             "pair_weighted_pred": round(float((g.deg * g.ph).sum() / g.deg.sum()), 4)}
    T4[r]["pair_weighted_gap_pp"] = round((T4[r]["pair_weighted_obs"] - T4[r]["pair_weighted_pred"]) * 100, 2)
me_c = {}
for c, g in d[d.region == "Middle East"].groupby("country"):
    if g.deg.sum() >= 300:
        me_c[c] = {"pair_weighted_obs": round(float((g.deg * g.success).sum() / g.deg.sum()), 3),
                   "pair_weighted_pred": round(float((g.deg * g.ph).sum() / g.deg.sum()), 3)}
out["T4_pair_weighted_calibration"] = T4
out["T4_ME_by_country"] = me_c

json.dump(out, open("region_recovery_diagnosis.json", "w"), indent=1, default=float)
print("\nT1 recovery by propensity variant (ME / LA / WE / EA / NAm):")
for nm, v in T1.items():
    rr = v["recovery"]
    print(f"  {nm:28} ME {rr['Middle East']}  LA {rr['Latin America & Caribbean']}  WE {rr['Western Europe']}  EA {rr['East Asia']}  NAm {rr['Northern America']}")
print("T2 ME hubs:", T2["Middle East"])
print("T4 pair-weighted gap pp:", {r: v["pair_weighted_gap_pp"] for r, v in T4.items()})
