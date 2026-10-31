#!/usr/bin/env python3
"""
27_estimand_weighted_propensity.py — sensitivity of regional pair-share recovery to how the
propensity model is fitted: string-weighted (SW, as in the manuscript) versus weighted by the
number of pairs each string enters (EW), and to the trimming cap (Supplementary Section S4.2).
The EW variant is not used in the manuscript.

V1  Point estimates, both fits, all regions (recovery = (ipw - obs) / (full - obs); residual = ipw - full, pp).
V2  Mechanics: with region fixed effects, the EW fit's score equations force
        sum_{i in r} deg_i * phat_i = sum_{i in r} deg_i * success_i     for every region r
    i.e. pair-weighted calibration is exact by construction. Verified numerically.
V3  Alternative unit weights: n_papers (appearances) instead of pair degree; sqrt(degree).
V4  Sensitivity: trimming cap (95/99/untrimmed); pair size filter on all vs geocoded strings.
V5  Does EW also move the DISTANCE indicators toward the near-complete benchmark of script 25?
    (needs institution_coordinates inputs; uses the same near-complete construction)
V6  Cluster bootstrap by work with BOTH propensity models re-fitted in each replicate:
    intervals for residuals under SW and EW and for |res_EW| - |res_SW|.
    Run in chunks: B_REPS, B_SEED, B_DUMP (npz); combine with --combine.

Run from full_run/:  TIE_RULE=exclude python ../analysis/27_estimand_weighted_propensity.py
         bootstrap:  TIE_RULE=exclude B_REPS=60 B_SEED=1 B_DUMP=ew_boot_1.npz python ... --boot
         combine:    python ../analysis/27_estimand_weighted_propensity.py --combine ew_boot_*.npz
Output: estimand_weighted_propensity.json, ew_bootstrap.json
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from country_rule import assign_country, TIE_RULE, REGION  # noqa: E402

MODE = "boot" if "--boot" in sys.argv else "combine" if "--combine" in sys.argv else "main"

if MODE == "combine":
    files = [f for a in sys.argv[sys.argv.index("--combine") + 1:] for f in glob.glob(a)]
    Z = [np.load(f, allow_pickle=True) for f in files]
    regions = list(Z[0]["regions"])
    sw = np.concatenate([z["res_sw"] for z in Z])
    ew = np.concatenate([z["res_ew"] for z in Z])
    out = {"n_reps": int(len(sw)), "files": files, "regions": {}}
    for j, r in enumerate(regions):
        dif = np.abs(ew[:, j]) - np.abs(sw[:, j])
        out["regions"][r] = {"res_sw_ci": np.percentile(sw[:, j], [2.5, 97.5]).round(3).tolist(),
                             "res_ew_ci": np.percentile(ew[:, j], [2.5, 97.5]).round(3).tolist(),
                             "abs_improvement_ci": np.percentile(-dif, [2.5, 97.5]).round(3).tolist(),
                             "share_reps_ew_better": round(float((dif < 0).mean()), 3)}
    tot_sw, tot_ew = np.abs(sw).sum(1), np.abs(ew).sum(1)
    out["sum_abs_residual"] = {"sw_ci": np.percentile(tot_sw, [2.5, 97.5]).round(3).tolist(),
                               "ew_ci": np.percentile(tot_ew, [2.5, 97.5]).round(3).tolist(),
                               "share_reps_ew_better": round(float((tot_ew < tot_sw).mean()), 3)}
    json.dump(out, open("ew_bootstrap.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(0)

# ---------------------------------------------------------------- data (identical to 21/26)
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
REGIONS = sorted(d.region.unique())
F0 = "success ~ C(region) + log_papers + html_ent + n_commas + strlen"
y, X = __import__("patsy").dmatrices(F0, d, return_type="dataframe")
y = np.asarray(y).ravel()
X = np.asarray(X)

mem = af[["Paper_ID", "Raw_Affiliation"]].drop_duplicates().merge(
    d[["Raw_Affiliation"]].reset_index().rename(columns={"index": "i"}), on="Raw_Affiliation")
papers = mem.Paper_ID.unique()
pidx = {p: k for k, p in enumerate(papers)}


def build_pairs(size_on="all"):
    A, B, Pp = [], [], []
    succ = d.success.values
    for pid, idx in mem.groupby("Paper_ID").i:
        idx = idx.values
        n = len(idx) if size_on == "all" else int(succ[idx].sum())
        if n < 2 or n > 60 or len(idx) < 2:
            continue
        for a, b in itertools.combinations(idx, 2):
            A.append(a); B.append(b); Pp.append(pidx[pid])
    return np.array(A), np.array(B), np.array(Pp)


A, B, PP = build_pairs("all")
succ = d.success.values
reg = d.region.values
R_IDX = np.stack([((reg[A] == r) | (reg[B] == r)) for r in REGIONS], axis=1).astype(float)
both = (succ[A] & succ[B]).astype(float)


def degree(A_, B_, pmult=None):
    if pmult is None:
        return np.bincount(np.concatenate([A_, B_]), minlength=len(d)).astype(float)
    k = pmult[PP]
    return (np.bincount(A_, weights=k, minlength=len(d)) + np.bincount(B_, weights=k, minlength=len(d)))


DEG = degree(A, B)


def fit_ph(freq):
    use = freq > 0
    g = sm.GLM(y[use], X[use], family=sm.families.Binomial(), freq_weights=freq[use]).fit(maxiter=100)
    return 1 / (1 + np.exp(-(X @ g.params)))


def shares(ph, q=0.99, k=None, idx=slice(None)):
    """k: pair multiplicity (bootstrap); returns obs, ipw, full shares (pp) per region."""
    w = np.where(succ == 1, 1 / np.clip(ph, 1e-3, None), 0.0)
    if q is not None:
        w = np.clip(w, None, np.quantile(w[succ == 1], q))
    kk = np.ones(len(A)) if k is None else k
    Ri, bo, a_, b_ = R_IDX[idx], both[idx], A[idx], B[idx]
    kk = kk[idx] if k is not None else kk[idx]
    pw = w[a_] * w[b_] * bo * kk
    full = (Ri * kk[:, None]).sum(0) / kk.sum() * 100
    obs = (Ri * (bo * kk)[:, None]).sum(0) / (bo * kk).sum() * 100
    ipw = (Ri * pw[:, None]).sum(0) / pw.sum() * 100
    return obs, ipw, full


def table(ph, **kw):
    o, i, f = shares(ph, **kw)
    return {r: {"obs": round(float(o[j]), 3), "ipw": round(float(i[j]), 3), "full": round(float(f[j]), 3),
                "residual_pp": round(float(i[j] - f[j]), 3),
                "recovery": round(float((i[j] - o[j]) / (f[j] - o[j])), 3) if abs(f[j] - o[j]) > 1e-9 else None}
            for j, r in enumerate(REGIONS)}


if MODE == "boot":
    BR = int(os.environ.get("B_REPS", 50))
    rng = np.random.default_rng(int(os.environ.get("B_SEED", 1)))
    M = np.zeros((len(papers), len(d)))  # not materialised; use sparse membership instead
    from scipy import sparse
    Mem = sparse.csr_matrix((np.ones(len(mem)), (mem.Paper_ID.map(pidx).values, mem.i.values)), shape=(len(papers), len(d)))
    res_sw, res_ew = [], []
    for _ in range(BR):
        pm = np.bincount(rng.integers(0, len(papers), len(papers)), minlength=len(papers)).astype(float)
        fw = np.asarray(Mem.T @ pm).ravel()             # string frequency in the replicate
        k = pm[PP]                                       # pair multiplicity
        deg_b = degree(A, B, pm)
        ph_sw = fit_ph(fw)
        ph_ew = fit_ph(deg_b)
        o, i1, f = shares(ph_sw, k=k)
        _, i2, _ = shares(ph_ew, k=k)
        res_sw.append(i1 - f)
        res_ew.append(i2 - f)
    np.savez(os.environ.get("B_DUMP", "ew_boot.npz"), res_sw=np.array(res_sw), res_ew=np.array(res_ew), regions=np.array(REGIONS))
    print("saved", BR, "reps")
    sys.exit(0)

# ---------------------------------------------------------------- V1
out = {"tie_rule": TIE_RULE, "n_strings": int(len(d)), "n_pairs": int(len(A))}
ph_sw = fit_ph(np.ones(len(d)))
ph_ew = fit_ph(DEG)
# sanity: SW fit equals smf.logit used in scripts 16-26
ph_chk = smf.logit(F0, d).fit(disp=0).predict(d).values
out["check_sw_equals_manuscript_fit_maxabsdiff"] = float(np.abs(ph_sw - ph_chk).max())
out["V1_string_weighted"] = table(ph_sw)
out["V1_estimand_weighted"] = table(ph_ew)

# ---------------------------------------------------------------- V2 mechanics
v2 = {}
for r in REGIONS:
    g = (reg == r)
    v2[r] = {"pairw_obs": round(float((DEG[g] * succ[g]).sum() / DEG[g].sum()), 5),
             "pairw_pred_SW": round(float((DEG[g] * ph_sw[g]).sum() / DEG[g].sum()), 5),
             "pairw_pred_EW": round(float((DEG[g] * ph_ew[g]).sum() / DEG[g].sum()), 5),
             "stringw_obs": round(float(succ[g].mean()), 5),
             "stringw_pred_EW": round(float(ph_ew[g].mean()), 5)}
out["V2_calibration"] = v2
out["V2_strings_with_zero_degree"] = int((DEG == 0).sum())

# ---------------------------------------------------------------- V3 alternative unit weights
out["V3_weights_n_papers"] = table(fit_ph(d.n_papers.values.astype(float)))
out["V3_weights_sqrt_degree"] = table(fit_ph(np.sqrt(DEG)))

# ---------------------------------------------------------------- V4 sensitivity
out["V4_trim"] = {}
for q in [0.95, None]:
    out["V4_trim"][str(q)] = {"SW": table(ph_sw, q=q), "EW": table(ph_ew, q=q)}
Ag, Bg, _ = build_pairs("geocoded")
sel = np.isin(np.arange(len(A)), [])  # placeholder (geocoded-size filter evaluated separately below)
A_s, B_s, PP_s, R_s, both_s = A, B, PP, R_IDX, both
A, B = Ag, Bg
R_IDX = np.stack([((reg[A] == r) | (reg[B] == r)) for r in REGIONS], axis=1).astype(float)
both = (succ[A] & succ[B]).astype(float)
DEG_g = degree(A, B)
out["V4_size_filter_on_geocoded"] = {"SW": table(ph_sw), "EW_refit_on_this_pairing": table(fit_ph(DEG_g))}
A, B, PP, R_IDX, both = A_s, B_s, PP_s, R_s, both_s

# ---------------------------------------------------------------- V5 distance indicators vs near-complete benchmark
inst_json = "institution_coordinates.json"
v5 = {}
try:
    # rebuild near-complete coordinates exactly as in script 25 by importing its helpers
    import importlib.util
    spec = importlib.util.spec_from_file_location("s25", os.path.join(HERE, "25_institution_coordinates.py"))
    src = open(os.path.join(HERE, "25_institution_coordinates.py"), encoding="utf-8").read()
    src = src.split("# ---------------------------------------------------------------- Q1 proxy quality")[0]
    ns = {"__file__": os.path.join(HERE, "25_institution_coordinates.py")}
    exec(src, ns)
    s25 = ns["s"]
    loc = s25.set_index("Raw_Affiliation")[["inst_lat", "inst_lon"]]
    d2 = d.join(loc, on="Raw_Affiliation")
    lat_nc = np.where(succ == 1, d2.Latitude, d2.inst_lat)
    lon_nc = np.where(succ == 1, d2.Longitude, d2.inst_lon)

    def hav(la1, lo1, la2, lo2):
        p1, p2 = np.radians(la1), np.radians(la2)
        x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(x))

    cty = d.country.values
    dist_cc = hav(d2.Latitude.values[A], d2.Longitude.values[A], d2.Latitude.values[B], d2.Longitude.values[B])
    dist_nc = hav(lat_nc[A], lon_nc[A], lat_nc[B], lon_nc[B])
    intl = (cty[A] != cty[B]).astype(float)
    okp = both == 1
    ncp = ~np.isnan(dist_nc)

    def wq(x, w, q):
        o = np.argsort(x)
        return float(np.interp(q, np.cumsum(w[o]) / w[o].sum(), x[o]))

    def dist_stats(ph):
        w = np.where(succ == 1, 1 / np.clip(ph, 1e-3, None), 0.0)
        w = np.clip(w, None, np.quantile(w[succ == 1], 0.99))
        pw = (w[A] * w[B])[okp]
        return {"mean_km": round(float(np.average(dist_cc[okp], weights=pw)), 1),
                "median_km": round(wq(dist_cc[okp], pw, 0.5), 1),
                "intl_pct": round(float(np.average(intl[okp], weights=pw) * 100), 2)}

    v5["complete_case"] = {"mean_km": round(float(dist_cc[okp].mean()), 1), "median_km": round(float(np.median(dist_cc[okp])), 1),
                           "intl_pct": round(float(intl[okp].mean() * 100), 2)}
    v5["ipw_SW"] = dist_stats(ph_sw)
    v5["ipw_EW"] = dist_stats(ph_ew)
    v5["near_complete"] = {"mean_km": round(float(dist_nc[ncp].mean()), 1), "median_km": round(float(np.median(dist_nc[ncp])), 1),
                           "intl_pct": round(float(intl[ncp].mean() * 100), 2), "pair_coverage_pct": round(float(ncp.mean() * 100), 2)}
    v5["note"] = "same pair set as V1 (size filter on all strings with region); near-complete uses the institution point for failed strings"
except Exception as e:  # pragma: no cover
    v5["error"] = repr(e)[:300]
out["V5_distance"] = v5

json.dump(out, open("estimand_weighted_propensity.json", "w"), indent=1, default=float)
print("check SW == manuscript fit, max |diff|:", out["check_sw_equals_manuscript_fit_maxabsdiff"])
print(f"\n{'region':32} {'obs':>6} {'full':>6} | {'SW ipw':>7} {'res':>6} {'rec':>5} | {'EW ipw':>7} {'res':>6} {'rec':>5} | pairw obs / SW / EW")
for r in REGIONS:
    a_, b_, c_ = out["V1_string_weighted"][r], out["V1_estimand_weighted"][r], v2[r]
    print(f"{r:32} {a_['obs']:6.3f} {a_['full']:6.3f} | {a_['ipw']:7.3f} {a_['residual_pp']:+6.3f} {a_['recovery']:5} | "
          f"{b_['ipw']:7.3f} {b_['residual_pp']:+6.3f} {b_['recovery']:5} | {c_['pairw_obs']:.4f} / {c_['pairw_pred_SW']:.4f} / {c_['pairw_pred_EW']:.4f}")
for nm in ["V3_weights_n_papers", "V3_weights_sqrt_degree"]:
    print(nm, {r[:14]: v["residual_pp"] for r, v in out[nm].items()})
for q, v in out["V4_trim"].items():
    print("trim", q, "SW", {r[:10]: x["residual_pp"] for r, x in v["SW"].items()})
    print("trim", q, "EW", {r[:10]: x["residual_pp"] for r, x in v["EW"].items()})
print("size filter on geocoded: SW", {r[:10]: x["residual_pp"] for r, x in out["V4_size_filter_on_geocoded"]["SW"].items()})
print("size filter on geocoded: EW", {r[:10]: x["residual_pp"] for r, x in out["V4_size_filter_on_geocoded"]["EW_refit_on_this_pairing"].items()})
print("V5:", json.dumps(v5, indent=1))
