#!/usr/bin/env python3
"""
23_models_and_classification.py  (revision, Reviewer 2 Major 4 and Major 5; minor 3)

Major 4(a)  Inference for the regional model M2 when regions contain few countries:
            (i)  original: logit, SE clustered by country
            (ii) wild cluster bootstrap (Rademacher, country clusters, score-based,
                 restricted under H0 not imposed -> percentile-t on the coefficient), B reps
            (iii) country-level aggregated binomial GLM with region fixed effects and an
                 observation-level random intercept approximated by quasi-binomial dispersion
                 (Pearson chi2 / df) -> overdispersion-adjusted intervals
            (iv) mixed logit with country random intercept (statsmodels BinomialBayesMixedGLM, VB)
Major 4(b)  Comma count as categories (0,1,2,3,4+) and natural cubic spline (df=4) in M3.
Major 5     North-South / script contrast under three schemes:
            custom (manuscript), World Bank high-income (FY2026 API list; Taiwan coded HIC
            by convention), and a 2x2 cross of income x script to break collinearity.
minor 3     Country x failure chi-square: asymptotic, Monte Carlo (fixed margins), and
            restricted to countries with n >= 30.

Run from full_run/:  TIE_RULE=exclude python ../analysis/23_models_and_classification.py
Output: models_classification.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_rule import assign_country, TIE_RULE, REGION  # noqa: E402

SEED = 20260811
B_WILD = int(os.environ.get("B_WILD", 999))
B_MC = int(os.environ.get("B_MC", 2000))
rng = np.random.default_rng(SEED)

GLOBAL_NORTH = set("""US CA JP DE FR IT GB KR TW AU NZ CH DK BE PL SE AT NL RO CZ NO SK FI HU GR IE HR EE SI CY LV LT MT LU IS PT ES BG RU UA BY MD RS ME MK BA AL IL SM AD MC LI GI""".split())
LATIN = set("""US CA GB AU NZ IE DE FR IT ES PT NL BE CH AT DK SE NO FI IS PL CZ SK HU RO HR SI EE LV LT MT LU CY TR BR MX CL CO AR EC PY PE UY CR PA CU VE DO GT HN NI SV BO ID MY VN PH SG NG ZA KE GH TZ MZ BW ZW CM SN BJ BF TD SS FJ MU AZ UZ TM AL BA ME MD TN DZ MA HT AO GQ CV ST TL PG SB VU WS TO KI NR TV FM MH PW BN""".split())
# World Bank FY2026 high-income economies (api.worldbank.org/v2/country?incomeLevel=HIC, retrieved 2026-10-22)
WB_HIC = set("""AW AD AE AS AG AU AT BE BG BH BS BM BB BN CA CH JG CL CR CW KY CY CZ DE DK ES EE FI FR FO GB GI GR GL GU GY
HK HR HU IM IE IS IL IT JP KN KR KW LI LT LU LV MO MF MC MT MP NC NL NO NR NZ OM PA PW PL PR PT PF QA RO RU SA SG SM SK
SI SE SX SC TC TT UY US VG VI""".split()) | {"TW"}

af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Raw_Affiliation", "Institution_Countries"])
af = af[af.Raw_Affiliation.notna()]
country = assign_country(af)
n_papers = af.groupby("Raw_Affiliation")["Paper_ID"].nunique()
geo = pd.read_csv("geocoded.csv")
m = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True, how="left")
m = m.merge(n_papers.rename("n_papers"), left_on="Raw_Affiliation", right_index=True, how="left")
m["fail"] = m.Latitude.isna().astype(int)
m["log_papers"] = np.log(m.n_papers)
m["n_commas"] = m.Raw_Affiliation.str.count(",")
m["comma_cat"] = pd.cut(m.n_commas, [-1, 0, 1, 2, 3, 999], labels=["c0", "c1", "c2", "c3", "c4plus"]).astype(str)
m["strlen"] = m.Raw_Affiliation.str.len()
m["strlen100"] = m.strlen / 100
m["html_ent"] = m.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int)
m["region"] = m.country.map(REGION)
d = m[m.region.notna()].copy()
d["global_south"] = (~d.country.isin(GLOBAL_NORTH)).astype(int)
d["non_latin"] = (~d.country.isin(LATIN)).astype(int)
d["wb_nonhic"] = (~d.country.isin(WB_HIC)).astype(int)
out = {"tie_rule": TIE_RULE, "n": int(len(d)), "n_countries": int(d.country.nunique())}


def tidy(r, exp=True):
    ci = r.conf_int()
    t = pd.DataFrame({"coef": r.params, "lo": ci[0], "hi": ci[1], "p": r.pvalues})
    if exp:
        t[["coef", "lo", "hi"]] = np.exp(t[["coef", "lo", "hi"]])
    t.index = [i.split("[T.")[-1].rstrip("]") for i in t.index]
    return {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in t.round(4).to_dict("index").items()}


def clfit(f, data):
    return smf.logit(f, data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": data.country})


# ---------------------------------------------------------------- Major 5: classification
comp = {}
comp["custom_scheme_counts"] = {"global_south_strings": int(d.global_south.sum()),
                                "global_south_share_CN_pct": round(float((d[d.global_south == 1].country == "CN").mean() * 100), 1),
                                "global_south_and_nonlatin_pct": round(float(d[d.global_south == 1].non_latin.mean() * 100), 1),
                                "custom_south_but_WB_HIC": sorted(set(d[(d.global_south == 1) & (d.wb_nonhic == 0)].country)),
                                "custom_north_but_WB_nonHIC": sorted(set(d[(d.global_south == 0) & (d.wb_nonhic == 1)].country))}
comp["wb_counts"] = {"wb_nonhic_strings": int(d.wb_nonhic.sum()),
                     "wb_nonhic_and_nonlatin_pct": round(float(d[d.wb_nonhic == 1].non_latin.mean() * 100), 1),
                     "phi_custom_vs_script": round(float(np.corrcoef(d.global_south, d.non_latin)[0, 1]), 3),
                     "phi_wb_vs_script": round(float(np.corrcoef(d.wb_nonhic, d.non_latin)[0, 1]), 3)}
comp["M1_custom"] = tidy(clfit("fail ~ global_south + non_latin + log_papers", d))
comp["M1_worldbank"] = tidy(clfit("fail ~ wb_nonhic + non_latin + log_papers", d))
comp["M1_worldbank_only"] = tidy(clfit("fail ~ wb_nonhic + log_papers", d))
comp["M1_script_only"] = tidy(clfit("fail ~ non_latin + log_papers", d))
d["cell"] = np.where(d.wb_nonhic == 1, "nonHIC", "HIC") + "_" + np.where(d.non_latin == 1, "nonLatin", "Latin")
comp["cell_fail_rates"] = {c: {"n": int(g.size), "fail_pct": round(float(g.mean() * 100), 1),
                               "n_countries": int(d[d.cell == c].country.nunique())}
                           for c, g in d.groupby("cell").fail}
comp["M1_2x2_cells"] = tidy(clfit('fail ~ C(cell, Treatment(reference="HIC_Latin")) + log_papers', d))
# within-North spread (the scheme-free argument)
hic = d[d.wb_nonhic == 0].groupby("country").fail.agg(["size", "mean"])
hic = hic[hic["size"] >= 100].sort_values("mean")
comp["within_HIC_spread"] = {"n_countries_n100": int(len(hic)),
                             "min": [hic.index[0], round(float(hic["mean"].iloc[0] * 100), 1)],
                             "max": [hic.index[-1], round(float(hic["mean"].iloc[-1] * 100), 1)],
                             "FR": round(float(hic.loc["FR", "mean"] * 100), 1) if "FR" in hic.index else None,
                             "US": round(float(hic.loc["US", "mean"] * 100), 1)}
nh = d[d.wb_nonhic == 1].groupby("country").fail.agg(["size", "mean"])
nh = nh[nh["size"] >= 100]
comp["nonHIC_n100_rates"] = {c: round(float(v * 100), 1) for c, v in nh["mean"].sort_values().items()}
out["major5_classification"] = comp

# ---------------------------------------------------------------- Major 4(b): comma specification
spec = {}
F_lin = "fail ~ global_south + non_latin + log_papers + strlen + n_commas"
F_cat = "fail ~ global_south + non_latin + log_papers + strlen + C(comma_cat, Treatment('c2'))"
F_spl = "fail ~ global_south + non_latin + log_papers + strlen + cr(n_commas, df=4)"
for nm, f in [("linear", F_lin), ("categorical", F_cat), ("spline", F_spl)]:
    r = clfit(f, d)
    spec[nm] = {"aic": round(float(r.aic), 1), "bic": round(float(r.bic), 1), "llf": round(float(r.llf), 1),
                "coefs": tidy(r)}
for nm, f in [("M2_region_linear_comma", 'fail ~ C(region, Treatment("Northern America")) + log_papers + strlen + n_commas'),
              ("M2_region_cat_comma", 'fail ~ C(region, Treatment("Northern America")) + log_papers + strlen + C(comma_cat, Treatment("c2"))')]:
    r = clfit(f, d)
    spec[nm] = {"aic": round(float(r.aic), 1), "LA_OR": tidy(r)["Latin America & Caribbean"]}
out["major4b_comma_spec"] = spec

# ---------------------------------------------------------------- Major 4(a): inference on M2
F2 = 'fail ~ C(region, Treatment("Northern America")) + log_papers'
r0 = clfit(F2, d)
inf = {"cluster_country": tidy(r0)}
# wild cluster bootstrap (score / multiplier bootstrap on the logit estimating equations)
X = r0.model.exog
y = r0.model.endog
beta = r0.params.values
p = 1 / (1 + np.exp(-X @ beta))
H = (X * (p * (1 - p))[:, None]).T @ X
Hinv = np.linalg.inv(H)
score_i = X * (y - p)[:, None]
cl = d.country.values
uc, inv = np.unique(cl, return_inverse=True)
S_c = np.zeros((len(uc), X.shape[1]))
np.add.at(S_c, inv, score_i)
names = list(r0.params.index)
se_cl = np.sqrt(np.diag(Hinv @ (S_c.T @ S_c) @ Hinv))
draws = []
tstats = []
for _ in range(B_WILD):
    v = rng.choice([-1.0, 1.0], size=len(uc))
    Sb = S_c * v[:, None]
    db = Hinv @ Sb.sum(0)
    # one-step bootstrap estimate and its cluster SE
    draws.append(db)
    tstats.append(db / np.sqrt(np.diag(Hinv @ (Sb.T @ Sb) @ Hinv)))
draws = np.array(draws)
tstats = np.array(tstats)
wild = {}
for j, nm in enumerate(names):
    if "region" not in nm:
        continue
    key = nm.split("[T.")[-1].rstrip("]")
    lo_t, hi_t = np.percentile(tstats[:, j], [2.5, 97.5])
    wild[key] = {"OR": round(float(np.exp(beta[j])), 4),
                 "lo_percentile_t": round(float(np.exp(beta[j] - hi_t * se_cl[j])), 4),
                 "hi_percentile_t": round(float(np.exp(beta[j] - lo_t * se_cl[j])), 4),
                 "lo_percentile": round(float(np.exp(beta[j] + np.percentile(draws[:, j], 2.5))), 4),
                 "hi_percentile": round(float(np.exp(beta[j] + np.percentile(draws[:, j], 97.5))), 4),
                 "n_countries_in_region": int(d[d.region == key].country.nunique())}
inf["wild_cluster_bootstrap"] = wild
inf["wild_B"] = B_WILD
# mixed logit: country random intercept (variational Bayes)
try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    md = BinomialBayesMixedGLM.from_formula(F2, {"country": "0 + C(country)"}, d)
    rv = md.fit_vb()
    fe = pd.DataFrame({"coef": rv.fe_mean, "sd": rv.fe_sd}, index=md.exog_names)
    mixed = {}
    for nm, row in fe.iterrows():
        if "region" not in nm:
            continue
        key = nm.split("[T.")[-1].rstrip("]")
        mixed[key] = {"OR": round(float(np.exp(row.coef)), 4),
                      "lo": round(float(np.exp(row.coef - 1.96 * row.sd)), 4),
                      "hi": round(float(np.exp(row.coef + 1.96 * row.sd)), 4)}
    inf["mixed_country_RE"] = mixed
    inf["mixed_country_RE_sd"] = round(float(np.exp(rv.vcp_mean[0])), 4)
except Exception as e:  # pragma: no cover
    inf["mixed_country_RE_error"] = str(e)[:200]
out["major4a_inference"] = inf

# ---------------------------------------------------------------- minor 3: chi-square variants
ct = pd.crosstab(d.country, d.fail)
chi2, pval, dof, _ = stats.chi2_contingency(ct)
res = {"asymptotic": {"chi2": round(float(chi2), 1), "dof": int(dof), "cramer_v": round(float(np.sqrt(chi2 / len(d))), 3)}}
# Monte Carlo with fixed margins: permute failure labels across strings
f = d.fail.values.copy()
codes = pd.factorize(d.country)[0]
exp_fail = np.bincount(codes) * f.mean()
exp_ok = np.bincount(codes) * (1 - f.mean())


def chi(fv):
    of = np.bincount(codes, weights=fv)
    oo = np.bincount(codes) - of
    return ((of - exp_fail) ** 2 / exp_fail + (oo - exp_ok) ** 2 / exp_ok).sum()


obs = chi(f)
ge = 0
for _ in range(B_MC):
    ge += chi(rng.permutation(f)) >= obs
res["monte_carlo"] = {"B": B_MC, "p": float((ge + 1) / (B_MC + 1)), "chi2_obs": round(float(obs), 1)}
big = ct[ct.sum(1) >= 30]
c2, p2, df2, _ = stats.chi2_contingency(big)
res["n_ge_30"] = {"n_countries": int(len(big)), "chi2": round(float(c2), 1), "dof": int(df2),
                  "cramer_v": round(float(np.sqrt(c2 / big.values.sum())), 3), "p": float(p2)}
out["minor3_chi2"] = res

json.dump(out, open("models_classification.json", "w"), indent=1, default=str)
print(json.dumps({"major5": {k: comp[k] for k in ["custom_scheme_counts", "wb_counts", "cell_fail_rates", "within_HIC_spread"]},
                  "M1_custom": comp["M1_custom"], "M1_worldbank": comp["M1_worldbank"], "M1_2x2": comp["M1_2x2_cells"],
                  "comma_aic": {k: v.get("aic") for k, v in spec.items()},
                  "LA_OR": {"cluster": inf["cluster_country"]["Latin America & Caribbean"], "wild": wild.get("Latin America & Caribbean"),
                            "mixed": inf.get("mixed_country_RE", {}).get("Latin America & Caribbean")},
                  "chi2": res}, indent=1, default=str))

# ---------------------------------------------------------------- additions after first pass
# Major 5: the structural-control model M3 under each scheme (categorical comma term, per Major 4b)
for nm, f in [("M3cat_custom", "fail ~ global_south + non_latin + log_papers + strlen + C(comma_cat, Treatment('c2'))"),
              ("M3cat_worldbank", "fail ~ wb_nonhic + non_latin + log_papers + strlen + C(comma_cat, Treatment('c2'))"),
              ("M3lin_worldbank", "fail ~ wb_nonhic + non_latin + log_papers + strlen + n_commas")]:
    comp[nm] = tidy(clfit(f, d))
out["major5_classification"] = comp

# Major 4(a): between-country inference. Country-level binomial GLM with region effects,
# overdispersion scaled by Pearson chi2/df (quasi-binomial): the effective n is the number of countries.
cty = d.groupby("country").agg(fail=("fail", "sum"), n=("fail", "size"), region=("region", "first"),
                               log_papers=("log_papers", "mean")).reset_index()
cty["ok"] = cty.n - cty.fail
q = smf.glm('fail + ok ~ C(region, Treatment("Northern America")) + log_papers', cty,
            family=sm.families.Binomial()).fit(scale="X2")
inf["country_level_quasibinomial"] = {k: v for k, v in tidy(q).items() if k not in ("Intercept", "log_papers")}
inf["country_level_quasibinomial_dispersion"] = round(float(q.scale), 2)
inf["country_level_n_countries"] = int(len(cty))
# reference-free coding: each region against the grand mean (sum contrasts)
s = clfit("fail ~ C(region, Sum) + log_papers", d)
inf["sum_contrast_cluster"] = {k: v for k, v in tidy(s).items() if "S." in k or "Sum" in k}
out["major4a_inference"] = inf

json.dump(out, open("models_classification.json", "w"), indent=1, default=str)
print(json.dumps({"M3cat_custom": comp["M3cat_custom"], "M3cat_worldbank": comp["M3cat_worldbank"],
                  "quasi": inf["country_level_quasibinomial"], "disp": inf["country_level_quasibinomial_dispersion"]}, indent=1))
