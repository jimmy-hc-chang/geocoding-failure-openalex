#!/usr/bin/env python3
"""
24_paper_level_and_minor.py  (revision)

Reviewer 1 comment 3   failure by work-level team structure: number of distinct strings,
                       number of catalogue countries, international vs domestic, author count;
                       logit at appearance level clustered by work
Reviewer 1 comment 4   profile of strings never assigned a standardized institution (7.9%):
                       region, comma count, length, HTML entity, institution-type keywords
Reviewer 2 minor 1     appearance-weighted failure rates; pairs excluding two strings that
                       resolve to the same OpenAlex institution
Reviewer 2 minor 2     effect of the 2..60 geocoded-strings-per-work pair filter
Reviewer 2 minor 7     Wilson intervals for every country in Table 1; regional scheme table

Run from full_run/:  TIE_RULE=exclude python ../analysis/24_paper_level_and_minor.py
Output: paper_level_minor.json, table1_full_wilson.csv, regional_scheme.csv
"""
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pycountry
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_rule import assign_country, TIE_RULE, REGION  # noqa: E402

af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Author_ID", "Raw_Affiliation", "Standardized_Institutions",
                                            "Institution_Countries"])
af = af[af.Raw_Affiliation.notna()]
country = assign_country(af)
geo = pd.read_csv("geocoded.csv")
ok_set = set(geo.loc[geo.Latitude.notna(), "Raw_Affiliation"])
af["success"] = af.Raw_Affiliation.isin(ok_set).astype(int)
af["fail"] = 1 - af.success
af["country"] = af.Raw_Affiliation.map(country)
af["region"] = af.country.map(REGION)
out = {"tie_rule": TIE_RULE}

# ---------------------------------------------------------------- R2 minor 1: appearance-weighted
s_level = geo.assign(fail=geo.Latitude.isna())
app = af.drop_duplicates(["Paper_ID", "Raw_Affiliation"])
out["fail_rate_unique_strings_pct"] = round(float(s_level.fail.mean() * 100), 2)
out["fail_rate_work_string_appearances_pct"] = round(float(app.fail.mean() * 100), 2)
out["fail_rate_authorship_rows_pct"] = round(float(af.fail.mean() * 100), 2)
nap = app.groupby("Raw_Affiliation").size()
s_level["n_app"] = s_level.Raw_Affiliation.map(nap)
s_level["app_bin"] = pd.cut(s_level.n_app, [0, 1, 2, 5, 20, 1e9], labels=["1", "2", "3-5", "6-20", ">20"])
out["fail_by_string_frequency"] = {str(k): {"n": int(g.size), "fail_pct": round(float(g.mean() * 100), 1)}
                                   for k, g in s_level.groupby("app_bin", observed=True).fail}
by_reg = app[app.region.notna()].groupby("region").fail.mean().mul(100).round(1)
uniq_reg = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True)
uniq_reg = uniq_reg.assign(region=uniq_reg.country.map(REGION), fail=uniq_reg.Latitude.isna()).groupby("region").fail.mean().mul(100).round(1)
out["regional_fail_unique_vs_appearance"] = {r: [float(uniq_reg.get(r, np.nan)), float(by_reg.get(r, np.nan))] for r in sorted(by_reg.index)}

# ---------------------------------------------------------------- R1 comment 3: work-level structure
w = app.groupby("Paper_ID").agg(n_strings=("Raw_Affiliation", "size"),
                                n_countries=("country", lambda s: s.dropna().nunique()),
                                n_fail=("fail", "sum"))
w["n_authors"] = af.groupby("Paper_ID").Author_ID.nunique()
w["intl"] = (w.n_countries >= 2).astype(int)
w["any_fail"] = (w.n_fail > 0).astype(int)
w["all_fail"] = (w.n_fail == w.n_strings).astype(int)
w["size_bin"] = pd.cut(w.n_strings, [0, 1, 3, 9, 1e9], labels=["1", "2-3", "4-9", "10+"])
w["auth_bin"] = pd.cut(w.n_authors, [0, 3, 6, 10, 1e9], labels=["1-3", "4-6", "7-10", "11+"])
app2 = app.merge(w[["n_strings", "n_countries", "intl", "n_authors", "size_bin", "auth_bin"]], left_on="Paper_ID", right_index=True)
out["r1c3_fail_by_work_size"] = {str(k): {"appearances": int(g.size), "fail_pct": round(float(g.mean() * 100), 1)}
                                for k, g in app2.groupby("size_bin", observed=True).fail}
out["r1c3_fail_by_authors"] = {str(k): {"appearances": int(g.size), "fail_pct": round(float(g.mean() * 100), 1)}
                              for k, g in app2.groupby("auth_bin", observed=True).fail}
out["r1c3_fail_intl_vs_domestic"] = {("international" if k else "domestic"): {"appearances": int(g.size), "fail_pct": round(float(g.mean() * 100), 1)}
                                    for k, g in app2.groupby("intl").fail}
x = app2[app2.region.notna()].copy()
x["log_strings"] = np.log(x.n_strings)
x["log_authors"] = np.log(x.n_authors.clip(lower=1))
r = smf.logit('fail ~ C(region, Treatment("Northern America")) + intl + log_strings + log_authors', x).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": x.Paper_ID})
ci = np.exp(r.conf_int())
out["r1c3_logit_clustered_by_work"] = {k.split("[T.")[-1].rstrip("]"): [round(float(np.exp(r.params[k])), 3),
                                                                       round(float(ci.loc[k, 0]), 3), round(float(ci.loc[k, 1]), 3)]
                                       for k in r.params.index if k != "Intercept"}
# co-failure within works: probability a second string fails given one failed, by region of the failing string
pairs = []
for pid, g in app2[app2.n_strings.between(2, 60)].groupby("Paper_ID"):
    v = g[["fail", "region"]].values
    for a, b in itertools.permutations(v, 2):
        pairs.append((a[0], b[0], a[1], b[1]))
P = pd.DataFrame(pairs, columns=["fa", "fb", "ra", "rb"])
out["r1c3_cofailure"] = {"P(b fails | a fails)": round(float(P[P.fa == 1].fb.mean() * 100), 1),
                         "P(b fails | a ok)": round(float(P[P.fa == 0].fb.mean() * 100), 1),
                         "same_region_P(b fails | a fails)": round(float(P[(P.fa == 1) & (P.ra == P.rb)].fb.mean() * 100), 1),
                         "diff_region_P(b fails | a fails)": round(float(P[(P.fa == 1) & (P.ra != P.rb)].fb.mean() * 100), 1)}
out["r1c3_works_all_strings_failed_pct"] = round(float(w[w.n_strings >= 2].all_fail.mean() * 100), 2)

# ---------------------------------------------------------------- R1 comment 4: never-resolved strings
res_share = af.assign(h=af.Standardized_Institutions.notna()).groupby("Raw_Affiliation").h.mean()
s = geo.copy()
s["unresolved"] = (s.Raw_Affiliation.map(res_share).fillna(0) == 0)
s["fail"] = s.Latitude.isna()
s["country"] = s.Raw_Affiliation.map(country)
s["region"] = s.country.map(REGION)
# catalogue country exists only for resolved strings, so locate strings by their geocode (offline reverse geocoding)
import reverse_geocoder as rg  # noqa: E402
okm = s.Latitude.notna()
s.loc[okm, "geo_cc"] = [z["cc"] for z in rg.search(list(zip(s.loc[okm, "Latitude"], s.loc[okm, "Longitude"])), mode=1)]
s["geo_region"] = s.geo_cc.map(REGION).fillna("not geocoded / unmapped")
s["n_commas"] = s.Raw_Affiliation.str.count(",")
s["comma_bin"] = pd.cut(s.n_commas, [-1, 0, 1, 2, 999], labels=["0", "1", "2", "3+"])
s["strlen"] = s.Raw_Affiliation.str.len()
s["html_ent"] = s.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False)
TYPES = [("company", r"\b(?:inc|ltd|llc|gmbh|corp|corporation|company|limited|plc|pvt|k\.k\.|co\.,?\s*ltd|s\.p\.a|s\.a\.s|b\.v\.|ag\b(?=$|[ ,]))"),
         ("hospital", r"\b(?:hospital|clinic|medical cent(?:er|re)|h[oô]pital|klinik)\b"),
         ("government", r"\b(?:ministry|national laborator(?:y|ies)|government|agency|bureau|council|commission)\b"),
         ("university", r"\b(?:universit(?:y|é|ä|à|ad|ade|at|e|eit|i)|college|polytechnic|politecnico|iit|nit)\b"),
         ("research_institute", r"\b(?:institute|institut|instituto|istituto|academy|akadem\w*|research cent(?:er|re)|cnrs|csic|cnr|fraunhofer|max planck)\b")]
for nm, rx in TYPES:
    s[f"type_{nm}"] = s.Raw_Affiliation.str.contains(rx, case=False, regex=True, na=False)
s["type_none"] = ~s[[f"type_{nm}" for nm, _ in TYPES]].any(axis=1)
prof = {"n_unresolved": int(s.unresolved.sum()), "unresolved_pct": round(float(s.unresolved.mean() * 100), 2),
        "fail_unresolved_vs_resolved_pct": [round(float(s[s.unresolved].fail.mean() * 100), 1),
                                            round(float(s[~s.unresolved].fail.mean() * 100), 1)]}
prof["unresolved_rate_by_geocoded_region"] = s[okm].groupby("geo_region").unresolved.mean().mul(100).round(1).to_dict()
prof["unresolved_rate_by_geocoded_country_top"] = s[okm].groupby("geo_cc").unresolved.agg(["size","mean"]).query("size>=300").sort_values("mean",ascending=False).head(15)["mean"].mul(100).round(1).to_dict()
prof["unresolved_rate_by_comma"] = {str(k): round(float(v * 100), 1) for k, v in s.groupby("comma_bin", observed=True).unresolved.mean().items()}
prof["unresolved_rate_html_ent"] = {str(k): round(float(v * 100), 1) for k, v in s.groupby("html_ent").unresolved.mean().items()}
prof["mean_length_unresolved_vs_resolved"] = [round(float(s[s.unresolved].strlen.mean()), 1), round(float(s[~s.unresolved].strlen.mean()), 1)]
prof["unresolved_rate_by_type_keyword"] = {nm: {"n": int(s[f"type_{nm}"].sum()), "unresolved_pct": round(float(s[s[f"type_{nm}"]].unresolved.mean() * 100), 1)}
                                           for nm in [t for t, _ in TYPES] + ["none"]}
prof["share_of_unresolved_with_type"] = {nm: round(float(s[s.unresolved][f"type_{nm}"].mean() * 100), 1) for nm in [t for t, _ in TYPES] + ["none"]}
lt = s.copy()
lt["non_ascii"] = lt.Raw_Affiliation.map(lambda z: any(ord(c) > 127 for c in z))
prof["unresolved_rate_non_ascii"] = {str(k): round(float(v * 100), 1) for k, v in lt.groupby("non_ascii").unresolved.mean().items()}
r = smf.logit("unresolved ~ C(geo_region, Treatment('Northern America')) + C(comma_bin, Treatment('2')) + np.log(strlen) + html_ent"
              " + type_company + type_hospital + type_government + type_university + type_research_institute",
              s.assign(unresolved=s.unresolved.astype(int), html_ent=s.html_ent.astype(int),
                       **{f"type_{t}": s[f"type_{t}"].astype(int) for t, _ in TYPES})[okm]).fit(disp=0)
ci = np.exp(r.conf_int())
prof["logit_unresolved"] = {k.split("[T.")[-1].rstrip("]"): [round(float(np.exp(r.params[k])), 3), round(float(ci.loc[k, 0]), 3),
                                                             round(float(ci.loc[k, 1]), 3)] for k in r.params.index if k != "Intercept"}
out["r1c4_unresolved_profile"] = prof

# ---------------------------------------------------------------- R2 minor 2 and minor 1b: pair filter, same-institution pairs
inst = af.dropna(subset=["Standardized_Institutions"]).groupby("Raw_Affiliation").Standardized_Institutions.agg(
    lambda z: z.value_counts().index[0])
okd = geo[geo.Latitude.notna()].merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True)
okd["region"] = okd.country.map(REGION)
okd = okd[okd.region.notna()]
mem = app[["Paper_ID", "Raw_Affiliation"]].merge(okd[["Raw_Affiliation", "region"]], on="Raw_Affiliation")
mem["inst"] = mem.Raw_Affiliation.map(inst)
LA = "Latin America & Caribbean"


def la_share(cap, drop_same_inst=False):
    tot = la = 0
    nworks = 0
    for pid, g in mem.groupby("Paper_ID"):
        n = len(g)
        if n < 2 or (cap and n > cap):
            continue
        nworks += 1
        v = g[["region", "inst"]].values
        for a, b in itertools.combinations(v, 2):
            if drop_same_inst and isinstance(a[1], str) and a[1] == b[1]:
                continue
            tot += 1
            la += (a[0] == LA) or (b[0] == LA)
    return {"n_pairs": int(tot), "n_works": int(nworks), "LA_share_pct": round(la / tot * 100, 3)}


sizes = mem.groupby("Paper_ID").size()
out["r2m2_pair_filter"] = {"works_excluded_gt60": int((sizes > 60).sum()),
                           "pairs_excluded_gt60": int(sum(n * (n - 1) // 2 for n in sizes[sizes > 60])),
                           "with_filter_60": la_share(60), "no_filter": la_share(None), "filter_30": la_share(30)}
out["r2m1_same_institution_pairs"] = {"all_pairs": la_share(60), "excluding_same_institution": la_share(60, True)}

# ---------------------------------------------------------------- R2 minor 7: Table 1 with Wilson for all; regional scheme
d = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True)
d["fail"] = d.Latitude.isna()
t1 = d.groupby("country").fail.agg(n="size", failed="sum")
lo, hi = proportion_confint(t1.failed, t1.n, method="wilson")
t1["fail_pct"] = (t1.failed / t1.n * 100).round(1)
t1["wilson_lo"] = (lo * 100).round(1)
t1["wilson_hi"] = (hi * 100).round(1)
t1["region"] = t1.index.map(REGION)
t1["name"] = [getattr(pycountry.countries.get(alpha_2=c), "name", c) for c in t1.index]
t1.sort_values("n", ascending=False).to_csv("table1_full_wilson.csv")
out["table1_requested"] = {c: t1.loc[c, ["n", "fail_pct", "wilson_lo", "wilson_hi"]].to_dict() for c in ["MX", "IR", "IN"] if c in t1.index}
sch = pd.DataFrame([(c, getattr(pycountry.countries.get(alpha_2=c), "name", c), r) for c, r in REGION.items()],
                   columns=["iso2", "country", "region"]).sort_values(["region", "country"])
sch["in_corpus_n_strings"] = sch.iso2.map(t1.n).fillna(0).astype(int)
sch.to_csv("regional_scheme.csv", index=False)
out["boundary_cases"] = {c: REGION.get(c) for c in ["IL", "TR", "SD", "CY", "MT", "GR", "RU", "KZ", "AF", "PK", "EG", "MR", "MV"]}

json.dump(out, open("paper_level_minor.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:6000])
