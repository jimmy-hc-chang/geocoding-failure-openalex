#!/usr/bin/env python3
"""
25_institution_coordinates.py  (revision, Reviewer 2 Major 2; hybrid recommendation of Section 4.1(v))

Uses OpenAlex institution-entity coordinates (geo.latitude / geo.longitude) from the
2026-06 snapshot to give a location to failed strings that resolve to an institution,
and asks three questions:

  Q1  Are inst. coordinates a usable proxy?  For SUCCESSFUL strings with a resolved
      institution, distance between the geocoded point and the institution point.
  Q2  Within-country MAR: do failed strings sit in different places, within their
      country, from successful ones?  Per country (>=30 failed strings with inst. coords):
      distance to the country's success-weighted centroid and share in the country's
      top-5 cities (by successful strings), failed vs successful; pooled with country
      fixed effects.
  Q3  Distance indicators under five constructions, same works:
        complete-case (geocoded endpoints only)
        IPW (manuscript weights)
        near-complete (geocode where it succeeded, institution point where it failed)
        catalogue-only (institution point for every resolved string)
        hybrid (institution point where resolved, geocode for the unresolved remainder)

Input: openalex_institutions/inst_geo.tsv, built on first run from the snapshot copy at
       ../openalex_institutions/updated_date=*/part_*.gz (the public OpenAlex institutions snapshot, June 2026, from s3://openalex/data/institutions/)
Run from full_run/:  TIE_RULE=exclude python ../analysis/25_institution_coordinates.py
Output: institution_coordinates.json
"""
import csv
import glob
import gzip
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from country_rule import assign_country, TIE_RULE, REGION  # noqa: E402

INST_DIR = os.environ.get("INST_DIR", os.path.join(HERE, "..", "openalex_institutions"))
TSV = os.path.join(INST_DIR, "inst_geo.tsv")


def build_tsv():
    files = sorted(glob.glob(os.path.join(INST_DIR, "updated_date=*", "part_*.gz")))
    if not files:
        sys.exit(f"No snapshot files under {INST_DIR}. Download the OpenAlex institutions snapshot (s3://openalex/data/institutions/) there first.")
    with open(TSV, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["id", "display_name", "country_code", "type", "lat", "lon", "city", "works_count"])
        n = 0
        for f in files:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    g = d.get("geo") or {}
                    w.writerow([d["id"].rsplit("/", 1)[-1], d.get("display_name") or "", d.get("country_code") or "",
                                d.get("type") or "", g.get("latitude"), g.get("longitude"), g.get("city") or "",
                                d.get("works_count") or 0])
                    n += 1
    print(f"built {TSV}: {n} institutions")


if not os.path.exists(TSV):
    build_tsv()
inst = pd.read_csv(TSV, sep="\t", dtype={"id": str}, keep_default_na=False, na_values=[""])
inst = inst.dropna(subset=["lat", "lon"])
by_name = {}
for r in inst.sort_values("works_count", ascending=False).itertuples(index=False):
    by_name.setdefault(r.display_name, []).append(r)
NAMES = set(by_name)


def split_names(s):
    """Standardized_Institutions is a ', '-joined list of display names, and names may contain ', '.
    Greedy longest match against known display names."""
    toks = s.split(", ")
    out, i = [], 0
    while i < len(toks):
        hit = None
        for j in range(len(toks), i, -1):
            cand = ", ".join(toks[i:j])
            if cand in NAMES:
                hit = (cand, j)
                break
        if hit:
            out.append(hit[0])
            i = hit[1]
        else:
            i += 1
    return out


# ---------------------------------------------------------------- strings
af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Raw_Affiliation", "Standardized_Institutions", "Institution_Countries"])
af = af[af.Raw_Affiliation.notna()]
country = assign_country(af)
modal_std = af.dropna(subset=["Standardized_Institutions"]).groupby("Raw_Affiliation").Standardized_Institutions.agg(
    lambda z: z.value_counts().index[0])
geo = pd.read_csv("geocoded.csv")
s = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True, how="left")
s["std"] = s.Raw_Affiliation.map(modal_std)
s["success"] = s.Latitude.notna()


def pick(std, cc):
    if not isinstance(std, str):
        return None
    cands = []
    for nm in split_names(std):
        for r in by_name.get(nm, []):
            cands.append(r)
    if not cands:
        return None
    if isinstance(cc, str):
        same = [r for r in cands if r.country_code == cc]
        if same:
            return same[0]
    return cands[0] if len({r.id for r in cands}) == 1 else None


picked = [pick(a, b) for a, b in zip(s["std"], s.country)]
s["inst_id"] = [p.id if p else None for p in picked]
s["inst_lat"] = [p.lat if p else np.nan for p in picked]
s["inst_lon"] = [p.lon if p else np.nan for p in picked]
s["inst_cc"] = [p.country_code if p else None for p in picked]
s["region"] = s.country.map(REGION)
out = {"tie_rule": TIE_RULE, "n_institutions_with_coords": int(len(inst)),
       "resolved_strings_pct": round(float(s["std"].notna().mean() * 100), 2),
       "inst_coord_assigned_pct_all": round(float(s.inst_id.notna().mean() * 100), 2),
       "inst_coord_assigned_pct_failed": round(float(s[~s.success].inst_id.notna().mean() * 100), 2),
       "inst_coord_assigned_pct_failed_with_region": round(float(s[(~s.success) & s.region.notna()].inst_id.notna().mean() * 100), 2)}


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- Q1 proxy quality
v = s[s.success & s.inst_id.notna()].copy()
v["gap_km"] = hav(v.Latitude, v.Longitude, v.inst_lat, v.inst_lon)
q = v.gap_km.quantile([0.25, 0.5, 0.75, 0.9]).round(1).to_dict()
out["Q1_geocode_vs_institution_point"] = {"n": int(len(v)), "quantiles_km": {str(k): float(x) for k, x in q.items()},
                                          "within_10km_pct": round(float((v.gap_km <= 10).mean() * 100), 1),
                                          "within_50km_pct": round(float((v.gap_km <= 50).mean() * 100), 1),
                                          "over_500km_pct": round(float((v.gap_km > 500).mean() * 100), 1),
                                          "median_by_region": v.groupby("region").gap_km.median().round(1).to_dict()}

# ---------------------------------------------------------------- Q2 within-country location
s["lat_x"] = np.where(s.success, s.Latitude, s.inst_lat)
s["lon_x"] = np.where(s.success, s.Longitude, s.inst_lon)
k = s[s.country.notna() & s.lat_x.notna()].copy()
cent = k[k.success].groupby("country")[["Latitude", "Longitude"]].mean()
k = k.merge(cent.rename(columns={"Latitude": "c_lat", "Longitude": "c_lon"}), left_on="country", right_index=True)
k["d_cent"] = hav(k.lat_x, k.lon_x, k.c_lat, k.c_lon)
# top-5 cities by successful strings: use institution city where available, else rounded geocode cell
inst_city = inst.set_index("id").city
k["city"] = k.inst_id.map(inst_city)
top5 = k[k.success & k.city.notna()].groupby("country").city.agg(lambda z: set(z.value_counts().head(5).index))
k["in_top5"] = [isinstance(c, str) and c in top5.get(cc, set()) for c, cc in zip(k.city, k.country)]
k["failed"] = (~k.success).astype(int)
k["log_d"] = np.log1p(k.d_cent)
q2 = {}
elig = k.groupby("country").failed.sum()
elig = elig[elig >= 30].index
kk = k[k.country.isin(elig) & k.inst_id.notna()]
r1 = smf.ols("log_d ~ failed + C(country)", kk).fit(cov_type="cluster", cov_kwds={"groups": kk.country})
r2 = smf.ols("in_top5 ~ failed + C(country)", kk.assign(in_top5=kk.in_top5.astype(int))).fit(cov_type="cluster", cov_kwds={"groups": kk.country})
q2["n_countries"] = int(len(elig))
q2["log1p_dist_to_centroid_failed_minus_success"] = [round(float(r1.params["failed"]), 3)] + [round(float(x), 3) for x in r1.conf_int().loc["failed"]]
q2["top5_city_share_failed_minus_success_pp"] = [round(float(r2.params["failed"] * 100), 2)] + [round(float(x * 100), 2) for x in r2.conf_int().loc["failed"]]
per = []
for c in elig:
    g = kk[kk.country == c]
    per.append({"country": c, "n_fail": int(g.failed.sum()), "n_ok": int((1 - g.failed).sum()),
                "median_d_fail": round(float(g[g.failed == 1].d_cent.median()), 1),
                "median_d_ok": round(float(g[g.failed == 0].d_cent.median()), 1),
                "top5_fail_pct": round(float(g[g.failed == 1].in_top5.mean() * 100), 1),
                "top5_ok_pct": round(float(g[g.failed == 0].in_top5.mean() * 100), 1)})
q2["per_country"] = per
# like-for-like: institution point for BOTH failed and successful strings (same measurement on both sides)
kk2 = kk.copy()
c2 = kk2[kk2.failed == 0].groupby("country")[["inst_lat", "inst_lon"]].mean()
kk2 = kk2.merge(c2.rename(columns={"inst_lat": "ci_lat", "inst_lon": "ci_lon"}), left_on="country", right_index=True)
kk2["log_d_inst"] = np.log1p(hav(kk2.inst_lat, kk2.inst_lon, kk2.ci_lat, kk2.ci_lon))
r3 = smf.ols("log_d_inst ~ failed + C(country)", kk2).fit(cov_type="cluster", cov_kwds={"groups": kk2.country})
q2["like_for_like_inst_points_log1p_dist_failed_minus_success"] = [round(float(r3.params["failed"]), 3)] + [round(float(x), 3) for x in r3.conf_int().loc["failed"]]
q2["note"] = ("first contrast compares failed strings' institution points with successful strings' geocodes; "
              "the like-for-like contrast uses institution points on both sides")
out["Q2_within_country"] = q2

# ---------------------------------------------------------------- Q3 distance indicators
d = s[s.region.notna()].copy()
prop = smf.logit("success ~ C(region) + np.log(n) + html + commas + strlen",
                 d.assign(n=d.Raw_Affiliation.map(af.groupby("Raw_Affiliation").Paper_ID.nunique()),
                          html=d.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int),
                          commas=d.Raw_Affiliation.str.count(","), strlen=d.Raw_Affiliation.str.len(),
                          success=d.success.astype(int))).fit(disp=0)
ph = prop.predict(d.assign(n=d.Raw_Affiliation.map(af.groupby("Raw_Affiliation").Paper_ID.nunique()),
                           html=d.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int),
                           commas=d.Raw_Affiliation.str.count(","), strlen=d.Raw_Affiliation.str.len()))
wraw = np.where(d.success, 1 / ph, 0)
d["w"] = np.clip(wraw, None, np.quantile(wraw[d.success], 0.99))
d["cc_lat"] = np.where(d.success, d.Latitude, np.nan)
d["cc_lon"] = np.where(d.success, d.Longitude, np.nan)
d["nc_lat"] = np.where(d.success, d.Latitude, d.inst_lat)
d["nc_lon"] = np.where(d.success, d.Longitude, d.inst_lon)
d["cat_lat"], d["cat_lon"] = d.inst_lat, d.inst_lon
d["hy_lat"] = np.where(d.inst_lat.notna(), d.inst_lat, d.Latitude)
d["hy_lon"] = np.where(d.inst_lon.notna(), d.inst_lon, d.Longitude)
mem = af[["Paper_ID", "Raw_Affiliation"]].drop_duplicates().merge(
    d[["Raw_Affiliation", "country", "w", "cc_lat", "cc_lon", "nc_lat", "nc_lon", "cat_lat", "cat_lon", "hy_lat", "hy_lon"]],
    on="Raw_Affiliation")


def indicator(latc, lonc, weighted=False):
    sub = mem.dropna(subset=[latc, lonc])
    rows = []
    for pid, g in sub.groupby("Paper_ID"):
        if not (2 <= len(g) <= 60):
            continue
        vals = g[[latc, lonc, "country", "w"]].values
        for a, b in itertools.combinations(vals, 2):
            rows.append((a[0], a[1], b[0], b[1], a[2] != b[2], a[3] * b[3]))
    P = pd.DataFrame(rows, columns=["a1", "o1", "a2", "o2", "intl", "w"])
    P["dist"] = hav(P.a1, P.o1, P.a2, P.o2)
    wt = P.w if weighted else np.ones(len(P))

    def wq(x, w, qq):
        i = np.argsort(x.values)
        xs, ws = x.values[i], np.asarray(w)[i]
        return float(np.interp(qq, np.cumsum(ws) / ws.sum(), xs))
    return {"n_pairs": int(len(P)), "n_works": int(sub.Paper_ID.nunique()),
            "mean_km": round(float(np.average(P.dist, weights=wt)), 1),
            "median_km": round(wq(P.dist, wt, 0.5), 1),
            "share_over_1000km_pct": round(float(np.average(P.dist > 1000, weights=wt) * 100), 2),
            "intl_pct": round(float(np.average(P.intl, weights=wt) * 100), 2)}


out["Q3_distance_indicators"] = {
    "complete_case": indicator("cc_lat", "cc_lon"),
    "ipw": indicator("cc_lat", "cc_lon", weighted=True),
    "near_complete": indicator("nc_lat", "nc_lon"),
    "catalogue_only": indicator("cat_lat", "cat_lon"),
    "hybrid": indicator("hy_lat", "hy_lon"),
}
out["Q3_coverage"] = {"complete_case_strings": int(d.cc_lat.notna().sum()), "near_complete_strings": int(d.nc_lat.notna().sum()),
                      "catalogue_only_strings": int(d.cat_lat.notna().sum()), "hybrid_strings": int(d.hy_lat.notna().sum()),
                      "all_strings_with_region": int(len(d))}

json.dump(out, open("institution_coordinates.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "Q2_within_country"}, indent=1, default=str))
print(json.dumps({k: v for k, v in out["Q2_within_country"].items() if k != "per_country"}, indent=1))
