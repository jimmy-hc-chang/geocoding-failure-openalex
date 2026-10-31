#!/usr/bin/env python3
"""
22_country_discordance.py  (revision, Reviewer 2 Major 3; Reviewer 2 minor 5; Reviewer 1 comment 4)

Part A — country discordance among SUCCESSFUL geocodes.
  The geocoded point is reverse-geocoded offline (reverse_geocoder, GeoNames cities >1000)
  and compared with the string's catalogue country. Reported by region, comma count,
  HTML entity, string length, and resolution status; the largest directed flows
  (catalogue -> geocoded); and sensitivity of the distance and pair indicators to
  excluding discordant endpoints.
  Note: the original cascade cache stored coordinates only, not the cascade variant
  that produced them, so discordance by cascade step requires a re-query
  (see 22b_cascade_step_requery.py, run locally).

Part B — tied strings (catalogue country not unique).
  Agreement between the geocoded country and each of the tied catalogue countries,
  as evidence of which tied assignment is the mis-match.

Part C — samples for hand classification (written as .xlsx):
  discordance_sample_100.xlsx : 100 discordant strings from the largest gross flows
  tie_sample_100.xlsx         : 100 tied strings

Run from full_run/:  TIE_RULE=exclude python ../analysis/22_country_discordance.py
Outputs: country_discordance.json, discordance_by_*.csv, the two .xlsx samples
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import reverse_geocoder as rg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_rule import assign_country, TIE_RULE, REGION  # noqa: E402

SEED = 20260811
af = pd.read_csv("affil_full.csv", usecols=["Paper_ID", "Raw_Affiliation", "Institution_Countries", "Standardized_Institutions"])
af = af[af.Raw_Affiliation.notna()]
country = assign_country(af)
geo = pd.read_csv("geocoded.csv")
m = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True, how="left")
m["success"] = m.Latitude.notna()
m["n_commas"] = m.Raw_Affiliation.str.count(",")
m["comma_bin"] = pd.cut(m.n_commas, [-1, 0, 1, 2, 3, 99], labels=["0", "1", "2", "3", "4+"])
m["strlen"] = m.Raw_Affiliation.str.len()
m["len_bin"] = pd.qcut(m.strlen, 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
m["html_ent"] = m.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False)
res_share = af.assign(has=af.Standardized_Institutions.notna()).groupby("Raw_Affiliation").has.mean()
m["resolved"] = m.Raw_Affiliation.map(res_share).fillna(0) > 0
m["region"] = m.country.map(REGION)

ok = m[m.success].copy()
r = rg.search(list(zip(ok.Latitude, ok.Longitude)), mode=1)
ok["geo_cc"] = [x["cc"] for x in r]
ok["geo_admin1"] = [x["admin1"] for x in r]
ok["geo_name"] = [x["name"] for x in r]
k = ok[ok.country.notna()].copy()
k["disc"] = (k.geo_cc != k.country)
out = {"tie_rule_for_catalogue_country": TIE_RULE, "n_success_with_country": int(len(k)),
       "agreement_pct": round(float((~k.disc).mean() * 100), 2),
       "discordance_pct": round(float(k.disc.mean() * 100), 2)}


def tab(col):
    t = k.groupby(col, observed=True).agg(n=("disc", "size"), discordant=("disc", "sum"))
    t["disc_pct"] = (t.discordant / t.n * 100).round(1)
    return t


for col in ["region", "comma_bin", "len_bin", "html_ent", "resolved"]:
    t = tab(col)
    t.to_csv(f"discordance_by_{col}.csv")
    out[f"by_{col}"] = {str(i): {"n": int(v.n), "disc_pct": float(v.disc_pct)} for i, v in t.iterrows()}

# region x comma (<=1 vs >=2)
k["c2"] = np.where(k.n_commas <= 1, "<=1", ">=2")
rc = k.groupby(["region", "c2"]).disc.agg(["size", "mean"]).reset_index()
rc["mean"] = (rc["mean"] * 100).round(1)
out["by_region_x_comma"] = {f"{a}|{b}": {"n": int(n), "disc_pct": float(p)} for a, b, n, p in rc.itertuples(index=False)}

flows = k[k.disc].groupby(["country", "geo_cc"]).size().sort_values(ascending=False)
out["top_flows"] = [{"catalogue": a, "geocoded": b, "n": int(n)} for (a, b), n in flows.head(30).items()]

# adjacency (border/nearest-place effects): land neighbours + SAR pairs, coarse list
NEAR = {frozenset(p) for p in [("SG", "MY"), ("CN", "MO"), ("CN", "HK"), ("HK", "MO"), ("US", "CA"), ("US", "MX"),
                               ("FR", "CH"), ("FR", "BE"), ("FR", "DE"), ("DE", "NL"), ("DE", "AT"), ("DE", "CH"),
                               ("DE", "DK"), ("DE", "PL"), ("DE", "CZ"), ("BE", "NL"), ("AT", "CH"), ("IT", "CH"),
                               ("ES", "PT"), ("ES", "FR"), ("GB", "IE"), ("SE", "DK"), ("SE", "NO"), ("FI", "SE"),
                               ("CZ", "SK"), ("AT", "SK"), ("AT", "HU"), ("SI", "HR"), ("IN", "NP"), ("IN", "BD"),
                               ("IN", "PK"), ("CN", "KP"), ("KR", "KP"), ("MY", "TH"), ("MY", "ID"), ("SA", "AE"),
                               ("QA", "SA"), ("AE", "OM"), ("JO", "IL"), ("IL", "PS"), ("EG", "IL"), ("TR", "GR"),
                               ("TR", "CY"), ("CY", "GR"), ("RU", "BY"), ("RU", "UA"), ("RU", "KZ"), ("LU", "BE"),
                               ("LU", "FR"), ("LU", "DE"), ("VA", "IT"), ("SM", "IT"), ("MC", "FR"), ("LI", "CH")]}
k["adjacent"] = [frozenset((a, b)) in NEAR for a, b in zip(k.country, k.geo_cc)]
dd = k[k.disc]
out["discordant_adjacent_pct"] = round(float(dd.adjacent.mean() * 100), 1)
out["discordance_excl_adjacent_pct"] = round(float((k.disc & ~k.adjacent).mean() * 100), 2)
reg_nonadj = k.assign(gross=k.disc & ~k.adjacent).groupby("region").gross.mean().mul(100).round(1)
out["gross_discordance_by_region_pct"] = reg_nonadj.to_dict()

# ---------------------------------------------------------------- sensitivity: distance and pairs
pairs = af[["Paper_ID", "Raw_Affiliation"]].drop_duplicates().merge(
    k[["Raw_Affiliation", "Latitude", "Longitude", "disc", "region", "country"]], on="Raw_Affiliation")


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def pair_stats(df):
    rows = []
    for pid, g in df.groupby("Paper_ID"):
        if not (2 <= len(g) <= 60):
            continue
        v = g[["Latitude", "Longitude", "region", "country"]].values
        for a, b in itertools.combinations(v, 2):
            rows.append((a[0], a[1], b[0], b[1], a[2], b[2], a[3] != b[3]))
    P = pd.DataFrame(rows, columns=["la1", "lo1", "la2", "lo2", "r1", "r2", "intl"])
    P["dist"] = hav(P.la1, P.lo1, P.la2, P.lo2)
    la = ((P.r1 == "Latin America & Caribbean") | (P.r2 == "Latin America & Caribbean")).mean() * 100
    return {"n_pairs": int(len(P)), "mean_dist_km": round(float(P.dist.mean()), 1),
            "median_dist_km": round(float(P.dist.median()), 1), "intl_share_pct": round(float(P.intl.mean() * 100), 2),
            "LA_share_pct": round(float(la), 3)}


out["sensitivity"] = {"all_geocoded": pair_stats(pairs),
                      "excluding_discordant": pair_stats(pairs[~pairs.disc]),
                      "excluding_gross_discordant": pair_stats(pairs[~(pairs.Raw_Affiliation.isin(k[k.disc & ~k.adjacent].Raw_Affiliation))])}

# ---------------------------------------------------------------- Part B: tied strings
cc = af[["Raw_Affiliation", "Institution_Countries"]].dropna()
cc = cc.assign(c=cc.Institution_Countries.str.split(",")).explode("c")
cc["c"] = cc.c.str.strip()
vc = cc.groupby(["Raw_Affiliation", "c"]).size().reset_index(name="n")
top = vc.groupby("Raw_Affiliation").n.transform("max")
best = vc[vc.n == top]
tied = best.groupby("Raw_Affiliation").c.agg(list)
tied = tied[tied.str.len() > 1]
tok = ok[ok.Raw_Affiliation.isin(tied.index)].copy()
tok["tied_set"] = tok.Raw_Affiliation.map(tied)
tok["geo_in_tied"] = [g in s for g, s in zip(tok.geo_cc, tok.tied_set)]
out["tied"] = {"n_tied": int(len(tied)), "n_tied_geocoded": int(len(tok)),
               "geocoded_country_is_one_of_tied_pct": round(float(tok.geo_in_tied.mean() * 100), 1),
               "geocode_fail_rate_tied_pct": round(float(1 - m[m.Raw_Affiliation.isin(tied.index)].success.mean()) * 100, 1),
               "geocode_fail_rate_untied_pct": round(float(1 - m[~m.Raw_Affiliation.isin(tied.index)].success.mean()) * 100, 1)}
# how often the "losing" tied country is the mis-match, judged by the geocode
inst = af.dropna(subset=["Standardized_Institutions"]).groupby("Raw_Affiliation").Standardized_Institutions.first()

# ---------------------------------------------------------------- Part C: samples for hand classification
rng = np.random.default_rng(SEED)
gross = k[k.disc & ~k.adjacent].copy()
topflows = [(a, b) for (a, b) in flows.index if frozenset((a, b)) not in NEAR][:12]
gross["flow"] = list(zip(gross.country, gross.geo_cc))
cand = gross[gross.flow.isin(topflows)]
samp = cand.sample(min(100, len(cand)), random_state=SEED)
samp = samp.assign(institution=samp.Raw_Affiliation.map(inst))
cols = ["Raw_Affiliation", "institution", "country", "geo_cc", "geo_name", "geo_admin1", "Latitude", "Longitude", "n_commas"]
S = samp[cols].rename(columns={"country": "catalogue_country", "geo_cc": "geocoded_country"}).reset_index(drop=True)
S.insert(0, "id", [f"DC{i+1:03d}" for i in range(len(S))])
S["who_is_wrong"] = ""      # G = geocoder misplaced | C = catalogue country wrong | B = string names two countries | X = other/unclear
S["notes"] = ""
tsamp = tied.sample(min(100, len(tied)), random_state=SEED)
T = pd.DataFrame({"Raw_Affiliation": tsamp.index, "tied_countries": tsamp.map(lambda s: ", ".join(s)).values})
inst_all = af.dropna(subset=["Standardized_Institutions"]).groupby("Raw_Affiliation").Standardized_Institutions.agg(lambda s: " || ".join(sorted(set(s))))
T["openalex_institutions"] = T.Raw_Affiliation.map(inst_all)
T["geocoded_country"] = T.Raw_Affiliation.map(ok.set_index("Raw_Affiliation").geo_cc)
T.insert(0, "id", [f"TI{i+1:03d}" for i in range(len(T))])
T["classification"] = ""    # M = one institution is a mis-match | L = both legitimate (string names both) | U = unclear
T["correct_country"] = ""   # ISO-2 of the country the string actually refers to
T["notes"] = ""

from openpyxl.styles import Alignment, PatternFill, Font  # noqa: E402
for df, name, fill_cols, guide in [
    (S, "discordance_sample_100.xlsx", ["who_is_wrong", "notes"],
     ["國家不一致抽樣(對應 R2 Major 3)",
      "每筆是「地理編碼成功,但編碼點所在國家 ≠ OpenAlex catalogue 國家」的字串,抽自最大的幾個非鄰國流向(如 US→CN)。",
      "who_is_wrong 填一個字母:",
      "  G = 編碼器放錯地方(字串明顯在 catalogue 國家,編碼點跑到別國)",
      "  C = catalogue 國家錯(字串明顯在編碼國家,OpenAlex 指派錯國家的機構)",
      "  B = 字串本身提到兩個國家(兩邊都對一半)",
      "  X = 其他/無法判斷",
      "判斷依據:只看 Raw_Affiliation 字串本身寫的地名;institution 欄是 OpenAlex 指派的機構,供參考。"]),
    (T, "tie_sample_100.xlsx", ["classification", "correct_country", "notes"],
     ["眾數國家平手抽樣(對應 R2 minor 5、R1 comment 4)",
      "每筆字串被 OpenAlex 指派了兩個(或以上)位於不同國家的機構,且各國出現次數相同。",
      "classification 填一個字母:",
      "  M = 其中一個機構是誤配(字串只指一個機構,另一個是 OpenAlex 配錯)",
      "  L = 兩個都合理(字串本身真的寫了兩個機構)",
      "  U = 無法判斷",
      "correct_country:字串實際所在國家的 ISO-2 代碼(如 RU、IN);L 或 U 可留空。"])]:
    with pd.ExcelWriter(name, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="classify")
        ws = xw.sheets["classify"]
        ws.freeze_panes = "B2"
        for c in ws[1]:
            c.font = Font(bold=True)
        hdr = [c.value for c in ws[1]]
        for i, h in enumerate(hdr):
            letter = ws.cell(row=1, column=i + 1).column_letter
            ws.column_dimensions[letter].width = 70 if h in ("Raw_Affiliation", "openalex_institutions", "institution") else 14
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if hdr[c.column - 1] in ("Raw_Affiliation", "openalex_institutions", "institution"):
                    c.alignment = Alignment(wrap_text=True, vertical="top")
                if hdr[c.column - 1] in fill_cols:
                    c.fill = PatternFill("solid", fgColor="FFF2CC")
        g = xw.book.create_sheet("說明")
        for i, l in enumerate(guide, 1):
            g.cell(row=i, column=1, value=l)
        g.column_dimensions["A"].width = 120

json.dump(out, open("country_discordance.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({x: out[x] for x in ["n_success_with_country", "agreement_pct", "discordance_pct",
                                      "discordant_adjacent_pct", "discordance_excl_adjacent_pct", "tied"]}, indent=1))
print("by region:", {a: b["disc_pct"] for a, b in out["by_region"].items()})
print("by comma:", {a: b["disc_pct"] for a, b in out["by_comma_bin"].items()})
print("gross by region:", out["gross_discordance_by_region_pct"])
print("top flows:", out["top_flows"][:12])
print("sensitivity:", json.dumps(out["sensitivity"], indent=1))
