"""
country_rule.py — one place for the string -> catalogue-country assignment.

A raw affiliation string can appear in many affiliation rows, and each row's
Institution_Countries may list several countries. The string's catalogue country
is its modal country across rows. About 2.8% of strings have a TIED mode; almost
all are strings to which OpenAlex attached two institutions in different
countries, one of which is often a mis-match (e.g. Ioffe Institute + IMEC).

TIE_RULE (environment variable) selects how ties are handled:
  exclude      (default, primary)  tied strings get no country -> "catalogue country unknown"
  first_listed  on a tie, the country that appears first in the Institution_Countries
                field of the string's earliest row (file order)
  stable        stable sort on n (deterministic given file order; tie -> first in groupby order,
                i.e. alphabetical country code)
  quicksort     the original submission's call (sort_values default); environment-dependent on ties
"""
import os
import pandas as pd

TIE_RULE = os.environ.get("TIE_RULE", "exclude")


def assign_country(af: pd.DataFrame, rule: str = None) -> pd.Series:
    rule = rule or TIE_RULE
    cc = af[["Raw_Affiliation", "Institution_Countries"]].dropna().copy()
    cc["row"] = range(len(cc))
    cc = cc.assign(country=cc.Institution_Countries.str.split(",")).explode("country")
    cc["country"] = cc.country.str.strip()
    cc["pos"] = cc.groupby("row").cumcount()
    vc = cc.groupby(["Raw_Affiliation", "country"]).agg(n=("row", "size"), first_row=("row", "min"),
                                                         first_pos=("pos", "min")).reset_index()
    if rule == "quicksort":
        vc = vc.sort_values("n", ascending=False).drop_duplicates("Raw_Affiliation")
        return vc.set_index("Raw_Affiliation")["country"]
    if rule == "stable":
        vc = vc.sort_values("n", ascending=False, kind="stable").drop_duplicates("Raw_Affiliation")
        return vc.set_index("Raw_Affiliation")["country"]
    top = vc.groupby("Raw_Affiliation").n.transform("max")
    best = vc[vc.n == top]
    if rule == "exclude":
        k = best.groupby("Raw_Affiliation").country.transform("size")
        return best[k == 1].set_index("Raw_Affiliation")["country"]
    if rule == "first_listed":
        best = best.sort_values(["Raw_Affiliation", "first_row", "first_pos"], kind="stable")
        return best.drop_duplicates("Raw_Affiliation").set_index("Raw_Affiliation")["country"]
    raise ValueError(rule)


def tie_diagnostics(af: pd.DataFrame) -> dict:
    cc = af[["Raw_Affiliation", "Institution_Countries"]].dropna()
    cc = cc.assign(country=cc.Institution_Countries.str.split(",")).explode("country")
    cc["country"] = cc.country.str.strip()
    vc = cc.groupby(["Raw_Affiliation", "country"]).size().reset_index(name="n")
    top = vc.groupby("Raw_Affiliation").n.transform("max")
    best = vc[vc.n == top]
    k = best.groupby("Raw_Affiliation").size()
    return {"n_strings_with_country": int(k.size), "n_tied": int((k > 1).sum()),
            "tied_pct": round(float((k > 1).mean() * 100), 2),
            "tie_sizes": {int(a): int(b) for a, b in (k[k > 1].value_counts().sort_index()).items()}}


# ---------------------------------------------------------------- regional scheme (identical to scripts 16-21)
REGION = {}
for _c in "US CA".split(): REGION[_c] = "Northern America"
for _c in "MX GT HN SV NI CR PA CU DO HT JM TT BB BS PR BR AR CL CO PE EC VE UY PY BO GY SR".split(): REGION[_c] = "Latin America & Caribbean"
for _c in "DE FR IT GB ES PT NL BE CH AT DK SE NO FI IS IE LU MT CY GR".split(): REGION[_c] = "Western Europe"
for _c in "PL CZ SK HU RO BG HR SI EE LV LT RS ME MK BA AL UA BY MD RU KZ UZ TJ TM KG AZ AM GE XK".split(): REGION[_c] = "Eastern Europe & Central Asia"
for _c in "CN JP KR TW HK MO MN KP".split(): REGION[_c] = "East Asia"
for _c in "IN PK BD LK NP AF BT MV".split(): REGION[_c] = "South Asia"
for _c in "SG MY TH VN ID PH BN KH LA MM TL".split(): REGION[_c] = "Southeast Asia"
for _c in "IR IQ SA AE QA KW OM BH YE JO LB SY IL PS TR".split(): REGION[_c] = "Middle East"
for _c in "EG DZ TN MA LY SD MR".split(): REGION[_c] = "North Africa"
for _c in "NG ZA KE GH TZ MZ BW ZW CM SN BJ BF TD SS ET MU UG RW ZM MW AO CD CG GA CI ML NE SO SZ TG MG".split(): REGION[_c] = "Sub-Saharan Africa"
for _c in "AU NZ FJ PG".split(): REGION[_c] = "Oceania"
