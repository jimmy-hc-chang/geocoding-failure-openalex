#!/usr/bin/env python3
"""
20_make_figures.py

Regenerates every figure in the manuscript directly from the deposited data.
No figure element is drawn from anything other than the computed statistics;
every panel is a deterministic matplotlib rendering of numbers reproducible
via scripts 16-19.

  Figure 1   figs/fig1_choropleth.png   country geocoding-failure choropleth
  Figure 2   figs/fig2_mechanism.png    comma gradient + HTML-entity scatter
  Figure 3   figs/fig3_ipw.png          composition distortion + IPW correction
  Figure S1  figs/figS1_forest.png      regional odds ratios, both layers

Run from full_run/ AFTER 16_geocoding_mar_analysis.py, which writes
country_failure_table.csv, or_geocode.csv, or_resolution.csv and
ipw_metrics.csv; string-level inputs are rebuilt here from the raw deposits.

World geometry for Figure 1: place countries.geo.json (a public-domain world
GeoJSON with ISO-3166 alpha-3 feature ids, e.g. from
https://github.com/johan/world.geo.json) in the working directory. It is the
only input not redistributed in this repository.

Environment used for the manuscript versions: Python 3.10, matplotlib 3.x,
pandas, numpy, pycountry. All colours are colourblind-safe single-hue ramps
(YlOrRd) or a two-colour palette (#00539b / #b3282d).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd
import pycountry

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
import os
os.makedirs("figs", exist_ok=True)

LA = "Latin America & Caribbean"
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

# ------------------------------------------------ shared string-level frame
af = pd.read_csv("affil_full.csv", usecols=["Raw_Affiliation", "Institution_Countries"])
af = af[af.Raw_Affiliation.notna()]
cc2 = af[["Raw_Affiliation", "Institution_Countries"]].dropna()
cc2 = cc2.assign(country=cc2.Institution_Countries.str.split(",")).explode("country")
cc2["country"] = cc2.country.str.strip()
vc = (cc2.groupby(["Raw_Affiliation", "country"]).size().reset_index(name="n")
         .sort_values("n", ascending=False).drop_duplicates("Raw_Affiliation"))
country = vc.set_index("Raw_Affiliation")["country"]
geo = pd.read_csv("geocoded.csv")
d = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True, how="left")
d["success"] = d.Latitude.notna().astype(int)
d["fail"] = 1 - d.success
d["n_commas"] = d.Raw_Affiliation.str.count(",")
d["html_ent"] = d.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int)
d["region"] = d.country.map(REGION)
dr = d[d.region.notna()]

# ------------------------------------------------ Figure 1: choropleth
t = dr.groupby("country").agg(n=("fail", "size"), fail_rate=("fail", "mean"))
t = t[t.n >= 30]
iso23 = {c.alpha_2: c.alpha_3 for c in pycountry.countries}
rate = {iso23.get(k): v for k, v in t.fail_rate.items() if iso23.get(k)}
gj = json.load(open("countries.geo.json"))
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
cmap = plt.cm.YlOrRd
vmin, vmax = 0.05, 0.50
for f in gj["features"]:
    iso3 = f["id"]
    if iso3 == "ATA":
        continue
    r = rate.get(iso3)
    color = "#d9d9d9" if r is None else cmap((min(max(r, vmin), vmax) - vmin) / (vmax - vmin))
    g = f["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    for poly in polys:
        xy = np.array(poly[0])
        ax.fill(xy[:, 0], xy[:, 1], color=color, ec="white", lw=0.3, zorder=2)
ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); ax.set_aspect(1.2); ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin * 100, vmax * 100))
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
cb.set_label("Geocoding failure rate (%)")
ax.set_title("Nominatim geocoding failure rate by country (n ≥ 30 affiliation strings)", fontsize=11)
ax.legend(handles=[mpatches.Patch(color="#d9d9d9", label="< 30 strings / no data")],
          loc="lower left", frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig1_choropleth.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------ Figure 2: forest plots (both layers)
o1 = pd.read_csv("or_geocode.csv", index_col=0).drop(["Intercept", "log_papers"], errors="ignore")
o2 = pd.read_csv("or_resolution.csv", index_col=0).drop(["Intercept", "log_papers"], errors="ignore")
order = o1.OR.sort_values().index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=300, sharey=True)
for ax, o, title in [(axes[0], o1, "(a) Geocoding failure (Nominatim)"),
                     (axes[1], o2, "(b) Institution non-resolution (OpenAlex)")]:
    o = o.reindex(order)
    ypos = np.arange(len(o))
    ax.errorbar(o.OR, ypos, xerr=[o.OR - o.lo, o.hi - o.OR], fmt="o",
                color="#00539b", ecolor="#00539b", capsize=2, ms=4)
    ax.axvline(1, color="grey", ls="--", lw=0.8)
    ax.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 4]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    fmt = ScalarFormatter(); fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Odds ratio vs. Northern America (log scale)")
    ax.set_yticks(ypos); ax.set_yticklabels(o.index)
    ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.savefig("figs/figS1_forest.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------ Figure 3: mechanisms
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=300)
bins = [-1, 0, 1, 2, 3, 4, 6, 50]; labels = ["0", "1", "2", "3", "4", "5–6", "7+"]
dr2 = dr.copy(); dr2["cb"] = pd.cut(dr2.n_commas, bins=bins, labels=labels)
fr = dr2.groupby("cb", observed=True).fail.mean()
axes[0].bar(fr.index.astype(str), fr.values * 100, color="#00539b")
axes[0].set_xlabel("Commas in affiliation string"); axes[0].set_ylabel("Failure rate (%)")
axes[0].set_title("(a) Failure by address structure", fontsize=10)
regions = dr.groupby("region").agg(ent=("html_ent", "mean"), fail=("fail", "mean"))
axes[1].scatter(regions.ent * 100, regions.fail * 100, s=30, color="#b3282d")
for name, row in regions.iterrows():
    axes[1].annotate(name.replace(" & Caribbean", "").replace(" & Central Asia", ""),
                     (row.ent * 100, row.fail * 100), fontsize=7, xytext=(3, 3),
                     textcoords="offset points")
axes[1].set_xlabel("Share of strings with unescaped HTML entities (%)")
axes[1].set_ylabel("Failure rate (%)")
axes[1].set_title("(b) Encoding artefacts vs failure, by region", fontsize=10)
plt.tight_layout(); plt.savefig("figs/fig2_mechanism.png", bbox_inches="tight"); plt.close()

# ------------------------------------------------ Figure 4: composition + IPW
true_comp = dr.region.value_counts(normalize=True)
geo_comp = dr[dr.success == 1].region.value_counts(normalize=True)
under = ((geo_comp / true_comp - 1) * 100).sort_values()
ipw = pd.read_csv("ipw_metrics.csv", index_col=0).iloc[:, 0]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=300)
axes[0].barh(under.index, under.values,
             color=["#b3282d" if v < 0 else "#00539b" for v in under])
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlabel("Over/under-representation in geocoded sample (%)")
axes[0].set_title("(a) Composition distortion of the spatial sample", fontsize=10)
pairs = {
    "Latin America\n& Caribbean": (ipw["invshare_raw::Latin America & Caribbean"] * 100,
                                    ipw["invshare_ipw::Latin America & Caribbean"] * 100),
    "North Africa": (ipw["invshare_raw::North Africa"] * 100, ipw["invshare_ipw::North Africa"] * 100),
    "Middle East": (ipw["invshare_raw::Middle East"] * 100, ipw["invshare_ipw::Middle East"] * 100),
    "Sub-Saharan\nAfrica": (ipw["invshare_raw::Sub-Saharan Africa"] * 100,
                             ipw["invshare_ipw::Sub-Saharan Africa"] * 100),
}
x = np.arange(len(pairs)); w = 0.35
axes[1].bar(x - w / 2, [v[0] for v in pairs.values()], w, label="Observed", color="#9bb8d3")
axes[1].bar(x + w / 2, [v[1] for v in pairs.values()], w, label="IPW-corrected", color="#00539b")
axes[1].set_xticks(x); axes[1].set_xticklabels(pairs.keys(), fontsize=8)
axes[1].set_ylabel("Share of collaboration pairs (%)"); axes[1].legend(frameon=False)
axes[1].set_title("(b) Region involvement in co-affiliation pairs", fontsize=10)
plt.tight_layout(); plt.savefig("figs/fig3_ipw.png", bbox_inches="tight"); plt.close()

print("figures written to figs/")
