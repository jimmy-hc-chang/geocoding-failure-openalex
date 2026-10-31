#!/usr/bin/env python3
"""
20_make_figures.py  (revision version)

Regenerates Figures 1-3 of the manuscript from the deposited data and the
outputs of scripts 16 and 21, under the explicit tie rule of country_rule.py.

  Figure 1   figs/fig1_choropleth.png   country geocoding-failure choropleth
  Figure 2   figs/fig2_mechanism.png    comma gradient + HTML-entity scatter
  Figure 3   figs/fig3_ipw.png          composition distortion + validation

Inputs (working directory): affil_full.csv, geocoded.csv,
geo_mar_results.json (script 16), validation_ground_truth.json (script 21),
and a world GeoJSON. The map uses Natural Earth 1:110m admin-0 countries
(public domain, https://www.naturalearthdata.com/), file
ne_110m_admin_0_countries.geojson; pass another path with WORLD_GEOJSON.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_rule import assign_country, REGION  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
OUT = os.environ.get("FIG_DIR", "figs")
os.makedirs(OUT, exist_ok=True)
LA = "Latin America & Caribbean"

af = pd.read_csv("affil_full.csv", usecols=["Raw_Affiliation", "Institution_Countries"])
af = af[af.Raw_Affiliation.notna()]
country = assign_country(af)
geo = pd.read_csv("geocoded.csv")
d = geo.merge(country.rename("country"), left_on="Raw_Affiliation", right_index=True, how="left")
d["fail"] = d.Latitude.isna().astype(int)
d["n_commas"] = d.Raw_Affiliation.str.count(",")
d["html_ent"] = d.Raw_Affiliation.str.contains(r"&#x|&amp;|&\w+;", regex=True, na=False).astype(int)
d["region"] = d.country.map(REGION)
dr = d[d.region.notna()]

# ---------------------------------------------------------------- Figure 1
t = dr.groupby("country").agg(n=("fail", "size"), rate=("fail", "mean"))
t = t[t.n >= 30]
gj = json.load(open(os.environ.get("WORLD_GEOJSON", "ne_110m_admin_0_countries.geojson"), encoding="utf-8"))
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
cmap = plt.cm.YlOrRd
vmin, vmax = 0.05, 0.50
for f in gj["features"]:
    p = f["properties"]
    if p.get("ADM0_A3") == "ATA":
        continue
    a2 = p.get("ISO_A2_EH") if p.get("ISO_A2_EH") not in (None, "-99") else p.get("ISO_A2")
    r = t.rate.get(a2)
    color = "#d9d9d9" if r is None or pd.isna(r) else cmap((min(max(r, vmin), vmax) - vmin) / (vmax - vmin))
    g = f["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    for poly in polys:
        xy = np.array(poly[0])
        ax.fill(xy[:, 0], xy[:, 1], color=color, ec="white", lw=0.3, zorder=2)
ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); ax.set_aspect(1.2); ax.axis("off")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin * 100, vmax * 100))
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
cb.set_label("Geocoding failure rate (%)")
ax.legend(handles=[mpatches.Patch(color="#d9d9d9", label="< 30 strings / no data")],
          loc="lower left", frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_choropleth.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), dpi=300)
bins = [-1, 0, 1, 2, 3, 4, 6, 50]; labels = ["0", "1", "2", "3", "4", "5–6", "7+"]
dr2 = d.assign(cb=pd.cut(d.n_commas, bins=bins, labels=labels))  # all strings, as in the text
fr = dr2.groupby("cb", observed=True).fail.mean()
axes[0].bar(fr.index.astype(str), fr.values * 100, color="#00539b")
axes[0].set_xlabel("Commas in affiliation string (all strings)"); axes[0].set_ylabel("Failure rate (%)")
axes[0].set_title("(a) Failure by address structure", fontsize=10)
reg = dr.groupby("region").agg(ent=("html_ent", "mean"), fail=("fail", "mean"))
axes[1].scatter(reg.ent * 100, reg.fail * 100, s=30, color="#b3282d", zorder=3)
short = {LA: "Latin America", "Eastern Europe & Central Asia": "E. Europe & C. Asia",
         "Sub-Saharan Africa": "Sub-Saharan Africa", "Northern America": "N. America"}
# label positions (data coordinates) with leader lines, so that the low-entity cluster stays legible
pos = {LA: (10.6, 22.6), "North Africa": (6.5, 19.9), "Western Europe": (6.3, 16.6), "Middle East": (0.3, 19.3),
       "Southeast Asia": (3.7, 15.4), "South Asia": (0.3, 14.4), "Oceania": (0.3, 13.2),
       "Sub-Saharan Africa": (4.2, 13.0), "Eastern Europe & Central Asia": (0.3, 9.6),
       "East Asia": (4.2, 10.4), "Northern America": (3.4, 7.0)}
for name, row in reg.iterrows():
    axes[1].annotate(short.get(name, name), (row.ent * 100, row.fail * 100), xytext=pos[name],
                     textcoords="data", fontsize=7, va="center",
                     arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, shrinkA=0, shrinkB=3))
axes[1].set_xlim(0, 15)
axes[1].set_xlabel("Share of strings with unescaped HTML entities (%)")
axes[1].set_ylabel("Failure rate (%)")
axes[1].set_title("(b) Encoding artefacts vs failure, by region", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_mechanism.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 3
g = json.load(open("geo_mar_results.json"))
# composition from unrounded string-level shares (strings with a region)
true_comp = dr.region.value_counts(normalize=True)
obs_comp = dr[dr.fail == 0].region.value_counts(normalize=True)
under = ((obs_comp / true_comp - 1) * 100).sort_values()
v = json.load(open("validation_ground_truth.json"))
rows = {r["region"]: r for r in v["size_on_all"]["regions"]}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), dpi=300)
axes[0].barh(under.index, under.values, color=["#b3282d" if x < 0 else "#00539b" for x in under])
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlabel("Over/under-representation in geocoded sample (%)")
axes[0].set_title("(a) Composition distortion of the spatial sample", fontsize=10)
sel = [(LA, "Latin America\n& Caribbean"), ("North Africa", "North Africa"),
       ("Middle East", "Middle East"), ("Sub-Saharan Africa", "Sub-Saharan\nAfrica")]
x = np.arange(len(sel)); w = 0.27
for k, (key, lab, col) in enumerate([("observed", "Observed", "#c6d6e6"),
                                     ("ipw_kappa", "IPW-corrected", "#00539b"),
                                     ("full_data", "Full data (ground truth)", "#e08214")]):
    axes[1].bar(x + (k - 1) * w, [rows[r][key] for r, _ in sel], w, label=lab, color=col)
axes[1].set_xticks(x); axes[1].set_xticklabels([l for _, l in sel], fontsize=8)
axes[1].set_ylabel("Share of co-affiliation pairs (%)"); axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("(b) Regional involvement: correction vs ground truth", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_ipw.png", bbox_inches="tight"); plt.close()
print("figures written to", OUT)
