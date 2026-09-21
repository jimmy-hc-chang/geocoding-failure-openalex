# geocoding-failure-openalex

Data and code for the manuscript:

> **The geography of geocoding failure: non-random missingness in open-infrastructure
> spatial scientometrics, and the limits of its correction by inverse-probability weighting**
> (Hung-Chi Chang, Short Communication under review at *Journal of Informetrics*, 2026)

The paper audits geocoding failure (Nominatim/OpenStreetMap) and institution
non-resolution (OpenAlex) across all 80,176 unique affiliation strings of a complete
OpenAlex field corpus, diagnoses the missingness as MAR given observables, and
corrects downstream spatial statistics by inverse-probability weighting, with a
mechanism replication in a second field (CAR-T cell therapy).

Because the manuscript is a Short Communication, most of the supporting evidence is
reported in its Supplementary Material rather than the main text. The mapping below
therefore names both. "S" numbers refer to sections, tables and figures of the
Supplementary Material.

## Relationship to a companion project

The raw OpenAlex extraction (and one descriptive failure-by-country tabulation) is
shared with a companion manuscript on science-park location and network centrality,
deposited at <https://github.com/tiger-lab-png/scpark-network>. All research
questions and inferential analyses are disjoint; the present repository hosts the
full corpora and every script behind the geocoding-failure manuscript.

## Repository layout

```
full_run/     silicon-carbide corpus, geocoding results and caches
cart_run/     CAR-T replication corpus, same schema
analysis/     16-19: primary analysis, diagnostics, robustness, final checks
figures/      20: regenerates every figure from the deposited data
results/      precomputed outputs (JSON/CSV)
```

## Data

All CSV files are UTF-8 (`utf-8-sig`). Files over 3 MB are stored gzipped;
`full_run/affil_full.csv.gz` is split into three parts, with reassembly
instructions and an MD5 checksum in `full_run/REASSEMBLE.md`.

| Directory   | Contents |
|---|---|
| `full_run/` | Silicon-carbide corpus (OpenAlex Topic T10361, 2018–2024): 54,671 works, 319,968 affiliation rows, 80,176 unique raw strings. Nominatim geocoding results and caches, retrieved July 2026 at 1 request/second against the public API. Overpass and Wikidata provenance files. |
| `cart_run/` | CAR-T cell-therapy replication corpus: 36,892 unique raw strings, same schema, retrieved with the identical pipeline. |

Column-level documentation is in the data-dictionary comments at the top of each
analysis script.

## Reproducing the manuscript

Run from `full_run/`, with `cart_run/` as a sibling directory, in this order:

```
cd full_run
python ../analysis/16_geocoding_mar_analysis.py
python ../analysis/17_diagnostics_and_replication.py
python ../analysis/18_robustness_checks.py
python ../analysis/19_presubmission_pass.py
python ../figures/20_make_figures.py
```

| Script | Reproduces |
|---|---|
| `analysis/16_geocoding_mar_analysis.py` | Failure geography (Table 1, Figure 1); country × failure contingency test; logistic models M1–M3 (Table S1) and the regional model M2 (Table 2); IPW correction and within-paper pair statistics (Table 3, Figure 3); entity-resolution audit (Table S3, Figure S1) |
| `analysis/17_diagnostics_and_replication.py` | CAR-T replication (Section 3.5, Table S2); weight diagnostics, effective sample size and calibration (Section S3); within-work dependence of endpoint selection (Section S4.3); Wilson intervals for Table 1; bounds for strings without catalogue country (Section S6.4); sensitivity to the magnitude of the modelled region effect (Section S4.4) |
| `analysis/18_robustness_checks.py` | Propensity-specification battery and stabilized weights (Section S4.1); trimming sweep (Section S4.2); decomposition of failures into delimiter-poor and place-level cases (Section 3.3); reverse-geocode reclamation and the augmented-sample bootstrap (Section S4.5); HTML-entity decode counterfactual (Section S4.6) |
| `analysis/19_presubmission_pass.py` | Country-fixed-effects propensity specification with per-country calibration (Section S4.1 and the corresponding row of Table 3); primary 2,000-replicate cluster bootstrap with the propensity model re-fit inside every replicate (Table 3) |
| `figures/20_make_figures.py` | Figures 1–3 of the main text and Supplementary Figure S1, regenerated deterministically from the deposited data |

Precomputed outputs are in `results/`. All random seeds are fixed module constants
(20260811); no script requires new API calls.

`20_make_figures.py` needs one file that is not redistributed here: a world-country
GeoJSON at `countries.geo.json` in the working directory, for the choropleth in
Figure 1. Any equivalent country-boundary GeoJSON works; the version used was
`countries.geo.json` from <https://github.com/johan/world.geo.json>. Every other
output requires nothing beyond this repository.

Analysis scripts were drafted with LLM assistance and verified by the author
against the deposited data; every reported number, table, and figure in the
manuscript is deterministic and fully reproducible from this repository.

## Citing this deposit

Release `v1.0-joi-submission` archives the version of the data and code used for
the manuscript. See `CITATION.cff`, or use GitHub's "Cite this repository".

## Environment

Python ≥ 3.10; `pip install -r requirements.txt`
(pandas, numpy, scipy, statsmodels, patsy, matplotlib, pycountry, reverse_geocoder).

## License

MIT (code). Data are derived from OpenAlex (CC0), OpenStreetMap/Nominatim
(ODbL), and Wikidata (CC0); attribution notices apply per source.
