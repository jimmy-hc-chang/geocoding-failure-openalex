# geocoding-failure-openalex

Data and code for the manuscript:

> **The geography of geocoding failure: non-random missingness in open-infrastructure
> spatial scientometrics, and the limits of its correction by inverse-probability weighting**
> (Hung-Chi Chang, submitted to *Journal of Informetrics*, 2026)

The paper audits geocoding failure (Nominatim/OpenStreetMap) and institution
non-resolution (OpenAlex) across all 80,176 unique affiliation strings of a complete
OpenAlex field corpus, diagnoses the missingness as MAR given observables, and
corrects downstream spatial statistics by inverse-probability weighting, with a
mechanism replication in a second field (CAR-T cell therapy).

## Relationship to a companion project

The raw OpenAlex extraction (and one descriptive failure-by-country tabulation) is
shared with a companion manuscript on science-park location and network centrality,
deposited at <https://github.com/tiger-lab-png/scpark-network>. All research
questions and inferential analyses are disjoint; the present repository hosts the
full corpora and every script behind the geocoding-failure manuscript.

## Data

All CSV files are UTF-8 (`utf-8-sig`). Files over 3 MB are stored gzipped.

| Directory | Contents |
|---|---|
| `data/full_run/` | Silicon-carbide corpus (OpenAlex Topic T10361, 2018–2024): 54,671 works, 319,968 affiliation rows, 80,176 unique raw strings; Nominatim geocoding results and caches (retrieved 2026-07, 1 req/s); Overpass/Wikidata provenance files. |
| `data/cart_run/` | CAR-T cell-therapy replication corpus (capped sample): 36,892 unique strings, same schema. |

See the data dictionary comments at the top of each analysis script for
column-level documentation.

## Reproducing the manuscript

Run from `data/full_run/` (with `data/cart_run/` as a sibling directory), in order:

| Script | Reproduces |
|---|---|
| `analysis/16_geocoding_mar_analysis.py` | Descriptive failure geography (Table 1, Fig. 1), MCAR test, logistic models M1–M3 (Tables 2–3, Fig. 2a), IPW correction and pair statistics (Table 4, Fig. 4), entity-resolution audit (Table 5, Fig. 2b) |
| `analysis/17_reviewer_diagnostics.py` | CAR-T replication (Section 4.6), weight diagnostics/ESS/calibration, within-work dependence (κ), Wilson CIs, unknown-country bounds, MNAR tipping point |
| `analysis/18_minor_revision_checks.py` | Propensity-model robustness battery (interactions, trimming sweep, stabilized weights), mechanism decomposition (structure-side vs place-level failures), reverse-geocode reclamation, 2,000-rep bootstrap on the augmented sample |
| `analysis/19_presubmission_pass.py` | Country-fixed-effects propensity specification with per-country calibration; primary 2,000-replicate cluster bootstrap with the propensity model re-fit inside every replicate |
| `figures/20_make_figures.py` | All four manuscript figures, regenerated deterministically from the deposited data (requires `countries.geo.json`, e.g. from github.com/johan/world.geo.json, in the working directory) |

Precomputed outputs are in `results/`. All random seeds are fixed module
constants (20260811); no script requires new API calls.

Analysis scripts were drafted with LLM assistance and verified by the author
against the deposited data; every reported number, table, and figure in the
manuscript is deterministic and fully reproducible from this repository.

## Environment

Python ≥ 3.10; `pip install -r requirements.txt`
(pandas, numpy, scipy, statsmodels, patsy, matplotlib, pycountry, reverse_geocoder).

## License

MIT (code). Data are derived from OpenAlex (CC0), OpenStreetMap/Nominatim
(ODbL), and Wikidata (CC0); attribution notices apply per source.
