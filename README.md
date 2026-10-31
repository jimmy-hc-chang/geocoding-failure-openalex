# geocoding-failure-openalex

Data and code for the manuscript:

> **The geography of geocoding failure: missingness in OpenAlex affiliation strings is non-random and driven by writing conventions rather than map coverage**
> (Jimmy Hung-Chi Chang, Short Communication under review at *Journal of Informetrics*, 2026)

The paper audits geocoding failure (Nominatim/OpenStreetMap) across all 80,176 unique raw affiliation strings of a complete OpenAlex field corpus (silicon carbide, 2018–2024), traces it to address structure and text hygiene, validates an inverse-probability-weighting correction against a catalogue-region ground truth, benchmarks distance indicators against OpenAlex institution coordinates, and replicates the mechanisms in two further corpora.

Release **`v1.1-joi-revision`** archives the version used for the revised manuscript. Release `v1.0-joi-submission` archives the originally submitted version. Section, table and figure numbers below refer to the revised manuscript; "S" refers to its Supplementary Material.

## Repository layout

```
full_run/                  silicon-carbide corpus, geocoding results and caches (unchanged since v1.0)
cart_run/                  CAR-T corpus, same schema (unchanged since v1.0)
analysis/, figures/, results/   scripts and outputs of the submitted version (v1.0)
revision/                  everything added for the revision (v1.1):
  analysis/                  scripts 16–28, 19b and country_rule.py (explicit tie rule)
  results_revised_rule/      all outputs under the revised country rule (TIE_RULE=exclude)
  results_submitted_rule/    outputs under the submitted version's sort-dependent rule
  requery/                   L1–L4 Nominatim re-queries (decode-and-requery, held-structure test,
                             cascade-step, drift control) with inputs, results and caches
  scopus_comparison/         scripts and aggregate results of the OpenAlex–Scopus comparison
  hand_coding/               the author's hand coding: held-structure place names (300),
                             tied strings (100), discordant geocodes (100); instructions in sheet 先看這頁
  held_structure/            sampling design of the held-structure test
  photovoltaics_tables/      derived regional and mechanism tables of the third corpus
requirements_pinned.txt    exact package versions used for the revision
```

## Country assignment (revision)

Each string is assigned the modal catalogue country across its rows. In the submitted version a tied mode was broken by the default sort order, so results depended on the sorting algorithm. The revision uses an explicit rule (`revision/analysis/country_rule.py`, `TIE_RULE=exclude`): tied strings receive no country (Section 2.2, Supplementary Section S7.3). `TIE_RULE=quicksort` reproduces the submitted numbers.

## Reproducing the revised manuscript

Environment: Python 3.10+, `pip install -r requirements_pinned.txt` (pandas 2.3.3, numpy 2.2.6, scipy 1.15.3, statsmodels 0.14.6, patsy 1.0.2, reverse_geocoder 1.5.1, pycountry 26.2.16, openpyxl 3.1.5, matplotlib 3.10.9). Reassemble `full_run/affil_full.csv.gz` as described in `full_run/REASSEMBLE.md`, then:

```
cd full_run
export TIE_RULE=exclude
python ../revision/analysis/16_geocoding_mar_analysis.py      # failure geography, M1–M3, M2, IPW
python ../revision/analysis/17_diagnostics_and_replication.py # weights, CAR-T replication
python ../revision/analysis/18_robustness_checks.py           # specification, trimming, augmented sample
python ../revision/analysis/19_presubmission_pass.py          # country-FE propensity; bootstrap chunks (B_REPS, B_SEED, B_DUMP)
python ../revision/analysis/19b_pool_bootstrap.py boot19_*.npy  # pools five 400-replicate chunks into Table 2's intervals
python ../revision/analysis/21_validation_ground_truth.py     # Table 2 upper panel, Table S7
python ../revision/analysis/22_country_discordance.py         # S4.5, Table S5
python ../revision/analysis/23_models_and_classification.py   # S2: World Bank, categorical commas, wild bootstrap, random intercept
python ../revision/analysis/24_paper_level_and_minor.py       # S3.2–S3.3, Table S1, regional scheme
python ../revision/analysis/25_institution_coordinates.py     # S4.7, Table 2 lower panel (needs INST_DIR, see below)
python ../revision/analysis/26_region_recovery_diagnosis.py   # S4.8 (Middle East)
python ../revision/analysis/27_estimand_weighted_propensity.py # S4.2 trimming by region
python ../revision/analysis/28_mechanism_all_strings.py       # Section 3.3 statistics on all strings
python ../revision/analysis/20_make_figures.py                # Figures 1–3
```

For CAR-T statistics on all strings, run script 28 inside `cart_run/` with `CORPUS=cart`.

Script 25 needs the public OpenAlex institutions snapshot (June 2026; `s3://openalex/data/institutions/`) in the directory given by `INST_DIR`. Script 20 needs Natural Earth 1:110m admin-0 countries (`ne_110m_admin_0_countries.geojson`, public domain) in the working directory, or a path in `WORLD_GEOJSON`. All other scripts run on this repository alone. The scripts in `revision/requery/` call the public Nominatim service (1 request/second); their caches are included, so re-running them reuses the cached responses. The Scopus comparison needs Scopus API access; Scopus records are not redistributed, only the scripts and aggregate results.

## Data

All CSV files are UTF-8. See `full_run/` and `cart_run/` for the corpora (unchanged since v1.0). Data are derived from OpenAlex (CC0), OpenStreetMap/Nominatim (ODbL) and Wikidata (CC0); attribution notices apply per source.

## Citing this deposit

See `CITATION.cff`, or use GitHub's "Cite this repository".

## License

MIT (code).
