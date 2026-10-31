| Set | Unique strings | Failure % [95% CI] | ≤1 comma: share of strings % / share of failures % | 0-comma fail % | 1-comma fail % | 2-comma fail % | 3+ fail % | HTML-entity % | Mean commas |
|---|---|---|---|---|---|---|---|---|---|
| A Scopus affilname, city, country | 1,573 | 0.3 [0.1, 0.7] | 0.0 / 0.0 | None | None | 0.3 | 0.0 | 0.0 | 2.12 |
| B OpenAlex raw_affiliation_strings | 3,000 | 10.7 [9.6, 11.9] | 12.0 / 68.2 | 60.9 | 60.0 | 4.7 | 3.8 | 1.7 | 3.26 |
| C Scopus city, country | 766 | 0.5 [0.2, 1.3] | 99.7 / 100.0 | None | 0.5 | 0.0 | None | 0.0 | 1.0 |

Coverage: {"works_in_both": 1806, "scopus_affiliation_rows": 3793, "scopus_rows_with_city_and_country_pct": 99.9, "scopus_rows_with_country_pct": 100.0, "scopus_rows_with_afid_pct": 0.0, "openalex_raw_strings": 14823, "openalex_strings_with_institution_pct": 97.1, "note": "Scopus raw affiliation text not available at this API service level; Scopus sets are structured fields"}
Paired works: {"n_works": 1529, "all_strings_geocoded_A_pct": 99.7, "all_strings_geocoded_B_pct": 85.9, "A_ok_B_fail_pct": 14.0, "B_ok_A_fail_pct": 0.2}