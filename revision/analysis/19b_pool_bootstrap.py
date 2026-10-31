#!/usr/bin/env python3
"""
19b_pool_bootstrap.py

Pools bootstrap chunks written by 19_presubmission_pass.py into the 2,000-replicate
intervals reported for Latin America in main-text Table 2.

Each chunk is produced by
    TIE_RULE=exclude B_REPS=400 B_SEED=<seed> B_DUMP=boot19_<k>.npy python ../revision/analysis/19_presubmission_pass.py
with seeds 20260812-20260816 (k = 1..5). Each .npy holds one row per replicate:
[observed Latin America pair share (%), IPW-corrected share (%)].

Usage:  python 19b_pool_bootstrap.py boot19_1.npy ... boot19_5.npy  -> boot19_combined.json
"""
import json, sys
import numpy as np
S = np.vstack([np.load(f) for f in sys.argv[1:]])
pc = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
rel = (S[:, 1] - S[:, 0]) / S[:, 1] * 100
out = {"n_reps": int(len(S)), "LA_raw_CI": pc(S[:, 0]), "LA_ipw_CI": pc(S[:, 1]),
       "understatement_CI": [round(float(np.percentile(rel, 2.5)), 1), round(float(np.percentile(rel, 97.5)), 1)],
       "diff_CI": pc(S[:, 1] - S[:, 0]), "seeds": "20260812-20260816, 5x400"}
json.dump(out, open("boot19_combined.json", "w"), indent=1); print(out)
