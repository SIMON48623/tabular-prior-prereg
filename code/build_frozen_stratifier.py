"""Compute the v2 stratifiers and derive the tertile cutpoints.

Input: the final pool written by merge_pool.py (default: <repo>/final_pool; argv[1] overrides).
Output: stratifier_v2.csv and cutpoints_v2.json (default: <repo>/frozen; argv[2] overrides).
No model is fitted.
Rule for cutpoints: numpy.quantile(method='linear') at 1/3 and 2/3 of log N_eff_dn over target-region
datasets with defined N_eff_dn. T1: x < c1; T2: c1 <= x < c2; T3: x >= c2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import stratifier as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P0B = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "final_pool"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "frozen"
TARGET = (0.60, 0.85)

d = pd.read_csv(P0B / "datasets_stratifiers.csv", float_precision="round_trip")
d = d[d.excluded_reason.isna()].copy()
m = json.loads((P0B / "marginal_aucs.json").read_text())
rows = []
for _, r in d.iterrows():
    a = m[str(int(r.openml_id))]
    n1 = int(round(r.n * r.minority_rate))
    n0 = int(r.n) - n1
    rows.append({"openml_id": int(r.openml_id), "n1": n1, "n0": n0, "se0": S.se0(n1, n0),
                 "neff_raw": S.neff_raw(a.values()), "neff_dn": S.neff_dn(a.values(), n1, n0),
                 "n_features_above_null": int((S.lifts_dn(a.values(), n1, n0) > 0).sum())})
s = d[["openml_id", "name", "family", "family_size", "n", "p_used", "minority_rate", "auc_lr"]].merge(pd.DataFrame(rows), on="openml_id")
s["log_neff_raw"] = np.log(s.neff_raw)
s["log_neff_dn"] = np.log(s.neff_dn)
s["target_region"] = s.auc_lr.between(*TARGET)
tr = s[s.target_region & s.log_neff_dn.notna()]
c1, c2 = np.quantile(tr.log_neff_dn, [1 / 3, 2 / 3], method="linear")
s["tertile_dn"] = np.where(s.log_neff_dn.isna(), "undefined",
                           np.where(s.log_neff_dn < c1, "T1", np.where(s.log_neff_dn < c2, "T2", "T3")))
s.loc[~s.target_region, "tertile_dn"] = "outside_target"
# raw-stratifier tertiles (sensitivity analyses only), by the same quantile rule on the same datasets
r1, r2 = np.quantile(tr.log_neff_raw, [1 / 3, 2 / 3], method="linear")
s["tertile_raw"] = np.where(s.log_neff_raw < r1, "T1", np.where(s.log_neff_raw < r2, "T2", "T3"))
s.loc[~s.target_region, "tertile_raw"] = "outside_target"
OUT.mkdir(exist_ok=True, parents=True)
s.to_csv(OUT / "stratifier_v2.csv", index=False)
cut = {"rule": "numpy.quantile(method='linear'), target region, defined N_eff_dn",
       "c1_log_neff_dn": float(c1), "c2_log_neff_dn": float(c2),
       "c1_neff_dn": float(np.exp(c1)), "c2_neff_dn": float(np.exp(c2)),
       "z_denoise": S.Z_DENOISE, "target_region_auc_lr": TARGET,
       "raw_sensitivity_c1_log_neff_raw": float(r1), "raw_sensitivity_c2_log_neff_raw": float(r2)}
(OUT / "cutpoints_v2.json").write_text(json.dumps(cut, indent=1))
t = s[s.target_region]
print("pool", len(s), "target", len(t), "undefined in target", int(t.log_neff_dn.isna().sum()))
print("cutpoints", cut)
print(t.groupby("tertile_dn").agg(datasets=("name", "size"), families=("family", "nunique")))
from scipy.stats import spearmanr
x = t.dropna(subset=["log_neff_dn"])
for col, lab in [("p_used", "log p"), ("n", "log n"), ("auc_lr", "auc_lr"), ("minority_rate", "minority")]:
    v = np.log(x[col]) if col in ("p_used", "n") else x[col]
    print(f"Spearman(log N_eff_dn, {lab}) = {spearmanr(x.log_neff_dn, v).statistic:+.3f}   raw: {spearmanr(x.log_neff_raw, v).statistic:+.3f}")
