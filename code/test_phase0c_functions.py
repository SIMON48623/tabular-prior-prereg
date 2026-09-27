"""Unit tests for phase0c_functions.py. Run: python3 test_phase0c_functions.py"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import phase0c_functions as F  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


rng = np.random.default_rng(1)
n = 3000
# x0 strong, x1 near-duplicate of x0, x2..x5 independent weak
x0 = rng.standard_normal(n)
X = np.column_stack([x0, x0 + 0.1 * rng.standard_normal(n)] + [rng.standard_normal(n) for _ in range(4)])
names = [f"x{j}" for j in range(6)]
cl = F.feature_clusters(X, names)
check("duplicates share a cluster", cl["x0"] == cl["x1"])
check("independent features separate", len({cl[f"x{j}"] for j in range(2, 6)}) == 4 and cl["x2"] != cl["x0"])
m = {"x0": 0.80, "x1": 0.79, "x2": 0.55, "x3": 0.55, "x4": 0.55, "x5": 0.55}
import stratifier as S
n1, n0 = 900, 2100
ne_dn = S.neff_dn(m.values(), n1, n0)
ne_cl = F.neff_cluster(m, cl, n1, n0)
check("cluster N_eff lower than unclustered denoised N_eff with a duplicate", ne_cl < ne_dn, f"dn {ne_dn:.2f} cluster {ne_cl:.2f}")
thr = 1.645 * S.se0(n1, n0)
exp_l = [0.3 - thr] + [0.05 - thr] * 4
check("cluster N_eff value (denoised lifts)", abs(ne_cl - sum(exp_l) ** 2 / sum(x * x for x in exp_l)) < 1e-9)
Xn = X.copy(); Xn[rng.random(n) < 0.2, 3] = np.nan
check("missing values handled", len(F.feature_clusters(Xn, names)) == 6)

# conditional N_eff: y depends on x0 only; x1 duplicate should get small conditional share after LR?
# Use a clean case: equal effects on x2..x5, none on x0/x1 -> N_eff_cond ~ 4
eta = 0.6 * X[:, 2:6].sum(axis=1)
y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
per_fold = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
    per_fold.append(F.feature_contributions(lr.coef_[0], sc.transform(X[te]), names))
ne_c = F.neff_conditional(per_fold, names)
check("conditional N_eff ~ 4 for four equal effects (small upward bias from null features)", 3.8 < ne_c < 5.0, f"{ne_c:.2f}")
# one-hot grouping: two transformed columns belonging to one feature
contrib = F.feature_contributions(np.array([1.0, -1.0, 2.0]), np.array([[1, 0, 1], [0, 1, 0]], float), ["c", "c", "z"])
check("one-hot grouping sums columns", np.allclose(contrib["c"], [1, -1]) and np.allclose(contrib["z"], [2, 0]))
check("absent feature gets lift 0", abs(F.neff_conditional([{"a": np.array([0., 1.])}], ["a", "b"]) - 1.0) < 1e-12)

# medical rule
check("medical candidate hit", F.medical_candidate("ilpd", "Indian Liver Patient Dataset", ["Health"]) != [])
check("non-medical miss", F.medical_candidate("kc1", "software defect prediction NASA", []) == [])
check("GSE code", "gse31210" in F.medical_candidate("lungcancer_GSE31210", "", []))

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
