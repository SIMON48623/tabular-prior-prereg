"""Unit tests for stats_core.py. Run: python3 test_stats_core.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import stats_core as SC

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: FAILS.append(name)

rng = np.random.default_rng(3)
n, G = 300, 60
fam = rng.integers(0, G, n)
X = rng.standard_normal((n, 2))
y = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + rng.standard_normal(G)[fam] * 0.5 + rng.standard_normal(n)
w1 = np.ones(n)
r = SC.wls_cluster(y, X, fam, weights=w1)
b_ls = np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)[0]
check("unweighted WLS equals OLS", np.allclose(r["beta"], b_ls))
# manual CR1
Xc = np.column_stack([np.ones(n), X]); e = y - Xc @ b_ls; B = np.linalg.inv(Xc.T @ Xc)
M = sum(np.outer(Xc[fam == g].T @ e[fam == g], Xc[fam == g].T @ e[fam == g]) for g in np.unique(fam))
Gn = len(np.unique(fam)); V = (Gn / (Gn - 1)) * ((n - 1) / (n - 3)) * B @ M @ B
check("CR1 covariance matches manual formula", np.allclose(r["cov"], V))
check("df = G - 1", r["df"] == Gn - 1)
w = SC.family_weights(fam)
check("family weights sum to number of families", abs(w.sum() - Gn) < 1e-9)
m = SC.weighted_mean_cluster(y, fam)
fam_means = np.array([y[fam == g].mean() for g in np.unique(fam)])
check("family-weighted mean = mean of family means", abs(m["mean"] - fam_means.mean()) < 1e-12)
h = SC.holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
check("holm adjusted", abs(h["a"]["p_holm"] - 0.003) < 1e-12 and abs(h["b"]["p_holm"] - 0.04) < 1e-12 and abs(h["c"]["p_holm"] - 0.04) < 1e-12)
t = SC.tertile([0.5, 1.3021159127, 1.5, 1.6808121547, np.nan], 1.3021159127204662, 1.680812154716401)
check("tertile boundaries", list(t) == ["T1", "T1", "T2", "T2", "undefined"], str(list(t)))
# weights recomputed after dropping non-finite rows
yy = np.array([0.01, 0.01, np.nan, 0.03, 0.05]); ff = np.array(["A", "A", "A", "B", "C"])
mm = SC.weighted_mean_cluster(yy, ff)
check("family weights recomputed after NaN rows are dropped", abs(mm["mean"] - 0.03) < 1e-12, f"{mm['mean']:.4f}")
# explicit weights honoured
ww = np.array([1.0, 3.0, 1.0, 1.0, 1.0])
r2 = SC.wls_cluster(np.array([0., 1., 2., 3., 4.]), np.empty((5, 0)), np.array(list("ABCDE")), weights=ww)
check("explicit weights honoured", abs(r2["beta"][0] - (0 + 3 + 2 + 3 + 4) / 7) < 1e-12)
# one-sided p
check("one_sided_p greater/less symmetric", abs(SC.one_sided_p(1.0, 1.0, 30) + SC.one_sided_p(1.0, 1.0, 30, direction="less") - 1) < 1e-12)
check("one_sided_p with null", abs(SC.one_sided_p(0.012, 0.001, 50, null=0.01, direction="less") - SC.stats.t.cdf(2.0, 50)) < 1e-12)
check("defaults are 0.025", SC.holm.__defaults__[0] == 0.025 and SC.one_sided_upper.__defaults__[0] == 0.025)
# weighted CR1 against a hand-computed sandwich
wv = rng.uniform(0.2, 2.0, n)
rw = SC.wls_cluster(y, X, fam, weights=wv)
W = np.diag(wv); Bw = np.linalg.inv(Xc.T @ W @ Xc); bw = Bw @ Xc.T @ W @ y; ew = y - Xc @ bw
Mw = sum(np.outer(Xc[fam == g].T @ (wv[fam == g] * ew[fam == g]), Xc[fam == g].T @ (wv[fam == g] * ew[fam == g])) for g in np.unique(fam))
Vw = (Gn / (Gn - 1)) * ((n - 1) / (n - 3)) * Bw @ Mw @ Bw
check("weighted CR1 matches hand-computed sandwich", np.allclose(rw["beta"], bw) and np.allclose(rw["cov"], Vw))
# hierarchical weights (S5): recomputed after NaN rows are dropped
fam5 = np.array(["A", "A", "A", "A", "B", "B"]); ds5 = np.array([1, 1, 2, 2, 3, 3])
hw = SC.hierarchical_weights(fam5, ds5)
check("hierarchical weights: family total 1, datasets equal", np.allclose(hw, [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]))
y5 = np.array([1.0, np.nan, 3.0, 5.0, 10.0, 20.0])
r5 = SC.wls_cluster(y5, np.empty((6, 0)), fam5, dataset=ds5)
# after dropping row 2: dataset 1 has one row (weight 0.5), dataset 2 two rows (0.25 each); family B 0.5 each
exp5 = ((0.5 * 1 + 0.25 * 3 + 0.25 * 5) + (0.5 * 10 + 0.5 * 20)) / 2.0
check("hierarchical weights recomputed after NaN rows are dropped", abs(r5["beta"][0] - exp5) < 1e-12, f"{r5['beta'][0]:.4f} vs {exp5:.4f}")
print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
