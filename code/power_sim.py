"""Simulation-based power for the confirmatory tests of preregistration v2.

Uses the real target-region design (log N_eff_dn, log p, log n, auc_lr, family labels, v2 tertiles) and
simulated outcomes only. No model-comparison result exists or is used. Also reports power under the v1
stratifier (raw N_eff, covariate log p only) to document the correction of the v1 power table.

Outcome model for dataset i in family f:  y_i = effect_i + u_f + e_i,
u_f ~ N(0, icc * sd^2), e_i ~ N(0, (1 - icc) * sd^2). All tests use the frozen estimators in stats_core.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from stats_core import wls_cluster, weighted_mean_cluster, one_sided_upper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_csv(ROOT / "frozen" / "stratifier_v2.csv")
t = S[S.target_region & S.log_neff_dn.notna()].copy().reset_index(drop=True)
t["log_neff"] = t.log_neff_dn
t["log_p"] = np.log(t.p_used)
t["log_n"] = np.log(t.n)
t["T"] = t.tertile_dn
fam = t.family.values
ufam, finv = np.unique(fam, return_inverse=True)
R = 2000
ICC = 0.3
rng = np.random.default_rng(20260926)
HOLM_WORST = 0.05 / 6   # two-sided equivalent of the smallest Holm threshold (one-sided 0.025 / 6) among S1-S6
out = {}


def draw(effect, sd):
    u = rng.normal(0, np.sqrt(ICC) * sd, len(ufam))[finv]
    e = rng.normal(0, np.sqrt(1 - ICC) * sd, len(t))
    return effect + u + e


# ---------------- v1 check: raw N_eff with log p only (the v1 analysis model), sd 0.02
Xv1 = np.column_stack([t.log_neff_raw, t.log_p])
res = []
for b1 in (0.008, 0.015):
    hit = 0
    for _ in range(R):
        y = draw(b1 * (t.log_neff_raw.values - t.log_neff_raw.mean()), 0.02)
        hit += wls_cluster(y, Xv1, fam, weights=np.ones(len(t)))["ci95"][1, 0] > 0
    res.append({"beta1": b1, "power_v1_model_unweighted": hit / R})
out["v1_model_check"] = res
print(pd.DataFrame(res))

# ---------------- P2 (primary, fixed sequence after P1): slope on log N_eff_dn | log p, log n, auc_lr
X = t[["log_neff", "log_p", "log_n", "auc_lr"]].values
res = []
for sd in (0.015, 0.02, 0.03):
    for b1 in (0.004, 0.006, 0.008, 0.010, 0.015):
        hit, hh = 0, 0
        for _ in range(R):
            y = draw(b1 * (t.log_neff.values - t.log_neff.mean()), sd)
            r = wls_cluster(y, X, fam)
            hit += (r["ci95"][1, 0] > 0)
            hh += (r["beta"][1] - stats.t.ppf(1 - HOLM_WORST / 2, r["df"]) * r["se"][1]) > 0
        res.append({"sd": sd, "beta1": b1, "power_alpha05": hit / R, "power_holm_worst": hh / R})
out["P2"] = res
print(pd.DataFrame(res))

# ---------------- S1: T1 non-superiority (upper one-sided 97.5% bound < +0.01) when true T1 mean = mu
T1 = t["T"].values == "T1"
T3 = t["T"].values == "T3"
res = []
for sd in (0.015, 0.02, 0.03):
    for mu in (-0.005, 0.0, 0.003):
        h, hh = 0, 0
        for _ in range(R):
            y = draw(mu, sd)
            r = weighted_mean_cluster(y[T1], fam[T1])
            h += one_sided_upper(r["mean"], r["se"], r["df"], 0.025) < 0.01
            hh += one_sided_upper(r["mean"], r["se"], r["df"], HOLM_WORST / 2) < 0.01
        res.append({"sd": sd, "true_mean_T1": mu, "power_one_sided_025": h / R, "power_holm_worst": hh / R})
out["S1_T1_nonsuperiority"] = res
print(pd.DataFrame(res))

# ---------------- S2: T3 superiority (lower two-sided 95% bound > 0)
res = []
for sd in (0.015, 0.02, 0.03):
    for mu in (0.005, 0.010, 0.015):
        h, hh = 0, 0
        for _ in range(R):
            y = draw(mu, sd)
            r = weighted_mean_cluster(y[T3], fam[T3])
            h += r["ci95"][0] > 0
            hh += (r["mean"] - stats.t.ppf(1 - HOLM_WORST / 2, r["df"]) * r["se"]) > 0
        res.append({"sd": sd, "true_mean_T3": mu, "power_alpha05": h / R, "power_holm_worst": hh / R})
out["S2_T3_superiority"] = res
print(pd.DataFrame(res))

# ---------------- P1 (primary) and S3/S4/S6 (Holm): injection contrast, all target-region datasets with a defined N_eff_dn
res = []
for sd in (0.01, 0.02, 0.03):
    for mu in (-0.002, -0.004, -0.005, -0.008):
        h, hh = 0, 0
        for _ in range(R):
            y = draw(mu, sd)
            r = weighted_mean_cluster(y, fam)
            h += r["ci95"][1] < 0
            hh += (r["mean"] + stats.t.ppf(1 - HOLM_WORST / 2, r["df"]) * r["se"]) < 0
        res.append({"sd": sd, "true_mean_contrast": mu, "power_alpha05": h / R, "power_holm_worst": hh / R})
out["P1_and_S3_S4_S6"] = res
print(pd.DataFrame(res))

# ---------------- S5: interaction log N_eff_dn x log n_train in the sample-size arm
sizes = np.array([100, 250, 500, 1000])
rows = []
for i, r in t.iterrows():
    ntr = int(np.floor(r.n * 0.8))
    for s in sizes:
        if s < ntr and s * r.minority_rate >= 10:
            rows.append((i, np.log(s)))
    rows.append((i, np.log(ntr)))           # full training fold (main arm)
ss = pd.DataFrame(rows, columns=["i", "log_ntr"])
ss["log_neff"] = t.log_neff.values[ss.i]
ss["log_p"] = t.log_p.values[ss.i]
ss["auc_lr"] = t.auc_lr.values[ss.i]
ss["fam"] = fam[ss.i]
ss["c_neff"] = ss.log_neff - t.log_neff.mean()
ss["c_n"] = ss.log_ntr - np.log(500)
ss["inter"] = ss.c_neff * ss.c_n
# weights: family weight split equally over that dataset's size levels
cnt = ss.groupby("i").i.transform("size")
fw = 1.0 / pd.Series(fam).map(pd.Series(fam).value_counts()).values
ss["w"] = fw[ss.i] / cnt
Xs = ss[["c_neff", "c_n", "inter", "log_p", "auc_lr"]].values
res = []
for sd in (0.02, 0.03):
    for bint in (-0.002, -0.003, -0.005):
        h, hh = 0, 0
        for _ in range(R // 2):
            u = rng.normal(0, np.sqrt(ICC) * sd, len(ufam))[finv][ss.i]
            v = rng.normal(0, np.sqrt(0.3) * sd, len(t))[ss.i]          # dataset effect shared over sizes
            e = rng.normal(0, np.sqrt(0.4) * sd, len(ss))
            y = 0.008 * ss.c_neff - 0.01 * ss.c_n + bint * ss.inter + u + v + e
            r = wls_cluster(y.values, Xs, ss.fam.values, dataset=ss.i.values)
            h += r["ci95"][3, 1] < 0
            hh += (r["beta"][3] + stats.t.ppf(1 - HOLM_WORST / 2, r["df"]) * r["se"][3]) < 0
        res.append({"sd": sd, "beta_interaction": bint, "power_alpha05": h / (R // 2), "power_holm_worst": hh / (R // 2)})
out["S5_n_interaction"] = res
out["S5_design"] = {"rows": len(ss), "datasets": int(ss.i.nunique()),
                    "rows_per_size": ss.groupby(np.exp(ss.log_ntr).round().clip(upper=1001)).size().to_dict()}
print(pd.DataFrame(res))
print(out["S5_design"])
(ROOT / "power" / "power_results.json").write_text(json.dumps(out, indent=1, default=float))
