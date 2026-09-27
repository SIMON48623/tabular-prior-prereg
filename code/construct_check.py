"""Construct check of the stratifiers on synthetic data with known dispersion (true k).

For each synthetic cell the marginal AUROCs are computed by an emulation of the Phase 0b procedure
(direction chosen on the training split, held-out scores mapped through the training-fold empirical
CDF, pooled AUROC over the 5-fold split); the Phase 0b code itself is not part of this package, then raw and denoised N_eff are formed; the conditional N_eff uses the reference L2
logistic regression. Only simulated data are used. Output: frozen/construct_check.csv and a summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import generators as G  # noqa: E402
import phase0c_functions as F  # noqa: E402
import stratifier as S  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "frozen"
cal = json.loads(G.CALIB_FILE.read_text())


def nested_marginal(X, y):
    """Emulation of the Phase 0b marginal AUROC: direction chosen on the training fold; held-out
    scores mapped through the training-fold empirical CDF (mid-rank for ties); held-out values pooled
    over the five folds of StratifiedKFold(5, shuffle=True, random_state=42)."""
    n, p = X.shape
    sc = np.zeros((n, p))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        for j in range(p):
            sgn = 1.0 if roc_auc_score(y[tr], X[tr, j]) >= 0.5 else -1.0
            ref = np.sort(sgn * X[tr, j])
            v = sgn * X[te, j]
            sc[te, j] = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / (2.0 * len(tr))
    return np.array([roc_auc_score(y, sc[:, j]) for j in range(p)])


def conditional(X, y, names):
    per = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        s = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000).fit(s.transform(X[tr]), y[tr])
        per.append(F.feature_contributions(lr.coef_[0], s.transform(X[te]), names))
    return F.neff_conditional(per, names)


def ln(x):
    return float(np.log(x)) if x == x and x > 0 else float("nan")


def main():
    rows = []
    for gen in G.SYN_GEN:
        for rho in G.SYN_RHO:
            for auc in G.SYN_AUC:
                for p_keep in (12, 32):
                    for k in G.SYN_K:
                        if k > p_keep:
                            continue
                        for n in (300, 1000, 3000):
                            for rep in range(4):
                                X, y, _, inf = G.synth_draw(gen, k, rho, auc, n, rep, "train", cal)
                                noise = [j for j in range(G.SYN_P) if j not in inf][: p_keep - k]
                                cols = sorted(inf + noise)
                                Xs = X[:, cols]
                                a = nested_marginal(Xs, y)
                                n1 = int(y.sum()); n0 = n - n1
                                names = [f"f{j}" for j in range(len(cols))]
                                rows.append({"gen": gen, "rho": rho, "auc": auc, "p": p_keep, "k": k, "n": n, "rep": rep,
                                             "log_k": np.log(k), "log_neff_raw": ln(S.neff_raw(a)),
                                             "log_neff_dn": ln(S.neff_dn(a, n1, n0)),
                                             "log_neff_cond": ln(conditional(Xs, y, names))})
                    print(gen, rho, auc, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "construct_check.csv", index=False)
    V = ["log_neff_raw", "log_neff_dn", "log_neff_cond"]
    summ = {}
    for rho in G.SYN_RHO:
        g = df[df.rho == rho]
        summ[f"rho={rho}"] = {
            "mean_abs_error_vs_log_k": {v: float((g[v] - g.log_k).abs().mean()) for v in V},
            "mean_bias": {v: float((g[v] - g.log_k).mean()) for v in V},
            "spearman_with_log_k_within_p_n_mean": {v: float(g.groupby(["gen", "auc", "p", "n"]).apply(
                lambda h: spearmanr(h[v], h.log_k, nan_policy="omit").statistic).mean()) for v in V},
            "inflation_p32_minus_p12_same_k_n": {v: float((g[g.p == 32].groupby(["gen", "auc", "k", "n"])[v].mean()
                                                            - g[g.p == 12].groupby(["gen", "auc", "k", "n"])[v].mean()).mean()) for v in V},
            "k1_mean_estimate_by_n": {v: g[g.k == 1].groupby("n")[v].mean().round(3).to_dict() for v in V},
            "undefined_share": {v: float(g[v].isna().mean()) for v in V},
        }
    (OUT / "construct_check_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
