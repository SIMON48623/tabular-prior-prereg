"""Choice of the denoising threshold z (made before registration on synthetic data only).

Linear generator, uncorrelated features, Bayes AUROC 0.70/0.80, p in {12, 32}, all k <= p,
n in {300, 1000, 3000}, 4 replicates. Marginal AUROCs by the emulated Phase 0b procedure
(construct_check.nested_marginal). The draws are the same as in the construct check, so this
comparison is not an independent validation of the chosen threshold.
Criteria: mean absolute error of log N_eff vs log k; inflation from p = 12 to p = 32 at fixed k and n;
mean within-(p, n) Spearman correlation with log k.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import generators as G  # noqa: E402
import stratifier as S  # noqa: E402
from construct_check import nested_marginal  # noqa: E402  (imports only the function; see guard below)

Z = (0.0, 1.0, 1.645, 1.96, 2.576)
cal = json.loads(G.CALIB_FILE.read_text())
rows = []
for auc in G.SYN_AUC:
    for p_keep in (12, 32):
        for k in G.SYN_K:
            if k > p_keep:
                continue
            for n in (300, 1000, 3000):
                for rep in range(4):
                    X, y, _, inf = G.synth_draw("linear", k, 0.0, auc, n, rep, "train", cal)
                    noise = [j for j in range(G.SYN_P) if j not in inf][: p_keep - k]
                    a = nested_marginal(X[:, sorted(inf + noise)], y)
                    n1 = int(y.sum()); n0 = n - n1
                    r = {"auc": auc, "p": p_keep, "k": k, "n": n, "rep": rep, "log_k": np.log(k)}
                    for z in Z:
                        v = S.hill2(np.maximum(a - 0.5 - z * S.se0(n1, n0), 0.0))
                        r[f"z{z}"] = np.log(v) if v == v and v > 0 else np.nan
                    rows.append(r)
df = pd.DataFrame(rows)
out = {}
for z in Z:
    c = f"z{z}"
    infl = (df[df.p == 32].groupby(["auc", "k", "n"])[c].mean() - df[df.p == 12].groupby(["auc", "k", "n"])[c].mean()).mean()
    rank = df.groupby(["auc", "p", "n"]).apply(lambda g: spearmanr(g[c], g.log_k, nan_policy="omit").statistic).mean()
    out[c] = {"mean_abs_error": float((df[c] - df.log_k).abs().mean()), "p_inflation": float(infl),
              "mean_rank_corr": float(rank), "undefined_share": float(df[c].isna().mean())}
(Path(__file__).resolve().parents[1] / "frozen" / "threshold_choice.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
