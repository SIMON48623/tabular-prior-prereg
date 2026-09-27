"""Unit tests for generators.py. Run: python3 test_generators.py  (all must print PASS)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import generators as G  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


rng = np.random.default_rng(0)
n = 200_000
y = (rng.random(n) < 0.3).astype(int)

# --- injection: marginal AUROCs, information matching, noise matching, determinism, common random numbers
for lev in G.INJECT_LEVELS:
    Zc, nc = G.injection_block(y, 12345, "concentrated", lev)
    Zd, nd = G.injection_block(y, 12345, "dispersed", lev)
    Zdn, ndn = G.injection_block(y, 12345, "dispersed_noise", lev)
    Zn, _ = G.injection_block(y, 12345, "noise", None)
    check(f"inject shapes ({lev})", Zc.shape[1] == 8 and Zd.shape[1] == 8 and Zdn.shape[1] == 15 and len(ndn) == 15)
    aucs_c = [roc_auc_score(y, Zc[:, j]) for j in range(8)]
    aucs_d = [roc_auc_score(y, Zd[:, j]) for j in range(8)]
    aucs_dn = [roc_auc_score(y, Zdn[:, j]) for j in range(15)]
    exp_d = G.injection_expected_lifts("dispersed", lev)
    check(f"inject conc max marginal AUROC ~ {lev}", abs(max(aucs_c) - lev) < 0.005, f"{max(aucs_c):.4f}")
    check(f"inject conc has 7 pure-noise columns ({lev})", sum(abs(a - 0.5) < 0.006 for a in aucs_c) == 7)
    check(f"inject disp each column ~ {0.5 + exp_d[0]:.4f}", max(abs(a - 0.5 - exp_d[0]) for a in aucs_d) < 0.006)
    n_noise_dn = sum(abs(a - 0.5) < 0.006 for a in aucs_dn)
    n_sig_dn = sum(abs(a - 0.5 - exp_d[0]) < 0.006 for a in aucs_dn)
    check(f"inject disp_noise = 8 weak + 7 noise ({lev})", n_noise_dn == 7 and n_sig_dn == 8, f"{n_sig_dn}+{n_noise_dn}")
    d = G.injection_separation(lev)
    sc = Zc[:, int(np.argmax(aucs_c))]
    a_c, a_d = roc_auc_score(y, sc), roc_auc_score(y, Zd.sum(axis=1))
    sig_cols = [j for j in range(15) if abs(aucs_dn[j] - 0.5 - exp_d[0]) < 0.006]
    a_dn = roc_auc_score(y, Zdn[:, sig_cols].sum(axis=1))
    target = norm.cdf(d / math.sqrt(2))
    check(f"inject information matched at {lev}", max(abs(a_c - target), abs(a_d - target), abs(a_dn - target)) < 0.005,
          f"conc {a_c:.4f} disp {a_d:.4f} disp_noise {a_dn:.4f} target {target:.4f}")
    pos_c = int(np.argmax(aucs_c))
    conc_noise = np.delete(Zc, pos_c, axis=1)
    noise_cols_dn = [k for k in range(15) if k not in sig_cols]
    check(f"inject conc shares its 7 pure-noise columns with disp_noise ({lev})",
          all(any(np.allclose(conc_noise[:, j], Zdn[:, k]) for k in noise_cols_dn) for j in range(7)))
    check(f"inject conc strong column shares its noise with weak column pos ({lev})",
          np.allclose(Zc[:, pos_c] - d * y, Zn[:, pos_c]) and np.allclose(Zd[:, pos_c] - (d / math.sqrt(8)) * y, Zn[:, pos_c]))
    check(f"inject disp columns contained in disp_noise ({lev})",
          all(any(np.allclose(Zd[:, j], Zdn[:, k]) for k in range(15)) for j in range(8)))
Z1, _ = G.injection_block(y, 777, "dispersed_noise", 0.85, rep=1)
Z2, _ = G.injection_block(y, 777, "dispersed_noise", 0.85, rep=1)
check("inject deterministic", np.array_equal(Z1, Z2))
Z3, _ = G.injection_block(y, 778, "dispersed_noise", 0.85, rep=1)
Z4, _ = G.injection_block(y, 777, "dispersed_noise", 0.85, rep=2)
check("inject differs across datasets and replicates", not np.allclose(Z1, Z3) and not np.allclose(Z1, Z4))
try:
    G.injection_block(y, 777, "dispersed", 0.85, rep=3)
    check("inject rejects rep outside 0..2", False)
except ValueError:
    check("inject rejects rep outside 0..2", True)

# --- removal plan
m = {f"f{j}": 0.5 + 0.01 * j for j in range(10)}
m["tie_a"] = 0.59
plan = G.removal_plan(99, m)
m2 = {"b": 0.7, "a": 0.7, "c": 0.6, "d": 0.55, "e": 0.52, "f": 0.51, "g": 0.5}
p2 = G.removal_plan(1, m2)
check("removal top1 highest, ties by name", p2[1]["top"] == ["a"] and p2[3]["top"] == ["a", "b", "c"], str(p2[3]["top"]))
check("removal random excludes top", all(not set(r) & set(plan[3]["top"]) for r in plan[3]["random"]))
check("removal 5 draws of size k", len(plan[1]["random"]) == 5 and all(len(r) == 3 for r in plan[3]["random"]))
check("removal deterministic", plan == G.removal_plan(99, m))
small = {f"g{j}": 0.6 - 0.01 * j for j in range(5)}
check("removal k=3 needs p>=6", G.removal_plan(5, small)[3] is None and G.removal_plan(5, small)[1] is not None)

# --- subsample
ytr = (rng.random(4000) < 0.2).astype(int)
idx = G.subsample_indices(ytr, 31, 0, 250, 2)
check("subsample size", len(idx) == 250 and len(np.unique(idx)) == 250)
check("subsample stratified", abs(ytr[idx].mean() - ytr.mean()) < 0.01, f"{ytr[idx].mean():.3f} vs {ytr.mean():.3f}")
check("subsample deterministic", np.array_equal(idx, G.subsample_indices(ytr, 31, 0, 250, 2)))
check("subsample eligibility", G.subsample_eligible(100, 800, 0.1) and not G.subsample_eligible(100, 800, 0.069)
      and not G.subsample_eligible(1000, 900, 0.5))

# --- synthetic
cal = json.loads(G.CALIB_FILE.read_text())
for key in ["linear|k=1|rho=0.0|auc=0.7", "threshold|k=8|rho=0.3|auc=0.8", "linear|k=32|rho=0.3|auc=0.8"]:
    gen, k, rho, auc = key.split("|")
    k, rho, auc = int(k[2:]), float(rho[4:]), float(auc[4:])
    X, yy, p, inf = G.synth_draw(gen, k, rho, auc, 100_000, 0, "test", cal)
    check(f"synth {key} prevalence", abs(yy.mean() - 0.3) < 0.006, f"{yy.mean():.4f}")
    check(f"synth {key} Bayes AUROC", abs(roc_auc_score(yy, p) - auc) < 0.006, f"{roc_auc_score(yy, p):.4f}")
    check(f"synth {key} informative count", len(inf) == k)
Xa, ya, _, ia = G.synth_draw("linear", 4, 0.0, 0.7, 300, 3, "train", cal)
Xb, yb, _, ib = G.synth_draw("linear", 4, 0.0, 0.7, G.SYN_N_TEST, 3, "test", cal)
Xc, yc, _, ic = G.synth_draw("linear", 4, 0.0, 0.7, 3000, 3, "train", cal)
check("synth same informative columns across n and split", Xa.shape[0] == 300 and Xb.shape[0] == G.SYN_N_TEST and ia == ib == ic)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=2000).fit(Xc, yc)
top = sorted(np.argsort(-np.abs(lr.coef_[0]))[:4].tolist())
check("synth informative columns recovered by LR on train (n=3000)", top == ia, f"{top} vs {ia}")
auc_te = roc_auc_score(yb, lr.predict_proba(Xb)[:, 1])
check("synth test AUROC of train-fitted LR near Bayes", abs(auc_te - 0.7) < 0.02, f"{auc_te:.3f}")

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
