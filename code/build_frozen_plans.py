"""Build the frozen removal plan and the manipulation-check table from the final pool (merge_pool.py output).
No model is fitted; only the frozen marginal AUROCs and class counts are used.

Manipulation check: how much each intervention moves the primary stratifier (log N_eff_dn) and how
much of the total marginal signal it removes. For injected columns the population marginal AUROC is
used (noise columns: 0.5)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import generators as G  # noqa: E402
import stratifier as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P0B = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "final_pool"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "frozen"

st = pd.read_csv(OUT / "stratifier_v2.csv")
m = json.loads((P0B / "marginal_aucs.json").read_text())
t = st[st.target_region].copy()


def ln(x):
    return float(np.log(x)) if x == x and x > 0 else float("nan")


plans, rows = {}, []
for _, r in t.iterrows():
    oid = int(r.openml_id)
    a = m[str(oid)]
    n1, n0 = int(r.n1), int(r.n0)
    plan = G.removal_plan(oid, a)
    plans[str(oid)] = plan
    lifts = dict(zip(a, S.lifts_raw(a.values())))
    L = sum(lifts.values())
    row = {"openml_id": oid, "name": r["name"], "family": r.family, "tertile_dn": r.tertile_dn,
           "p_used": int(r.p_used), "n": int(r.n), "log_neff_dn": r.log_neff_dn, "log_neff_raw": r.log_neff_raw}
    for k in G.REMOVAL_K:
        if plan[k] is None:
            continue
        keep = [v for f, v in a.items() if f not in plan[k]["top"]]
        row[f"log_neff_dn_top{k}_removed"] = ln(S.neff_dn(keep, n1, n0))
        row[f"lift_share_top{k}"] = sum(lifts[f] for f in plan[k]["top"]) / L
        rn, rs = [], []
        for dr in plan[k]["random"]:
            keep = [v for f, v in a.items() if f not in dr]
            rn.append(ln(S.neff_dn(keep, n1, n0)))
            rs.append(sum(lifts[f] for f in dr) / L)
        row[f"log_neff_dn_rand{k}_removed_mean"] = float(np.nanmean(rn)) if np.isfinite(rn).any() else float("nan")
        row[f"lift_share_rand{k}_mean"] = float(np.mean(rs))
    thr = S.Z_DENOISE * S.se0(n1, n0)
    for lev in G.INJECT_LEVELS:
        for arm, tag in (("concentrated", "conc"), ("dispersed", "disp"), ("dispersed_noise", "dispn")):
            inj = [0.5 + x for x in G.injection_expected_lifts(arm, lev)]
            row[f"log_neff_dn_inj_{tag}_{lev}"] = ln(S.neff_dn(list(a.values()) + inj, n1, n0))
        row[f"disp_column_below_threshold_{lev}"] = bool(G.injection_expected_lifts("dispersed", lev)[0] <= thr)
    rows.append(row)

(OUT / "removal_plan.json").write_text(json.dumps(plans, indent=1, sort_keys=True))
mc = pd.DataFrame(rows)
mc.to_csv(OUT / "manipulation_check.csv", index=False)


def fw_mean(x, fam):
    ok = np.isfinite(x)
    x, fam = x[ok], fam[ok]
    w = 1.0 / fam.map(fam.value_counts())
    return float((w * x).sum() / w.sum())


print("datasets", len(mc), "families", mc.family.nunique(), "| top-3 eligible", int(mc.lift_share_top3.notna().sum()))
summary = {}
for k in (1, 3):
    sub = mc.dropna(subset=[f"lift_share_top{k}"])
    dt = sub[f"log_neff_dn_top{k}_removed"] - sub.log_neff_dn
    dr = sub[f"log_neff_dn_rand{k}_removed_mean"] - sub.log_neff_dn
    summary[f"removal_k{k}"] = {"dlog_neff_dn_top": fw_mean(dt, sub.family), "dlog_neff_dn_random": fw_mean(dr, sub.family),
                                "lift_share_top": fw_mean(sub[f"lift_share_top{k}"], sub.family),
                                "lift_share_random": fw_mean(sub[f"lift_share_rand{k}_mean"], sub.family)}
    print(f"removal k={k}", {kk: round(v, 3) for kk, v in summary[f'removal_k{k}'].items()})
for lev in G.INJECT_LEVELS:
    c = mc[f"log_neff_dn_inj_conc_{lev}"] - mc.log_neff_dn
    dd = mc[f"log_neff_dn_inj_disp_{lev}"] - mc.log_neff_dn
    dn = mc[f"log_neff_dn_inj_dispn_{lev}"] - mc.log_neff_dn
    summary[f"inject_{lev}"] = {"dlog_neff_dn_concentrated": fw_mean(c, mc.family), "dlog_neff_dn_dispersed": fw_mean(dd, mc.family),
                                "dlog_neff_dn_dispersed_noise": fw_mean(dn, mc.family),
                                "share_datasets_disp_column_below_threshold": float(mc[f"disp_column_below_threshold_{lev}"].mean())}
    print(f"inject {lev}", {kk: round(v, 3) for kk, v in summary[f'inject_{lev}'].items()})
(OUT / "manipulation_summary.json").write_text(json.dumps(summary, indent=1))
