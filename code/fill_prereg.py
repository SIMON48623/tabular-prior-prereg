"""Fill the {{placeholders}} of preregistration_v2.template.md from the frozen outputs and write
preregistration_v2.md. Every pool-dependent or simulation-dependent number in the plan is a placeholder
filled here from frozen files, so that rebuilding the frozen files on the final pool and rerunning this
script updates the plan. Numbers that describe draft v1 or the Phase 0b history are fixed text."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "docs" / "preregistration_v2.template.md").read_text()
pw = json.loads((ROOT / "power" / "power_results.json").read_text())
cc = json.loads((ROOT / "frozen" / "construct_check_summary.json").read_text())
ccd = pd.read_csv(ROOT / "frozen" / "construct_check.csv")
tc = json.loads((ROOT / "frozen" / "threshold_choice.json").read_text())
cut = json.loads((ROOT / "frozen" / "cutpoints_v2.json").read_text())
V = {}

c0 = cc["rho=0.0"]
for tag, key in [("raw", "log_neff_raw"), ("dn", "log_neff_dn"), ("cond", "log_neff_cond")]:
    V[f"cc_{tag}_mae"] = f"{c0['mean_abs_error_vs_log_k'][key]:.2f}"
    V[f"cc_{tag}_bias"] = f"{c0['mean_bias'][key]:+.2f}"
    V[f"cc_{tag}_infl"] = f"{c0['inflation_p32_minus_p12_same_k_n'][key]:+.2f}"


def by_n(df, col):
    g = df.groupby("n")[col].mean()
    return " / ".join(f"{g[n]:.2f}" for n in (300, 1000, 3000))


u = ccd[ccd.rho == 0.0]
V["cc_k1_raw_by_n"] = by_n(u[u.k == 1], "log_neff_raw")
V["cc_k1_dn_by_n"] = by_n(u[u.k == 1], "log_neff_dn")
V["cc_rank"] = ", ".join(f"{t} {c0['spearman_with_log_k_within_p_n_mean'][k]:.2f}"
                         for t, k in [("raw", "log_neff_raw"), ("denoised", "log_neff_dn"), ("conditional", "log_neff_cond")])
k32 = u[(u.k == 32) & (u.p == 32)]
V["cc_k32_dn_by_n"] = by_n(k32, "log_neff_dn")
V["cc_k32_raw_by_n"] = by_n(k32, "log_neff_raw")
r3 = ccd[(ccd.rho == 0.3) & (ccd.k == 1)]
V["cc_k1_dn_rho3_by_n"] = by_n(r3, "log_neff_dn")
V["cc_rho3_k1_T3"] = f"{float((r3.log_neff_dn >= cut['c2_log_neff_dn']).mean()) * 100:.0f}%"

V["tc_mae_1645"] = f"{tc['z1.645']['mean_abs_error']:.2f}"
V["tc_mae_1"] = f"{tc['z1.0']['mean_abs_error']:.2f}"
V["tc_mae_196"] = f"{tc['z1.96']['mean_abs_error']:.2f}"
V["tc_rank_1645"] = f"{tc['z1.645']['mean_rank_corr']:.2f}"
V["tc_infl_1645"] = f"{tc['z1.645']['p_inflation']:+.2f}"
V["tc_rank_1"] = f"{tc['z1.0']['mean_rank_corr']:.2f}"
V["tc_infl_1"] = f"{tc['z1.0']['p_inflation']:+.2f}"

rows = []
for b in (0.004, 0.006, 0.008, 0.010, 0.015):
    cells = {r["sd"]: r for r in pw["P2"] if r["beta1"] == b}
    rows.append(f"| {b:.3f} | {cells[0.015]['power_alpha05']:.2f} | {cells[0.02]['power_alpha05']:.2f} | {cells[0.03]['power_alpha05']:.2f} |")
V["pow_p2_rows"] = "\n".join(rows)
V["pow_v1_008"] = f"{[r for r in pw['v1_model_check'] if r['beta1'] == 0.008][0]['power_v1_model_unweighted']:.2f}"
rows = []
for m in (-0.002, -0.004, -0.005, -0.008):
    cells = {r["sd"]: r for r in pw["P1_and_S3_S4_S6"] if r["true_mean_contrast"] == m}
    rows.append(f"| {m:+.3f} | {cells[0.01]['power_alpha05']:.2f} | {cells[0.02]['power_alpha05']:.2f} | "
                f"{cells[0.03]['power_alpha05']:.2f} | {cells[0.02]['power_holm_worst']:.2f} |")
V["pow_p1_rows"] = "\n".join(rows)
s1 = {r["sd"]: r for r in pw["S1_T1_nonsuperiority"] if r["true_mean_T1"] == 0.0}
V["pow_s1"] = "; ".join(f"SD {sd}: {s1[sd]['power_one_sided_025']:.2f} unadjusted, {s1[sd]['power_holm_worst']:.2f} at the worst Holm threshold" for sd in (0.015, 0.02))
s2 = {r["sd"]: r for r in pw["S2_T3_superiority"] if r["true_mean_T3"] == 0.01}
V["pow_s2"] = "; ".join(f"SD {sd}: {s2[sd]['power_alpha05']:.2f} / {s2[sd]['power_holm_worst']:.2f}" for sd in (0.015, 0.02))
s5 = {r["sd"]: r for r in pw["S5_n_interaction"] if r["beta_interaction"] == -0.005}
V["pow_s5"] = "; ".join(f"SD {sd}: {s5[sd]['power_alpha05']:.2f} / {s5[sd]['power_holm_worst']:.2f}" for sd in (0.02, 0.03))

# ---------------------------------------------------------------- pool-dependent values (frozen files)
import sys
from scipy.stats import spearmanr

sys.path.insert(0, str(ROOT / "code"))
import stratifier as S  # noqa: E402
from stats_core import tertile as _tertile, load_cutpoints  # noqa: E402

P0B = ROOT / "phase0b_outputs"
p0b = pd.read_csv(P0B / "datasets_stratifiers.csv", float_precision="round_trip")
p0b_inc = p0b[p0b.excluded_reason.isna()]
p0b_t = p0b_inc[(p0b_inc.auc_lr >= 0.60) & (p0b_inc.auc_lr <= 0.85)]
V["p0b_included"] = str(len(p0b_inc))
V["p0b_target"] = str(len(p0b_t))
V["p0b_target_fam"] = str(p0b_t.family.nunique())
V["p0b_raw_auclr_target"] = f"{spearmanr(np.log(p0b_t.N_eff), p0b_t.auc_lr)[0]:+.3f}"

sv = pd.read_csv(ROOT / "frozen" / "stratifier_v2.csv", float_precision="round_trip")
tr = sv[sv.target_region].copy()
dfn = tr.dropna(subset=["log_neff_dn"]).copy()
V["n_target"] = str(len(tr))
V["n_defined"] = str(len(dfn))
V["fam_defined"] = str(dfn.family.nunique())
und = tr[tr.log_neff_dn.isna()]
V["n_undef"] = str(len(und))
V["undef_ids"] = ", ".join(str(int(o)) for o in und.openml_id) or "none"


def rho(a, b):
    return f"{spearmanr(a, b)[0]:+.3f}"


V["raw_logp"] = rho(dfn.log_neff_raw, np.log(dfn.p_used))
V["dn_logp"] = rho(dfn.log_neff_dn, np.log(dfn.p_used))
V["dn_auclr"] = rho(dfn.log_neff_dn, dfn.auc_lr)
V["dn_range"] = f"{dfn.log_neff_dn.min():.2f} to {dfn.log_neff_dn.max():.2f}"
_m = json.loads((ROOT / "final_pool" / "marginal_aucs.json").read_text())
_z1 = np.array([np.log(S.hill2(S.lifts_dn(np.array(list(_m[str(int(r.openml_id))].values()), float), int(r.n1), int(r.n0), 1.0)))
                for r in dfn.itertuples()])
V["dn_auclr_z1"] = rho(_z1, dfn.auc_lr)
V["corr_rows"] = "\n".join(
    f"| {lab} | {rho(dfn[c], np.log(dfn.p_used))} | {rho(dfn[c], np.log(dfn.n))} | {rho(dfn[c], dfn.auc_lr)} | {rho(dfn[c], dfn.minority_rate)} |"
    for lab, c in [("log N_eff_dn", "log_neff_dn"), ("log N_eff_raw (v1)", "log_neff_raw")])

c1, c2 = cut["c1_log_neff_dn"], cut["c2_log_neff_dn"]
dfn["T"] = _tertile(dfn.log_neff_dn, c1, c2)
assert (dfn["T"] == dfn.tertile_dn).all(), "tertile column disagrees with cutpoints"
V["cut_gap"] = f"{c2 - c1:.2f}"
rows = []
labels = {"T1": "T1 narrowest", "T2": "T2 intermediate", "T3": "T3 broadest"}
rng_log = {"T1": f"< {c1:.10f}", "T2": f"{c1:.10f} – {c2:.10f}", "T3": f"≥ {c2:.10f}"}
rng_n = {"T1": f"< {np.exp(c1):.3f}", "T2": f"{np.exp(c1):.3f} – {np.exp(c2):.3f}", "T3": f"≥ {np.exp(c2):.3f}"}
for T in ("T1", "T2", "T3"):
    g = dfn[dfn["T"] == T]
    vc = g.family.value_counts()
    big = f"{vc.iloc[0]} (`{vc.index[0]}`)" if vc.iloc[0] > 1 else "1"
    rows.append(f"| {labels[T]} | {rng_log[T]} | {rng_n[T]} | {len(g)} | {g.family.nunique()} | {(vc == 1).sum()} | {big} |")
    V[f"{T}_fam"] = str(g.family.nunique())
    if T == "T1":
        V["T1_largest"] = f"`{vc.index[0]}` ({vc.iloc[0]} datasets)" + (", a synthetic generator grid" if vc.index[0] == "fri_c" else "")
V["tertile_rows"] = "\n".join(rows)

ms = json.loads((ROOT / "frozen" / "manipulation_summary.json").read_text())
V["rm1_share_top"] = f"{ms['removal_k1']['lift_share_top'] * 100:.0f}%"
V["rm1_dlog_top"] = f"{ms['removal_k1']['dlog_neff_dn_top']:+.2f}"
V["rm1_dlog_rand"] = f"{ms['removal_k1']['dlog_neff_dn_random']:+.2f}"
rr = []
for k in (1, 3):
    m = ms[f"removal_k{k}"]
    rr.append(f"| Top-{k} | {m['lift_share_top'] * 100:.0f}% | {m['dlog_neff_dn_top']:+.2f} |")
    rr.append(f"| Random-{k} | {m['lift_share_random'] * 100:.0f}% | {m['dlog_neff_dn_random']:+.2f} |")
V["removal_rows"] = "\n".join(rr[0:1] + rr[1:2] + rr[2:3] + rr[3:4])
mc = pd.read_csv(ROOT / "frozen" / "manipulation_check.csv")
V["n_top3"] = str(int((mc.p_used >= 6).sum()))
mr = []
for lev in ("0.65", "0.75", "0.85", "0.95"):
    m = ms[f"inject_{lev}"]
    assert abs(m["dlog_neff_dn_dispersed"] - m["dlog_neff_dn_dispersed_noise"]) < 1e-12
    mr.append(f"| {lev} | {m['dlog_neff_dn_concentrated']:+.2f} | {m['dlog_neff_dn_dispersed']:+.2f} | "
              f"{m['share_datasets_disp_column_below_threshold'] * 100:.0f}% |")
V["manip_rows"] = "\n".join(mr)

p1 = {(r["sd"], r["true_mean_contrast"]): r for r in pw["P1_and_S3_S4_S6"]}
V["p1_sd01_004"] = f"{p1[(0.01, -0.004)]['power_alpha05']:.2f}"
V["p1_sd02_mesoi"] = f"{p1[(0.02, -0.005)]['power_alpha05']:.2f}"
p2 = {(r["sd"], r["beta1"]): r for r in pw["P2"]}
V["p2_sd02_mesoi"] = f"{p2[(0.02, 0.008)]['power_alpha05']:.2f}"
b80 = [b for b in (0.004, 0.006, 0.008, 0.010, 0.015) if p2[(0.02, b)]["power_alpha05"] >= 0.8]
V["p2_beta80"] = f"{b80[0]:.3f}" if b80 else "> 0.015"

# ---------------------------------------------------------------- final pool and Phase 0c values
import hashlib
FP = ROOT / "final_pool"
P0C = ROOT / "phase0c_outputs"
fin = pd.read_csv(FP / "datasets_stratifiers.csv", float_precision="round_trip")
fin_inc = fin[fin.excluded_reason.isna()]
V["final_included"] = str(len(fin_inc))
V["final_families"] = str(fin_inc.family.nunique())
V["p0b_families"] = str(p0b_inc.family.nunique())
V["fam_target"] = str(tr.family.nunique())
dec = json.loads((ROOT / "frozen" / "pool_decisions.json").read_text())
V["n_alias_merges"] = str(len(dec["alias_merges"]))
V["n_family_corrections"] = str(len(dec["family_corrections"]))
V["family_corrections_list"] = "; ".join(f"`{a}` into `{b}`" for a, b in dec["family_corrections"].items())
V["alias_pairs"] = f"{len(pd.read_csv(P0C / 'alias_candidates.csv')):,}"
ext = pd.read_csv(P0C / "datasets_stratifiers_ext.csv", float_precision="round_trip")
orig_fam = dict(zip(pd.concat([p0b, ext]).openml_id, pd.concat([p0b, ext]).family))
V["fam_target_without_review"] = str(tr.openml_id.map(orig_fam).nunique())
alt = pd.read_csv(P0C / "alt_stratifiers.csv")
cu = alt[~np.isfinite(alt.neff_cluster)]
V["n_cluster_undef"] = str(len(cu))
V["cluster_undef_ids"] = ", ".join(str(int(o)) for o in cu.openml_id)
assert np.isfinite(alt.neff_cond).all() and len(alt) == len(fin_inc)
env = json.loads((P0C / "environment_phase1.json").read_text())
V["env_lock_sha"] = hashlib.sha256((P0C / "environment_phase1.lock.txt").read_bytes()).hexdigest()
V["env_python"] = env["python_version"]
pk = env["packages"]
V["env_packages"] = ", ".join(f"{k} {pk[k]}" for k in ["scikit-learn", "numpy", "scipy", "pandas", "tabpfn", "tabicl", "interpret",
                                                     "catboost", "lightgbm", "xgboost", "torch", "rtdl-revisiting-models", "threadpoolctl"])
V["env_gpu"] = env["gpu_model"]
V["env_cuda"] = env["cuda_runtime"]
ck = env["checkpoints"]
V["ckpt_tabpfn35"] = ck["tabpfn35"]["sha256"]
k2 = [k for k in ck if k.startswith("tabpfn2")][0]
V["ckpt_tabpfn2"] = ck[k2]["sha256"]
V["ckpt_tabpfn2_file"] = ck[k2]["filename"]
V["ckpt_tabicl2"] = ck["tabicl2"]["sha256"]
cp = (P0C / "compute_projection.md").read_text()
tot = re.search(r"\|\s*\*\*Total\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|\s*\*\*([\d.]+)\*\*", cp)
V["cpu_core_hours"] = f"{float(tot.group(1)):,.0f}"
V["gpu_hours"] = f"{float(tot.group(2)):.0f}"
V["ebm_share"] = re.search(r"([\d.]+)% of all projected traditional-model CPU core-hours", cp).group(1) + "%"
pc = pd.read_csv(P0C / "pilot_C.csv")
for fm in ("tabpfn35", "tabicl2", "tabpfn2"):
    V[f"pilot_sd_{fm}"] = f"{pc.loc[pc.foundation_model == fm, 'C_i'].std(ddof=1):.4f}"
# pre-flight (run before registration; see phase1_preflight_brief.md). Until all its deliverables and the
# compute plan exist and pass the checks below, the plan carries a visible pending marker and must not be
# registered (make_manifest.py refuses to run while the marker is present).
import hashlib  # noqa: E402

PF = ROOT / "preflight_outputs"
PLAN = ROOT / "frozen" / "compute_plan.json"
PF_KEYS = ("pf_hw", "pf_lr", "pf_ebm", "pf_all", "pf_wall", "pf_wall_one_all", "scope_decision", "m3_share", "m5_share",
           "compute_limit", "n_instances_word", "w_all", "w_nom5", "w_nom3")
pf_files = [PF / "lr_reference_phase1_preflight.csv", PF / "preflight_timing.csv", PF / "preflight_report.md",
            ROOT / "pipeline_code_sha256.txt", PLAN]
if all(f.exists() for f in pf_files) and (ROOT / "pipeline_code").is_dir():
    assert (PF / "preflight_report.md").read_text(encoding="utf-8").lstrip().startswith("完成"), "pre-flight did not complete"
    for ln in (ROOT / "pipeline_code_sha256.txt").read_text().splitlines():
        if ln.strip():
            h, f = ln.split(None, 1)
            f = f.lstrip("*")
            assert hashlib.sha256((ROOT / "pipeline_code" / f).read_bytes()).hexdigest() == h, f"pipeline hash mismatch: {f}"
    plan = json.loads(PLAN.read_text())
    lr = pd.read_csv(PF / "lr_reference_phase1_preflight.csv", float_precision="round_trip")
    fp_ids = set(pd.read_csv(ROOT / "final_pool" / "datasets_stratifiers.csv").query("excluded_reason.isna()").openml_id)
    assert set(lr.openml_id) == fp_ids and len(lr) == len(fp_ids), "pre-flight LR reference does not cover the final pool"
    ref0c = pd.read_csv(P0C / "lr_reference_phase1.csv", float_precision="round_trip").set_index("openml_id").lr_auc
    d_full = (lr.set_index("openml_id").lr_auc - ref0c.reindex(lr.openml_id).values).abs()
    assert d_full.notna().all()
    d = lr.abs_diff_vs_phase0c_reference.astype(float)
    tu = lr.tie_units.astype(float)
    assert np.isfinite(d).all() and np.isfinite(tu).all()
    n_ex = int((d <= 1e-6).sum())
    n_tie = int(((d > 1e-6) & ((tu - tu.round()).abs() <= 1e-3) & (d < 1e-3)).sum())
    assert n_ex + n_tie == len(lr), "pre-flight LR reference violates the tie rule"
    cpus = sorted(set(lr.cpu_model.astype(str)))
    assert len(cpus) == 1, cpus
    tim = pd.read_csv(PF / "preflight_timing.csv")
    assert (tim.model == "ebm").sum() > 0 and np.isfinite(tim.ratio).all(), "timing file lacks EBM rows or finite ratios"
    r_ebm = float(tim.loc[tim.model == "ebm", "ratio"].median())
    N, c = int(plan["instances"]), int(plan["allocated_cores_per_instance"])
    mods = re.findall(r"^\|\s*(M\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", cp, flags=re.M)
    assert [m[0] for m in mods] == ["M1", "M2", "M3", "M4", "M5", "M6"], mods
    drop = set(plan.get("modules_not_run", []))
    assert drop <= {"M3", "M5"}, drop
    wall = sum(max(float(cpu) * r_ebm / (N * c), float(gpu)) for m, cpu, gpu in mods if m not in drop)
    wall1 = sum(max(float(cpu) * r_ebm / c, float(gpu)) for _, cpu, gpu in mods)

    def _w(excl):
        h = sum(max(float(cpu) * r_ebm / (N * c), float(gpu)) for m, cpu, gpu in mods if m not in excl)
        return f"about {h:.0f} hours ({h / 24:.1f} days)"

    V["w_all"], V["w_nom5"], V["w_nom3"] = _w(set()), _w({"M5"}), _w({"M3"})
    lim = float(plan["compute_limit_hours"])
    V["compute_limit"] = f"about {lim / 168:.0f} weeks ({lim:.0f} hours of wall-clock time)"
    V["n_instances_word"] = {1: "one", 2: "two", 3: "three"}.get(N, str(N))
    cpu_tot = sum(float(cpu) for _, cpu, _ in mods)
    share = {m: float(cpu) / cpu_tot for m, cpu, _ in mods}
    V["pf_wall_one_all"] = f"about {wall1:,.0f} wall-clock hours (about {wall1 / 24:.0f} days)"
    V["m3_share"] = f"{100 * share['M3']:.0f}%"
    V["m5_share"] = f"{100 * share['M5']:.0f}%"
    kept = [m for m, _, _ in mods if m not in drop]
    V["scope_decision"] = ((f"{' and '.join(sorted(drop))} {'is' if len(drop) == 1 else 'are'} not run"
                            + (", so E1 is not reported" if "M3" in drop else "") + "; " if drop else "")
                           + (f"a second instance of the same CPU model is added; " if N == 2 else
                              (f"{N} instances of the same CPU model are used; " if N > 2 else ""))
                           + f"{', '.join(kept)} run, with three injection replicates.")
    V["pf_hw"] = (f"{'one rented instance' if N == 1 else f'{N} rented instances of one CPU type'} "
                  f"({cpus[0]}; {c} vCPUs allocated {'to it' if N == 1 else 'to each'}, one single-thread fit per vCPU)")
    if (d_full == 0).all():
        V["pf_lr"] = f"all {len(lr)} values are identical to the Phase 0c reference (compared at full precision)"
    else:
        V["pf_lr"] = (f"{n_ex} of {len(lr)} datasets within 1e-6 of the Phase 0c reference ({int((d_full == 0).sum())} identical), "
                      f"{n_tie} differing by whole tie units, largest difference {d_full.max():.2e}")
    V["pf_ebm"] = f"{r_ebm:.2f}"
    V["pf_all"] = f"{float(tim.ratio.median()):.2f}"
    V["pf_wall"] = (f"at least {wall:.0f} wall-clock hours (about {wall / 24:.1f} days) on "
                    f"{'this instance' if N == 1 else f'{N} instances'}, taking each module in turn with its CPU "
                    f"and GPU work in parallel and assuming perfect load balance")
else:
    print("WARNING: pre-flight deliverables or frozen/compute_plan.json missing; plan is marked pending")
    for k in PF_KEYS:
        V[k] = "[PENDING PRE-FLIGHT]"

med = pd.read_csv(ROOT / "frozen" / "medical_labels.csv")
mt = tr[tr.openml_id.isin(med.loc[med.label != "none", "openml_id"])]
V["med_target"] = str(len(mt))
V["med_target_fam"] = str(mt.family.nunique())

# typographic minus for signed numbers (the rest of the plan uses U+2212)
V = {k: re.sub(r"(?<![\w.])-(?=\d)", "\u2212", v) for k, v in V.items()}

out = src
for k, v in V.items():
    out = out.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[^}]+\}\}", out)
(ROOT / "docs" / "preregistration_v2.md").write_text(out)
print("filled", len(V), "remaining placeholders:", left)
print(json.dumps({k: v for k, v in V.items() if not k.endswith("rows")}, indent=1))
