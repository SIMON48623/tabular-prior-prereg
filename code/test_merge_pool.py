"""Test merge_pool.py on a mock extension built from two real Phase 0b rows (ids changed)."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import merge_pool as M

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond: FAILS.append(name)

P0B = Path(__file__).resolve().parents[1] / "phase0b_outputs"
old = pd.read_csv(P0B / "datasets_stratifiers.csv", float_precision="round_trip")
m_old = json.loads((P0B / "marginal_aucs.json").read_text())
with tempfile.TemporaryDirectory() as td:
    td = Path(td); p0c = td / "p0c"; p0c.mkdir()
    inc = old[old.excluded_reason.isna()]
    ext = inc.iloc[:2].copy(); src_ids = ext.openml_id.tolist()
    ext["openml_id"] = [900001, 900002]; ext["family"] = ["newfam_a", "newfam_b"]; ext["name"] = ["mock_a", "mock_b"]
    ext.to_csv(p0c / "datasets_stratifiers_ext.csv", index=False)
    (p0c / "marginal_aucs_ext.json").write_text(json.dumps({"900001": m_old[str(src_ids[0])], "900002": m_old[str(src_ids[1])]}))
    # every old included dataset gets a row, except inc[3] (missing row); inc[2] is NaN;
    # inc[1]: 3 whole tie units, marginals exact -> retained; inc[4]: 2.5 tie units -> excluded;
    # inc[5]: 2 tie units but marginals differ -> excluded; inc[6]: whole tie units but >= 1e-3 -> excluded
    ids = inc.openml_id.tolist()
    def unit(i):
        r = inc.iloc[i]; n1 = int(round(r.n * r.minority_rate)); return 0.5 / (n1 * (int(r.n) - n1))
    errs = [0.0] * len(ids)
    errs[1] = 3 * unit(1); errs[2] = float("nan"); errs[4] = 2.5 * unit(4); errs[5] = 2 * unit(5)
    errs[6] = float(np.ceil(1e-3 / unit(6))) * unit(6)
    rep = pd.DataFrame({"openml_id": ids, "lr_abs_error": errs}).drop(index=3)
    rep.to_csv(p0c / "reproduction_check.csv", index=False)
    pd.DataFrame({"openml_id": [ids[1], ids[4], ids[5], ids[6]],
                  "marginal_all_max_abs_error": [0.0, 0.0, 1e-6, 0.0]}).to_csv(p0c / "reproduction_flagged_marginals.csv", index=False)
    fam_target = inc.family.iloc[5]
    fam_src, fam_dst = inc.family.iloc[7], inc.family.iloc[8]
    dec = {"alias_merges": {"900002": fam_target}, "exclude": {"900001": "duplicate of an existing dataset"},
           "family_corrections": {fam_src: fam_dst}}
    r = M.merge(P0B, p0c, dec, td / "final")
    f = pd.read_csv(td / "final" / "datasets_stratifiers.csv")
    m = json.loads((td / "final" / "marginal_aucs.json").read_text())
    check("new rows appended", len(f) == len(old) + 2)
    check("alias merge applied", f.loc[f.openml_id == 900002, "family"].item() == fam_target)
    check("family correction applied to every member", (f.family == fam_src).sum() == 0 and (f.loc[f.openml_id.isin(old.loc[old.family == fam_src, "openml_id"]), "family"] == fam_dst).all())
    check("author exclusion applied", f.loc[f.openml_id == 900001, "excluded_reason"].item().startswith("author_decision"))
    reason = dict(zip(f.openml_id, f.excluded_reason))
    check("whole tie units with exact marginals retained", pd.isna(reason[ids[1]]) and any(t["openml_id"] == ids[1] and t["tie_units"] == 3 for t in r["retained_tie_order_differences"]))
    check("fractional tie units excluded", str(reason[ids[4]]).startswith("reproduction_failure: not a whole number"))
    check("marginal mismatch excluded", str(reason[ids[5]]).startswith("reproduction_failure: marginal"))
    check("error >= 1e-3 excluded", str(reason[ids[6]]).startswith("reproduction_failure: error >= 1e-3"))
    check("NaN reproduction error excluded", reason[ids[2]] == "reproduction_missing")
    check("missing reproduction row excluded", reason[ids[3]] == "reproduction_missing")
    check("included count", r["included"] == len(inc) - 5 + 1, f"{r['included']}")
    # a flagged dataset without a full marginal check is an error
    pd.DataFrame({"openml_id": [ids[4], ids[5], ids[6]], "marginal_all_max_abs_error": [0.0, 1e-6, 0.0]}).to_csv(p0c / "reproduction_flagged_marginals.csv", index=False)
    try:
        M.merge(P0B, p0c, dec, td / "final3"); check("flagged dataset without marginal check raises", False)
    except ValueError:
        check("flagged dataset without marginal check raises", True)
    for bad_dec, label in [({"family_corrections": {"no-such-family": fam_dst}}, "unknown correction source raises"),
                           ({"family_corrections": {fam_src: "no-such-family"}}, "unknown correction target raises")]:
        try:
            M.merge(P0B, p0c, {**dec, **bad_dec}, td / "bad"); check(label, False)
        except ValueError:
            check(label, True)
    rep.drop(columns="lr_abs_error").to_csv(p0c / "reproduction_check.csv", index=False)
    try:
        M.merge(P0B, p0c, dec, td / "final2"); check("missing lr_abs_error column raises", False)
    except ValueError:
        check("missing lr_abs_error column raises", True)
    check("marginal file only included", set(m) == set(str(int(o)) for o in f[f.excluded_reason.isna()].openml_id))
    fs = f[f.excluded_reason.isna()].family.value_counts()
    check("family_size recomputed", all(f.loc[f.excluded_reason.isna(), "family_size"] == f.loc[f.excluded_reason.isna(), "family"].map(fs)))
    check("phase 0b file untouched", pd.read_csv(P0B / "datasets_stratifiers.csv", float_precision="round_trip").equals(old))
keep, lab = M.classify_reproduction(2 * 0.5 / (100 * 100), 100, 100, 0.85, 0.0)
check("membership straddling the 0.85 boundary excluded", (not keep) and "membership" in lab, lab)
keep, lab = M.classify_reproduction(2 * 0.5 / (100 * 100), 100, 100, 0.80, 0.0)
check("membership robust retained", keep and lab == "tie_order_difference", lab)
print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
