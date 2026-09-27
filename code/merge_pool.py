"""Merge the Phase 0b pool, the Phase 0c extension and the author's pool decisions into the final pool.

Usage:
  python3 merge_pool.py <phase0b_dir> <phase0c_dir> <pool_decisions.json> <out_dir>

Inputs
  phase0b_dir/datasets_stratifiers.csv, marginal_aucs.json            (frozen, never modified)
  phase0c_dir/datasets_stratifiers_ext.csv, marginal_aucs_ext.json, reproduction_check.csv,
              reproduction_flagged_marginals.csv (required when any error exceeds 1e-6)
  pool_decisions.json (written by the author before registration, without any model result):
    {"alias_merges":       {"<openml_id>": "<existing family label>", ...},
     "family_corrections": {"<family label>": "<family label it is merged into>", ...},
     "exclude":            {"<openml_id>": "<reason>", ...}}
Rules
  * Reproduction of the Phase 0b reference logistic regression (preregistration Section 3.3).
    For each Phase 0b included dataset, err = |AUROC(Phase 0c) - auc_lr(Phase 0b)|:
      - err missing or not finite                      -> excluded ("reproduction_missing");
      - err <= 1e-6                                     -> retained;
      - err  > 1e-6 is retained ("tie_order_difference") only if ALL of:
          (a) every marginal AUROC of the dataset reproduces within 1e-9
              (reproduction_flagged_marginals.csv), so data, labels and folds are identical;
          (b) err is a whole multiple of the tie unit 0.5 / (n1 * n0), i.e. only the order of
              tied or near-tied pairs changed (|k - round(k)| <= 1e-3 and round(k) >= 1);
          (c) err < 1e-3;
          (d) target-region membership (auc_lr in [0.60, 0.85]) is the same at auc_lr - err and
              auc_lr + err;
        otherwise excluded ("reproduction_failure: <which condition failed>").
    A flagged dataset without a full marginal check is an error, and so is a reproduction file
    without the lr_abs_error column.
  * alias_merges relabels the family of a NEW dataset to an existing family.
  * family_corrections then relabels every dataset (Phase 0b or new) that carries a given family
    label, merging two labels that denote the same underlying records. Applied after alias_merges.
  * exclude marks a dataset as excluded ("author_decision: <reason>").
  * family_size is recomputed over included datasets.
Outputs (out_dir): datasets_stratifiers.csv and marginal_aucs.json with the same schema as Phase 0b,
so that build_frozen_stratifier.py and build_frozen_plans.py run on them unchanged; merge_report.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPRO_TOL = 1e-6
MARGINAL_TOL = 1e-9
TIE_INTEGER_TOL = 1e-3
TIE_MAX_ERR = 1e-3
TARGET = (0.60, 0.85)


def classify_reproduction(err, n1, n0, auc_lr, marginal_full_err):
    """Return (retained: bool, label: str) for one Phase 0b dataset (rules in the module docstring)."""
    if err is None or not np.isfinite(err):
        return False, "reproduction_missing"
    if err <= REPRO_TOL:
        return True, "ok"
    if marginal_full_err is None or not np.isfinite(marginal_full_err):
        raise ValueError("a dataset with lr_abs_error > 1e-6 needs a full marginal reproduction check")
    if marginal_full_err > MARGINAL_TOL:
        return False, "reproduction_failure: marginal AUROCs differ (data, labels or folds)"
    k = err / (0.5 / (n1 * n0))
    if round(k) < 1 or abs(k - round(k)) > TIE_INTEGER_TOL:
        return False, "reproduction_failure: not a whole number of tie units"
    if err >= TIE_MAX_ERR:
        return False, "reproduction_failure: error >= 1e-3"
    inside = lambda a: TARGET[0] <= a <= TARGET[1]  # noqa: E731
    if inside(auc_lr - err) != inside(auc_lr + err):
        return False, "reproduction_failure: target-region membership not robust"
    return True, "tie_order_difference"


def merge(p0b: Path, p0c: Path, decisions: dict, out: Path) -> dict:
    old = pd.read_csv(p0b / "datasets_stratifiers.csv", float_precision="round_trip")
    new = pd.read_csv(p0c / "datasets_stratifiers_ext.csv", float_precision="round_trip")
    if list(new.columns) != list(old.columns):
        raise ValueError("extension table must have exactly the Phase 0b columns")
    dup = set(old.openml_id) & set(new.openml_id)
    if dup:
        raise ValueError(f"datasets present in both tables: {sorted(dup)[:10]}")
    m_old = json.loads((p0b / "marginal_aucs.json").read_text())
    m_new = json.loads((p0c / "marginal_aucs_ext.json").read_text())
    rep = pd.read_csv(p0c / "reproduction_check.csv")
    new = new.copy()
    new["source_suite"] = new["source_suite"].fillna("openml-scan-ext")
    for oid, fam in decisions.get("alias_merges", {}).items():
        oid = int(oid)
        if oid not in set(new.openml_id):
            raise ValueError(f"alias merge for {oid}, which is not a new dataset")
        if fam not in set(old.family) | set(new.family):
            raise ValueError(f"alias target family {fam!r} does not exist")
        new.loc[new.openml_id == oid, "family"] = fam
    allp = pd.concat([old, new], ignore_index=True)
    corr = decisions.get("family_corrections", {})
    for src, dst in corr.items():
        if src not in set(allp.family):
            raise ValueError(f"family correction source {src!r} does not exist")
        if dst not in set(allp.family) or dst in corr:
            raise ValueError(f"family correction target {dst!r} must be an existing label that is not itself corrected")
    allp["family"] = allp.family.replace(corr)
    if "lr_abs_error" not in rep.columns or "openml_id" not in rep.columns:
        raise ValueError("reproduction_check.csv must contain the columns openml_id and lr_abs_error")
    err = dict(zip(rep.openml_id.astype(int), pd.to_numeric(rep.lr_abs_error, errors="coerce")))
    fm_path = p0c / "reproduction_flagged_marginals.csv"
    fm = {}
    if fm_path.exists():
        f = pd.read_csv(fm_path)
        fm = dict(zip(f.openml_id.astype(int), pd.to_numeric(f.marginal_all_max_abs_error, errors="coerce")))
    bad, missing_rep, ties = [], [], []
    for _, r in old.loc[old.excluded_reason.isna()].iterrows():
        oid = int(r.openml_id)
        n1 = int(round(r.n * r.minority_rate))
        n0 = int(r.n) - n1
        e = err.get(oid, float("nan"))
        keep, label = classify_reproduction(e, n1, n0, float(r.auc_lr), fm.get(oid))
        if label == "reproduction_missing":
            missing_rep.append(oid)
        elif not keep:
            bad.append({"openml_id": oid, "lr_abs_error": e, "reason": label})
        elif label == "tie_order_difference":
            ties.append({"openml_id": oid, "lr_abs_error": e, "tie_units": round(e / (0.5 / (n1 * n0)))})
        if not keep:
            sel = (allp.openml_id == oid) & allp.excluded_reason.isna()
            allp.loc[sel, "excluded_reason"] = label
    for oid, why in decisions.get("exclude", {}).items():
        sel = (allp.openml_id == int(oid)) & allp.excluded_reason.isna()
        allp.loc[sel, "excluded_reason"] = f"author_decision: {why}"
    inc = allp.excluded_reason.isna()
    allp["family_size"] = allp.family.map(allp[inc].family.value_counts()).fillna(0).astype(int)
    m_all = {**m_old, **m_new}
    missing = [int(o) for o in allp[inc].openml_id if str(int(o)) not in m_all]
    if missing:
        raise ValueError(f"included datasets without marginal AUROCs: {missing[:10]}")
    m_out = {str(int(o)): m_all[str(int(o))] for o in allp[inc].openml_id}
    out.mkdir(parents=True, exist_ok=True)
    allp.to_csv(out / "datasets_stratifiers.csv", index=False)
    (out / "marginal_aucs.json").write_text(json.dumps(m_out, indent=1))
    report = {"old_candidates": len(old), "new_candidates": len(new), "included": int(inc.sum()),
              "families_included": int(allp[inc].family.nunique()),
              "reproduction_failures": bad, "reproduction_missing": missing_rep,
              "retained_tie_order_differences": ties, "alias_merges": decisions.get("alias_merges", {}),
              "family_corrections": corr,
              "author_exclusions": decisions.get("exclude", {})}
    (out / "merge_report.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    p0b, p0c, dec, out = map(Path, sys.argv[1:5])
    print(json.dumps(merge(p0b, p0c, json.loads(dec.read_text()), out), indent=1))
