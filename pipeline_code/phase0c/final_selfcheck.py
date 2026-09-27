from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_HASHES = {
    "datasets_stratifiers.csv": "5bb2a3ce033a2a112e58024eea26abb8fc6d909a4fb293d871cfeccdfe2169de",
    "marginal_aucs.json": "0c6d23707fb6ecf93ba39d8e3421398fef91b103097ffe0762781ffacbcc2780",
    "environment.json": "ccac64ae468720478a84b9526ce264c15ececda5fddb49bc9f9791a8f5abe909",
}
TRADITIONAL = {"lr", "ebm", "catboost", "lgbm", "xgb", "rf"}
ALL_MODELS = TRADITIONAL | {"tabpfn35", "tabicl2", "tabpfn2", "ftt"}
FORBIDDEN_POOL = re.compile(
    r"TabPFN|TabICL|CatBoost|XGBoost|XGBClassifier|LightGBM|LGBMClassifier|"
    r"RandomForest|ExplainableBoosting|FTTransformer"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "phase0c_outputs"
    package = root / "project_resume/tabular_prior_v2"
    p0b = package / "phase0b_outputs"
    checks: dict[str, object] = {}

    actual = {name: sha256(p0b / name) for name in EXPECTED_HASHES}
    assert actual == EXPECTED_HASHES
    checks["phase0b_hashes"] = {"ok": True, "values": actual}

    r1 = pd.read_csv(out / "reproduction_flagged_marginals.csv")
    assert len(r1) == 11 and (r1.marginal_all_max_abs_error <= 1e-9).all()
    checks["r1"] = {"ok": True, "rows": len(r1), "max": float(r1.marginal_all_max_abs_error.max())}

    r6 = pd.read_csv(out / "lr_reference_phase1.csv")
    diff = r6.lr_abs_diff_vs_frozen.to_numpy(float)
    tie = r6.tie_units.to_numpy(float)
    bad = ((diff > 1e-6) & (np.abs(tie - np.rint(tie)) > 1e-3)) | (diff >= 1e-3)
    assert len(r6) == 478 and not bad.any() and set(r6.n_threads.astype(int)) == {1}
    checks["r6"] = {
        "ok": True,
        "rows": len(r6),
        "zero": int((diff == 0).sum()),
        "nonzero_lt_1e-6": int(((diff > 0) & (diff < 1e-6)).sum()),
        "integer_tie_gt_1e-6": int((diff > 1e-6).sum()),
        "max": float(diff.max()),
    }

    alt = pd.read_csv(out / "alt_stratifiers.csv")
    metadata = pd.read_csv(out / "metadata.csv")
    assert len(alt) == len(metadata) == 478
    cond_ok = np.isfinite(alt.neff_cond) & (alt.neff_cond > 0)
    cluster_ok = np.isfinite(alt.neff_cluster) & (alt.neff_cluster > 0)
    designed_na = alt.note.fillna("").eq("no feature above the denoising threshold")
    assert cond_ok.all() and (cluster_ok | designed_na).all()
    assert set(alt.loc[designed_na, "openml_id"].astype(int)) == {41538, 44776}
    assert len(json.loads((out / "clusters.json").read_text(encoding="utf-8"))) == 478
    assert len(json.loads((out / "lr_coefficients.json").read_text(encoding="utf-8"))) == 478
    checks["r4"] = {"ok": True, "rows": 478, "designed_nan_ids": [41538, 44776]}

    old = pd.read_csv(p0b / "datasets_stratifiers.csv")
    new = pd.read_csv(out / "datasets_stratifiers_ext.csv")
    pool = pd.concat([old, new], ignore_index=True)
    pool = pool[pool.excluded_reason.fillna("").eq("")].copy()
    marginals = {
        **json.loads((p0b / "marginal_aucs.json").read_text(encoding="utf-8")),
        **json.loads((out / "marginal_aucs_ext.json").read_text(encoding="utf-8")),
    }
    max_fold_rate_error = 0.0
    for row in pool.itertuples(index=False):
        did = int(row.openml_id)
        frame = pd.read_parquet(root / "cleaned_data" / f"{did}.parquet")
        assert len(frame) == int(row.n)
        assert abs(float(frame.y.mean()) - float(row.minority_rate)) <= 1e-12
        assert set(frame.fold.astype(int)) == {0, 1, 2, 3, 4}
        fold_error = float((frame.groupby("fold").y.mean() - frame.y.mean()).abs().max())
        max_fold_rate_error = max(max_fold_rate_error, fold_error)
        assert fold_error <= 0.02 + 1e-12
        features = set(frame.columns) - {"row_index", "fold", "y"}
        assert features == set(marginals[str(did)])
    checks["cleaned_data"] = {"ok": True, "rows": len(pool), "max_fold_rate_error": max_fold_rate_error}

    timing = pd.read_csv(out / "pilot_timing.csv")
    m2 = timing[(timing.stage == "outer") & timing.variant.ne("raw")]
    m1 = timing[(timing.stage == "outer") & timing.variant.eq("raw")]
    assert len(m2) == 12 * 13 * 5 * 9
    assert len(m1) == 12 * 5 * 10
    assert set(timing.dataset) == {f"pilot{i}" for i in range(12)}
    assert set(m1.model) == ALL_MODELS
    assert set(m2.model) == ALL_MODELS - {"ftt"}
    assert set(timing.loc[timing.model.isin(TRADITIONAL), "device"]) == {"CPU"}
    assert not timing[["fit_seconds", "predict_seconds"]].isna().any().any()
    assert (timing[["fit_seconds", "predict_seconds"]] >= 0).all().all()
    checks["pilot"] = {"ok": True, "m2_outer_rows": len(m2), "m1_outer_rows": len(m1), "datasets": 12}

    source = (root / "phase1_pilot.py").read_text(encoding="utf-8")
    required = ["n_jobs=1", "thread_count=1", "threadpool_limits(limits=1)"]
    assert all(item in source for item in required)
    checks["single_thread_source_audit"] = {"ok": True, "required": required}

    pool_sources = [root / "phase0c_pool_runner.py", root / "phase0c_resume_runner.py"]
    pool_matches = {p.name: FORBIDDEN_POOL.findall(p.read_text(encoding="utf-8")) for p in pool_sources}
    assert all(not values for values in pool_matches.values())
    checks["pool_model_grep"] = {"ok": True, "matches": pool_matches}

    required_outputs = [
        "datasets_stratifiers_ext.csv", "marginal_aucs_ext.json", "alias_candidates.csv",
        "run_log_ext.txt", "reproduction_check.csv", "metadata.csv", "metadata_descriptions.json",
        "clusters.json", "lr_coefficients.json", "alt_stratifiers.csv", "environment_phase1.json",
        "environment_phase1.lock.txt", "smoke_test.csv", "generator_fingerprints_phase1env.json",
        "reproduction_flagged_marginals.csv", "lr_reference_phase1.csv", "pilot_timing.csv",
        "pilot_results.csv", "pilot_C.csv", "compute_projection.md", "phase0c_report.md",
    ]
    missing = [name for name in required_outputs if not (out / name).is_file()]
    assert not missing
    checks["deliverables"] = {"ok": True, "count": len(required_outputs), "missing": missing}

    target = root / "run_state/final_selfcheck.json"
    target.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
