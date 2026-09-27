from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


FLAGGED = [1063, 336, 44467, 44464, 44466, 43947, 44447, 993, 44463, 1002, 44413]
PHASE0B_HASHES = {
    "datasets_stratifiers.csv": "5bb2a3ce033a2a112e58024eea26abb8fc6d909a4fb293d871cfeccdfe2169de",
    "marginal_aucs.json": "0c6d23707fb6ecf93ba39d8e3421398fef91b103097ffe0762781ffacbcc2780",
    "environment.json": "ccac64ae468720478a84b9526ce264c15ececda5fddb49bc9f9791a8f5abe909",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_phase0b(phase0b_dir: Path) -> dict[str, str]:
    actual = {name: sha256(phase0b_dir / name) for name in PHASE0B_HASHES}
    bad = {name: (actual[name], expected) for name, expected in PHASE0B_HASHES.items() if actual[name] != expected}
    if bad:
        raise RuntimeError(f"STOP Phase 0b hash mismatch: {bad}")
    return actual


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def load_base_runner(root: Path):
    return load_module(root / "phase0c_pool_runner.py", "phase0c_pool_runner_resume_base")


def included_tables(root: Path, phase0b_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(phase0b_dir / "datasets_stratifiers.csv", float_precision="round_trip")
    ext = pd.read_csv(root / "phase0c_outputs" / "datasets_stratifiers_ext.csv", float_precision="round_trip")
    old_inc = old.loc[old.excluded_reason.fillna("").eq("")].copy()
    new_inc = ext.loc[ext.excluded_reason.fillna("").eq("")].copy()
    combined = pd.concat([old_inc.assign(pool="old"), new_inc.assign(pool="new")], ignore_index=True)
    if len(old_inc) != 364 or len(new_inc) != 114 or len(combined) != 478:
        raise RuntimeError(f"STOP pool counts old={len(old_inc)} new={len(new_inc)} total={len(combined)}")
    return old_inc, new_inc, combined


def run_r1(root: Path, phase0b_dir: Path, old_script: Path) -> None:
    verify_phase0b(phase0b_dir)
    base = load_base_runner(root)
    old = base.get_old(str(old_script))
    frozen = json.loads((phase0b_dir / "marginal_aucs.json").read_text(encoding="utf-8"))
    cache = root / "openml_cache"
    rows: list[dict[str, Any]] = []
    for did in FLAGGED:
        _dataset, X, y, _info, categorical_map = base.fetch_dataset_clean(did, cache, old)
        with threadpool_limits(limits=1):
            _metrics, marginal = old.calculate_metrics(X, y, categorical_map)
        expected = frozen[str(did)]
        if set(marginal) != set(expected):
            raise RuntimeError(f"STOP R1 feature mismatch for {did}")
        error = max(abs(float(marginal[name]) - float(expected[name])) for name in expected)
        rows.append({"openml_id": did, "n_features": len(expected), "marginal_all_max_abs_error": error})
        print(f"R1 {did} features={len(expected)} max_abs_error={error:.17g}", flush=True)
    frame = pd.DataFrame(rows)
    atomic_csv(root / "phase0c_outputs" / "reproduction_flagged_marginals.csv", frame)
    if (frame.marginal_all_max_abs_error > 1e-9).any():
        bad = frame.loc[frame.marginal_all_max_abs_error > 1e-9].to_dict("records")
        raise RuntimeError(f"STOP R1 marginal reproduction failed: {bad}")


def normalize_alias_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"(?:-dataset|-data|-v\d+|-version-?\d+)$", "", value)


def normalized_description(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()[:8000]


def run_metadata(root: Path, phase0b_dir: Path, code_dir: Path) -> None:
    verify_phase0b(phase0b_dir)
    old_inc, new_inc, _combined = included_tables(root, phase0b_dir)
    records_path = root / "run_state" / "metadata_checkpoint.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if len(records) != 478:
        raise RuntimeError(f"STOP metadata checkpoint count={len(records)} expected=478")
    phase0c = load_module(code_dir / "phase0c_functions.py", "phase0c_functions_resume_metadata")
    rows: list[dict[str, Any]] = []
    descriptions: dict[str, str] = {}
    for key in sorted(records, key=int):
        rec = records[key]
        tags = rec.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        description = str(rec.get("description") or "")
        terms = phase0c.medical_candidate(str(rec.get("name") or ""), description, tags)
        creator = rec.get("creator")
        rows.append({
            "openml_id": int(key), "name": rec.get("name", ""), "version": rec.get("version"),
            "tags": ";".join(map(str, tags)), "original_data_url": rec.get("original_data_url") or "",
            "citation": rec.get("citation") or "", "collection_date": rec.get("collection_date") or "",
            "creator": ";".join(map(str, creator)) if isinstance(creator, list) else (creator or ""),
            "medical_terms": ";".join(terms), "n_numeric": rec["n_numeric"],
            "n_categorical": rec["n_categorical"], "missing_share": rec["missing_share"],
            "n1": rec["n1"], "n0": rec["n0"],
        })
        descriptions[key] = description
    out = root / "phase0c_outputs"
    atomic_csv(out / "metadata.csv", pd.DataFrame(rows).sort_values("openml_id"))
    atomic_json(out / "metadata_descriptions.json", descriptions)

    new_ids = set(new_inc.openml_id.astype(int))
    recs = {int(k): v for k, v in records.items()}
    ids = sorted(recs)
    pairs: list[dict[str, Any]] = []
    for pos, a in enumerate(ids):
        for b in ids[pos + 1:]:
            if not ({a, b} & new_ids):
                continue
            ra, rb = recs[a], recs[b]
            reasons: list[str] = []
            ua, ub = str(ra.get("original_data_url") or "").strip(), str(rb.get("original_data_url") or "").strip()
            if ua and ub and ua == ub:
                reasons.append("same_original_data_url")
            na, nb = normalize_alias_name(str(ra["name"])), normalize_alias_name(str(rb["name"]))
            name_ratio = difflib.SequenceMatcher(None, na, nb).ratio()
            same_shape = int(ra["n"]) == int(rb["n"]) and int(ra["p_used"]) == int(rb["p_used"])
            if same_shape and name_ratio >= 0.72:
                reasons.append(f"same_n_p_similar_name:{name_ratio:.3f}")
            if na == nb and str(ra["name"]).lower() != str(rb["name"]).lower():
                reasons.append("normalized_name_variant")
            da, db = normalized_description(str(ra.get("description") or "")), normalized_description(str(rb.get("description") or ""))
            if len(da) >= 120 and len(db) >= 120:
                ratio = difflib.SequenceMatcher(None, da, db).ratio()
                if ratio >= 0.92:
                    reasons.append(f"high_description_similarity:{ratio:.3f}")
            if reasons:
                pairs.append({"openml_id_a": a, "name_a": ra["name"], "openml_id_b": b, "name_b": rb["name"], "reason": ";".join(reasons)})
    atomic_csv(out / "alias_candidates.csv", pd.DataFrame(pairs, columns=["openml_id_a", "name_a", "openml_id_b", "name_b", "reason"]))
    print(f"METADATA rows={len(rows)} alias_candidates={len(pairs)}", flush=True)


def transformed_groups(pipeline: Any, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    preprocess = pipeline.named_steps["preprocess"]
    groups: list[str] = []
    if numeric_cols:
        imputer = preprocess.named_transformers_["numeric"].named_steps["imputer"]
        groups.extend([name for name, stat in zip(numeric_cols, imputer.statistics_) if not pd.isna(stat)])
    if categorical_cols:
        catpipe = preprocess.named_transformers_["categorical"]
        imputer = catpipe.named_steps["imputer"]
        retained = [name for name, stat in zip(categorical_cols, imputer.statistics_) if not pd.isna(stat)]
        onehot = catpipe.named_steps["onehot"]
        for name, levels in zip(retained, onehot.categories_):
            groups.extend([name] * len(levels))
    return groups


def load_marginals(root: Path, phase0b_dir: Path) -> dict[str, dict[str, float]]:
    return {
        **json.loads((phase0b_dir / "marginal_aucs.json").read_text(encoding="utf-8")),
        **json.loads((root / "phase0c_outputs" / "marginal_aucs_ext.json").read_text(encoding="utf-8")),
    }


def run_alt(root: Path, phase0b_dir: Path, old_script: Path, code_dir: Path) -> None:
    verify_phase0b(phase0b_dir)
    _old_inc, _new_inc, combined = included_tables(root, phase0b_dir)
    base = load_base_runner(root)
    old = base.get_old(str(old_script))
    phase0c = load_module(code_dir / "phase0c_functions.py", "phase0c_functions_resume_alt")
    marginals = load_marginals(root, phase0b_dir)
    state = root / "run_state" / "resume"
    state.mkdir(parents=True, exist_ok=True)
    checkpoint = state / "alt_checkpoint.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    done = {int(r["openml_id"]) for r in rows}
    clusters_path = state / "clusters_checkpoint.json"
    coefs_path = state / "lr_coefficients_checkpoint.json"
    clusters_all = json.loads(clusters_path.read_text(encoding="utf-8")) if clusters_path.exists() else {}
    coef_all = json.loads(coefs_path.read_text(encoding="utf-8")) if coefs_path.exists() else {}
    for row in combined.itertuples(index=False):
        did = int(row.openml_id)
        if did in done:
            continue
        started = time.perf_counter()
        df = pd.read_parquet(root / "cleaned_data" / f"{did}.parquet")
        schema = json.loads((root / "cleaned_data" / f"{did}.schema.json").read_text(encoding="utf-8"))
        names = list(marginals[str(did)].keys())
        X = df[names].copy(); y = df.y.to_numpy(dtype=int); folds = df.fold.to_numpy(dtype=int)
        categorical_map = {name: schema["columns"][name]["type"] == "categorical" for name in names}
        clusters = phase0c.feature_clusters(X.to_numpy(dtype=float), names)
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        neff_cluster = float(phase0c.neff_cluster(marginals[str(did)], clusters, n1, n0))
        X_prepared, numeric_cols, categorical_cols = old.prepare_feature_frame(X, categorical_map)
        per_fold: list[dict[str, np.ndarray]] = []
        audits: list[dict[str, Any]] = []
        for fold in range(5):
            train, held = folds != fold, folds == fold
            pipeline = old.build_lr_pipeline(numeric_cols, categorical_cols)
            with threadpool_limits(limits=1), warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                pipeline.fit(X_prepared.loc[train], y[train])
                Xt = pipeline.named_steps["preprocess"].transform(X_prepared.loc[held])
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            Xt = np.asarray(Xt, dtype=float)
            coef = np.asarray(pipeline.named_steps["model"].coef_, dtype=float).ravel()
            groups = transformed_groups(pipeline, numeric_cols, categorical_cols)
            if len(groups) != Xt.shape[1] or len(coef) != Xt.shape[1]:
                raise RuntimeError(f"STOP transformed group mismatch {did} fold={fold}")
            per_fold.append(phase0c.feature_contributions(coef, Xt, groups))
            audits.append({"fold": fold, "coef": coef.tolist(), "intercept": np.asarray(pipeline.named_steps["model"].intercept_).ravel().tolist(), "groups": groups})
        neff_cond = float(phase0c.neff_conditional(per_fold, names))
        rows.append({"openml_id": did, "neff_cluster": neff_cluster, "neff_cond": neff_cond, "n_clusters": len(set(clusters.values())), "p_used": len(names)})
        clusters_all[str(did)] = clusters; coef_all[str(did)] = audits
        atomic_csv(checkpoint, pd.DataFrame(rows).sort_values("openml_id"))
        atomic_json(clusters_path, clusters_all); atomic_json(coefs_path, coef_all)
        done.add(did)
        print(f"ALT {did} OK seconds={time.perf_counter()-started:.3f}", flush=True)
    frame = pd.DataFrame(rows).sort_values("openml_id")
    if len(frame) != 478:
        raise RuntimeError(f"STOP alt rows={len(frame)}")
    notes: list[str] = []
    for record in frame.itertuples(index=False):
        did = int(record.openml_id)
        cluster_value = float(record.neff_cluster)
        conditional_value = float(record.neff_cond)
        if not np.isfinite(conditional_value) or conditional_value <= 0:
            raise RuntimeError(f"STOP invalid neff_cond openml_id={did}: {conditional_value}")
        if np.isfinite(cluster_value) and cluster_value > 0:
            notes.append("")
            continue
        df = pd.read_parquet(root / "cleaned_data" / f"{did}.parquet", columns=["y"])
        y = df.y.to_numpy(dtype=int)
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        aucs = marginals[str(did)]
        lifts = np.asarray(phase0c.stratifier.lifts_dn(list(aucs.values()), n1, n0), dtype=float)
        if not np.isfinite(cluster_value) and np.all(lifts <= 0):
            notes.append("no feature above the denoising threshold")
            continue
        raise RuntimeError(
            f"STOP invalid neff_cluster despite feature above threshold openml_id={did}: "
            f"value={cluster_value}, positive_lifts={int((lifts > 0).sum())}"
        )
    frame["note"] = notes
    out = root / "phase0c_outputs"
    atomic_csv(out / "alt_stratifiers.csv", frame)
    atomic_json(out / "clusters.json", clusters_all); atomic_json(out / "lr_coefficients.json", coef_all)


def cpu_details() -> dict[str, Any]:
    text = subprocess.check_output(["lscpu"], text=True)
    items = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            items[key.strip()] = value.strip()
    logical = int(items.get("CPU(s)", os.cpu_count() or 0))
    sockets = int(items.get("Socket(s)", 1)); cores_per_socket = int(items.get("Core(s) per socket", logical))
    memory_bytes = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory_bytes = int(line.split()[1]) * 1024
            break
    return {"model_name": items.get("Model name", platform.processor()), "physical_cores": sockets * cores_per_socket, "logical_cores": logical, "memory_bytes": memory_bytes, "lscpu": items}


def run_environment(root: Path) -> None:
    path = root / "phase0c_outputs" / "environment_phase1.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    details = cpu_details()
    env["cpu_model"] = details["model_name"]
    env["cpu_physical_cores"] = details["physical_cores"]
    env["cpu_logical_cores"] = details["logical_cores"]
    env["memory_bytes"] = details["memory_bytes"]
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max_path.exists():
        quota, period = cpu_max_path.read_text().strip().split()
        env["allocated_cpu_cores"] = None if quota == "max" else float(quota) / float(period)
    else:
        env["allocated_cpu_cores"] = os.cpu_count()
    memory_max_path = Path("/sys/fs/cgroup/memory.max")
    if memory_max_path.exists():
        memory_limit = memory_max_path.read_text().strip()
        env["allocated_memory_bytes"] = None if memory_limit == "max" else int(memory_limit)
    env["threads"] = {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "traditional_models": 1, "lr_threadpoolctl": 1,
    }
    env.setdefault("categorical_parameter_methods", {})["xgb"] = "enable_categorical=True with pandas category dtype"
    env["machine_instances"] = {"cpu_instances": 1, "gpu_instances": 1}
    atomic_json(path, env)
    print(json.dumps({k: env[k] for k in ("cpu_model", "cpu_physical_cores", "cpu_logical_cores", "memory_bytes", "allocated_cpu_cores", "allocated_memory_bytes", "threads")}, indent=2))


def run_lr_reference(root: Path, phase0b_dir: Path, old_script: Path) -> None:
    verify_phase0b(phase0b_dir)
    old_inc, new_inc, combined = included_tables(root, phase0b_dir)
    base = load_base_runner(root); old = base.get_old(str(old_script))
    marginals = load_marginals(root, phase0b_dir)
    frozen = {int(r.openml_id): float(r.auc_lr) for r in old_inc.itertuples(index=False)}
    frozen.update({int(r.openml_id): float(r.auc_lr) for r in new_inc.itertuples(index=False)})
    checkpoint = root / "run_state" / "resume" / "lr_reference_checkpoint.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    done = {int(r["openml_id"]) for r in rows}
    cpu_model = cpu_details()["model_name"]
    for row in combined.itertuples(index=False):
        did = int(row.openml_id)
        if did in done:
            continue
        started = time.perf_counter()
        df = pd.read_parquet(root / "cleaned_data" / f"{did}.parquet")
        schema = json.loads((root / "cleaned_data" / f"{did}.schema.json").read_text(encoding="utf-8"))
        names = list(marginals[str(did)].keys())
        X = df[names].copy(); y = df.y.to_numpy(dtype=int); folds = df.fold.to_numpy(dtype=int)
        categorical_map = {name: schema["columns"][name]["type"] == "categorical" for name in names}
        X_prepared, numeric_cols, categorical_cols = old.prepare_feature_frame(X, categorical_map)
        oof = np.full(len(y), np.nan, dtype=float)
        for fold in range(5):
            train, held = folds != fold, folds == fold
            pipeline = old.build_lr_pipeline(numeric_cols, categorical_cols)
            with threadpool_limits(limits=1), warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                pipeline.fit(X_prepared.loc[train], y[train])
                oof[held] = pipeline.predict_proba(X_prepared.loc[held])[:, 1]
        auc = float(roc_auc_score(y, oof)); diff = abs(auc - frozen[did])
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        unit = 0.5 / (n1 * n0); ties = diff / unit
        rows.append({"openml_id": did, "lr_auc": auc, "lr_abs_diff_vs_frozen": diff, "tie_units": ties, "cpu_model": cpu_model, "n_threads": 1})
        atomic_csv(checkpoint, pd.DataFrame(rows).sort_values("openml_id"))
        done.add(did)
        print(f"LRREF {did} diff={diff:.17g} tie_units={ties:.9f} seconds={time.perf_counter()-started:.3f}", flush=True)
    frame = pd.DataFrame(rows).sort_values("openml_id")
    if len(frame) != 478:
        raise RuntimeError(f"STOP R6 rows={len(frame)}")
    bad = frame[(frame.lr_abs_diff_vs_frozen > 1e-6) & ((np.abs(frame.tie_units - np.rint(frame.tie_units)) > 1e-3) | (frame.lr_abs_diff_vs_frozen >= 1e-3))]
    atomic_csv(root / "phase0c_outputs" / "lr_reference_phase1.csv", frame)
    if len(bad):
        raise RuntimeError(f"STOP R6 criterion failed: {bad.to_dict('records')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["r1", "metadata", "alt", "environment", "lrref"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase0b-dir", required=True, type=Path)
    parser.add_argument("--old-script", required=True, type=Path)
    parser.add_argument("--code-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); phase0b = args.phase0b_dir.resolve(); old_script = args.old_script.resolve(); code_dir = args.code_dir.resolve()
    if args.action == "r1": run_r1(root, phase0b, old_script)
    elif args.action == "metadata": run_metadata(root, phase0b, code_dir)
    elif args.action == "alt": run_alt(root, phase0b, old_script, code_dir)
    elif args.action == "environment": run_environment(root)
    else: run_lr_reference(root, phase0b, old_script)


if __name__ == "__main__":
    main()
