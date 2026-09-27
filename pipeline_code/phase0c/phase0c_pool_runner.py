from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold


EXPECTED_HASHES = {
    "datasets_stratifiers.csv": "5bb2a3ce033a2a112e58024eea26abb8fc6d909a4fb293d871cfeccdfe2169de",
    "marginal_aucs.json": "0c6d23707fb6ecf93ba39d8e3421398fef91b103097ffe0762781ffacbcc2780",
    "environment.json": "ccac64ae468720478a84b9526ce264c15ececda5fddb49bc9f9791a8f5abe909",
}
START_DID = 44598
SEED = 42


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_phase0b(phase0b_dir: Path) -> dict[str, str]:
    got = {name: sha256(phase0b_dir / name) for name in EXPECTED_HASHES}
    bad = {name: (EXPECTED_HASHES[name], got[name]) for name in got if got[name] != EXPECTED_HASHES[name]}
    if bad:
        raise RuntimeError(f"STOP phase0b hash mismatch: {bad}")
    return got


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def json_dump_atomic(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set)):
        return [scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


_OLD_CACHE: dict[str, Any] = {}


def get_old(path: str):
    if path not in _OLD_CACHE:
        _OLD_CACHE[path] = load_module(Path(path), f"phase0b_frozen_{os.getpid()}")
    return _OLD_CACHE[path]


def prepare_active_metadata(root: Path) -> pd.DataFrame:
    state = root / "run_state"
    state.mkdir(parents=True, exist_ok=True)
    path = state / "openml_active_metadata.csv"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    metadata = openml.datasets.list_datasets(status="active", output_format="dataframe")
    metadata.to_csv(path, index=False)
    return metadata


def prepare_extension_candidates(metadata: pd.DataFrame, old: Any) -> pd.DataFrame:
    metadata = metadata.copy()
    for column in [
        "did", "version", "MinorityClassSize", "NumberOfClasses", "NumberOfFeatures", "NumberOfInstances"
    ]:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    eligible = metadata.loc[
        metadata["NumberOfClasses"].eq(2)
        & metadata["NumberOfInstances"].between(200, 10_000, inclusive="both")
        & metadata["NumberOfFeatures"].between(5, 100, inclusive="both")
    ].copy()
    eligible["minority_rate_meta"] = eligible["MinorityClassSize"] / eligible["NumberOfInstances"]
    eligible = eligible.loc[eligible["minority_rate_meta"].between(0.05, 0.50, inclusive="both")].copy()
    eligible = eligible.sort_values(["name", "NumberOfInstances", "NumberOfFeatures", "version", "did"])
    eligible = eligible.drop_duplicates(
        subset=["name", "NumberOfInstances", "NumberOfFeatures"], keep="first"
    ).copy()
    ext = eligible.loc[eligible["did"].ge(START_DID)].sort_values("did").reset_index(drop=True)
    ext["source_suite"] = "openml-scan-ext"
    ext["candidate_priority"] = np.arange(1, len(ext) + 1)
    return old.build_family_annotations(ext)


def blank_ext_result(row: dict[str, Any], old: Any, reason: str) -> dict[str, Any]:
    result = old.blank_result(row, reason)
    return {key: result.get(key) for key in old.INTERNAL_COLUMNS}


def ext_worker(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float] | None, str | None]:
    old = get_old(payload["old_script"])
    openml.config.cache_directory = payload["cache"]
    row = payload["row"]
    reason = old.structure_exclusion(row)
    if reason:
        return blank_ext_result(row, old, reason), None, None
    did = int(row["did"])
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            dataset = openml.datasets.get_dataset(
                did, download_data=True, download_qualities=True, download_features_meta_data=True
            )
            X, y, info, categorical_map = old.clean_and_screen_data(dataset)
            actual_reason = old.validate_actual_screen(info)
            result = blank_ext_result(row, old, actual_reason or "")
            result.update(info)
            if actual_reason:
                return result, None, None
            metrics, marginal = old.calculate_metrics(X, y, categorical_map)
            result.update(metrics)
            return result, marginal if not metrics["excluded_reason"] else None, None
        except Exception as exc:  # every failure is retained and logged
            last = exc
            if attempt < 3:
                time.sleep(2**attempt)
    assert last is not None
    reason = re.sub(r"\s+", " ", f"processing_failure:{type(last).__name__}:{last}").strip()
    return blank_ext_result(row, old, reason), None, traceback.format_exc()


def run_scan(root: Path, old_script: Path, workers: int) -> None:
    out = root / "phase0c_outputs"
    state = root / "run_state"
    cache = root / "openml_cache"
    out.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    old = get_old(str(old_script))
    openml.config.cache_directory = str(cache)
    metadata = prepare_active_metadata(root)
    candidates = prepare_extension_candidates(metadata, old)
    candidates.to_csv(state / "extension_candidates.csv", index=False)
    checkpoint = state / "extension_checkpoint.csv"
    marginal_dir = state / "extension_marginals"
    marginal_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        done_df = pd.read_csv(checkpoint, low_memory=False)
        rows = done_df.to_dict("records")
        done = set(pd.to_numeric(done_df["openml_id"], errors="coerce").dropna().astype(int))
    else:
        rows, done = [], set()
    pending = [row for row in candidates.to_dict("records") if int(row["did"]) not in done]
    append_log(out / "run_log_ext.txt", f"scan_start active_max={int(pd.to_numeric(metadata.did).max())} candidates={len(candidates)} pending={len(pending)}")
    payloads = [{"old_script": str(old_script), "cache": str(cache), "row": row} for row in pending]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ext_worker, p): int(p["row"]["did"]) for p in payloads}
        for future in as_completed(futures):
            did = futures[future]
            try:
                result, marginal, error = future.result()
            except Exception as exc:
                source = candidates.loc[candidates.did.astype(int).eq(did)].iloc[0].to_dict()
                result = blank_ext_result(source, old, f"worker_failure:{type(exc).__name__}:{exc}")
                marginal, error = None, traceback.format_exc()
            rows.append(result)
            frame = pd.DataFrame(rows).reindex(columns=old.INTERNAL_COLUMNS).sort_values("candidate_priority")
            frame.to_csv(checkpoint, index=False)
            if marginal is not None:
                json_dump_atomic(marginal_dir / f"{did}.json", marginal)
            if error:
                (state / f"extension_error_{did}.txt").write_text(error, encoding="utf-8")
            reason = str(result.get("excluded_reason") or "")
            append_log(out / "run_log_ext.txt", f"{did}\t{'included' if not reason else 'excluded'}\t{reason or 'included'}")
            print(f"EXT {did} {'INCLUDED' if not reason else 'EXCLUDED'} {reason}", flush=True)
    results = pd.DataFrame(rows).reindex(columns=old.INTERNAL_COLUMNS).sort_values("candidate_priority")
    if len(results) != len(candidates):
        raise RuntimeError(f"extension incomplete {len(results)} != {len(candidates)}")
    included = results["excluded_reason"].fillna("").eq("")
    sizes = results.loc[included, "family"].value_counts()
    results["family_size"] = results["family"].map(sizes).fillna(0).astype(int)
    results.reindex(columns=old.CSV_COLUMNS).to_csv(out / "datasets_stratifiers_ext.csv", index=False)
    marginals: dict[str, dict[str, float]] = {}
    for did in results.loc[included, "openml_id"].astype(int):
        marginals[str(did)] = json.loads((marginal_dir / f"{did}.json").read_text(encoding="utf-8"))
    json_dump_atomic(out / "marginal_aucs_ext.json", marginals)
    append_log(out / "run_log_ext.txt", f"scan_complete candidates={len(results)} included={int(included.sum())}")


def encode_clean_data(
    X: pd.DataFrame, encoded_y: np.ndarray, categorical_map: dict[str, bool]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = np.bincount(encoded_y, minlength=2)
    positive_encoded_class = int(np.argmin(counts)) if counts[0] != counts[1] else 1
    y = (encoded_y == positive_encoded_class).astype(np.int8)
    fold = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (_, test) in enumerate(splitter.split(X, y)):
        fold[test] = k
    out = pd.DataFrame(index=np.arange(len(X)))
    schema_cols: dict[str, Any] = {}
    for name in X.columns:
        if categorical_map[name]:
            values = X[name].map(lambda v: None if pd.isna(v) else str(v))
            levels = sorted(v for v in values.dropna().unique().tolist())
            mapping = {level: idx for idx, level in enumerate(levels)}
            out[name] = values.map(mapping).astype(float)
            schema_cols[name] = {"type": "categorical", "encoding": mapping}
        else:
            out[name] = pd.to_numeric(X[name], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
            schema_cols[name] = {"type": "numeric"}
    out["y"] = y
    out["fold"] = fold
    schema = {
        "feature_order": list(X.columns),
        "columns": schema_cols,
        "label": {"positive_phase0b_encoded_class": positive_encoded_class, "counts_before_flip": counts.tolist()},
        "fold": {"class": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": SEED},
    }
    return out, schema


def fetch_dataset_clean(did: int, cache: Path, old: Any):
    openml.config.cache_directory = str(cache)
    dataset = openml.datasets.get_dataset(
        did, download_data=True, download_qualities=True, download_features_meta_data=True
    )
    X, encoded_y, info, categorical_map = old.clean_and_screen_data(dataset)
    return dataset, X, encoded_y, info, categorical_map


def metadata_value(dataset: Any, name: str, default: Any = None) -> Any:
    value = getattr(dataset, name, default)
    return scalar(value)


def run_clean_reproduce(root: Path, phase0b_dir: Path, old_script: Path) -> None:
    out = root / "phase0c_outputs"
    clean_dir = root / "cleaned_data"
    state = root / "run_state"
    cache = root / "openml_cache"
    for path in (out, clean_dir, state, cache):
        path.mkdir(parents=True, exist_ok=True)
    hashes = verify_phase0b(phase0b_dir)
    json_dump_atomic(state / "verified_phase0b_hashes.json", hashes)
    old = get_old(str(old_script))
    old_table = pd.read_csv(phase0b_dir / "datasets_stratifiers.csv", float_precision="round_trip")
    ext = pd.read_csv(out / "datasets_stratifiers_ext.csv", float_precision="round_trip")
    old_inc = old_table.loc[old_table.excluded_reason.fillna("").eq("")].copy()
    new_inc = ext.loc[ext.excluded_reason.fillna("").eq("")].copy()
    if len(old_inc) != 364:
        raise RuntimeError(f"STOP expected 364 old included datasets, got {len(old_inc)}")
    all_inc = pd.concat([old_inc.assign(pool="old"), new_inc.assign(pool="new")], ignore_index=True)
    old_marginal = json.loads((phase0b_dir / "marginal_aucs.json").read_text(encoding="utf-8"))
    new_marginal = json.loads((out / "marginal_aucs_ext.json").read_text(encoding="utf-8"))
    all_marginal = {**old_marginal, **new_marginal}
    sample_ids = set(np.random.default_rng(SEED).choice(old_inc.openml_id.astype(int).to_numpy(), size=5, replace=False).tolist())
    repro_path = state / "reproduction_checkpoint.csv"
    if repro_path.exists():
        repro_rows = pd.read_csv(repro_path).to_dict("records")
        repro_done = set(pd.to_numeric(pd.DataFrame(repro_rows).openml_id, errors="coerce").dropna().astype(int))
    else:
        repro_rows, repro_done = [], set()
    meta_path = state / "metadata_checkpoint.json"
    metadata_records: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for row in all_inc.itertuples(index=False):
        did = int(row.openml_id)
        parquet = clean_dir / f"{did}.parquet"
        schema_path = clean_dir / f"{did}.schema.json"
        needs_repro = row.pool == "old" and did not in repro_done
        needs_files = not parquet.exists() or not schema_path.exists()
        needs_meta = str(did) not in metadata_records
        if not (needs_repro or needs_files or needs_meta):
            continue
        started = time.perf_counter()
        try:
            dataset, X, encoded_y, info, categorical_map = fetch_dataset_clean(did, cache, old)
            expected_names = list(all_marginal[str(did)].keys())
            if set(X.columns) != set(expected_names):
                raise RuntimeError(f"feature_set_mismatch expected={expected_names} got={list(X.columns)}")
            # Phase 0b wrote every per-dataset marginal JSON with sort_keys=True.
            # The cleaned artifact therefore follows that frozen key order even
            # when OpenML's source-column order differs.
            X_clean_order = X.reindex(columns=expected_names)
            categorical_map_clean = {name: categorical_map[name] for name in expected_names}
            cleaned, schema = encode_clean_data(X_clean_order, encoded_y, categorical_map_clean)
            cleaned.to_parquet(parquet, index=False, compression="snappy")
            json_dump_atomic(schema_path, schema)
            tags = metadata_value(dataset, "tags", []) or []
            if isinstance(tags, str):
                tags = [tags]
            description = metadata_value(dataset, "description", "") or ""
            metadata_records[str(did)] = {
                "openml_id": did,
                "name": str(row.name),
                "version": int(row.version),
                "description": str(description),
                "tags": scalar(tags),
                "original_data_url": metadata_value(dataset, "original_data_url", ""),
                "citation": metadata_value(dataset, "citation", ""),
                "collection_date": metadata_value(dataset, "collection_date", ""),
                "creator": metadata_value(dataset, "creator", ""),
                "n_numeric": int(sum(not v for v in categorical_map_clean.values())),
                "n_categorical": int(sum(bool(v) for v in categorical_map_clean.values())),
                "missing_share": float(X_clean_order.isna().to_numpy().mean()),
                "n1": int(cleaned.y.sum()),
                "n0": int(len(cleaned) - cleaned.y.sum()),
                "n": int(len(cleaned)),
                "p_used": int(len(X.columns)),
                "pool": str(row.pool),
            }
            json_dump_atomic(meta_path, metadata_records)
            if needs_repro:
                metrics, marginal = old.calculate_metrics(X, encoded_y, categorical_map)
                lr_error = abs(float(metrics["auc_lr"]) - float(row.auc_lr))
                marginal_error = math.nan
                if did in sample_ids:
                    marginal_error = max(
                        abs(float(marginal[name]) - float(old_marginal[str(did)][name])) for name in expected_names
                    )
                repro_rows.append(
                    {"openml_id": did, "lr_abs_error": lr_error, "marginal_max_abs_error": marginal_error}
                )
                pd.DataFrame(repro_rows).sort_values("openml_id").to_csv(repro_path, index=False)
                repro_done.add(did)
            append_log(out / "run_log_ext.txt", f"clean\t{did}\tok\tseconds={time.perf_counter()-started:.3f}")
            print(f"CLEAN {did} OK", flush=True)
        except Exception as exc:
            append_log(out / "run_log_ext.txt", f"clean\t{did}\tfailed\t{type(exc).__name__}:{exc}")
            (state / f"clean_error_{did}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
    repro = pd.DataFrame(repro_rows).sort_values("openml_id")
    if len(repro) != 364 or not np.isfinite(pd.to_numeric(repro.lr_abs_error, errors="coerce")).all():
        raise RuntimeError("STOP reproduction_check does not contain 364 finite LR errors")
    n_bad = int((repro.lr_abs_error > 1e-6).sum())
    sampled = pd.to_numeric(repro.marginal_max_abs_error, errors="coerce").dropna()
    if len(sampled) != 5 or not (sampled < 1e-9).all():
        raise RuntimeError(f"STOP marginal reproduction failed: {sampled.tolist()}")
    repro.to_csv(out / "reproduction_check.csv", index=False)
    if n_bad > 5:
        raise RuntimeError(f"STOP {n_bad} old datasets exceed LR reproduction tolerance")
    build_metadata_and_aliases(root, metadata_records, old_inc, new_inc, old_script)


def normalize_alias_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    value = re.sub(r"(?:-dataset|-data|-v\d+|-version-?\d+)$", "", value)
    return value


def normalized_description(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()[:8000]


def build_metadata_and_aliases(
    root: Path, records: dict[str, Any], old_inc: pd.DataFrame, new_inc: pd.DataFrame, old_script: Path
) -> None:
    out = root / "phase0c_outputs"
    code_dir = root / "project" / "code"
    phase0c = load_module(code_dir / "phase0c_functions.py", "phase0c_functions_frozen")
    rows: list[dict[str, Any]] = []
    descriptions: dict[str, str] = {}
    for key in sorted(records, key=int):
        rec = records[key]
        tags = rec.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        description = rec.get("description") or ""
        terms = phase0c.medical_candidate(rec.get("name", ""), description, tags)
        rows.append(
            {
                "openml_id": int(key),
                "name": rec.get("name", ""),
                "version": rec.get("version"),
                "tags": ";".join(map(str, tags)),
                "original_data_url": rec.get("original_data_url") or "",
                "citation": rec.get("citation") or "",
                "collection_date": rec.get("collection_date") or "",
                "creator": ";".join(map(str, rec.get("creator"))) if isinstance(rec.get("creator"), list) else (rec.get("creator") or ""),
                "medical_terms": ";".join(terms),
                "n_numeric": rec["n_numeric"],
                "n_categorical": rec["n_categorical"],
                "missing_share": rec["missing_share"],
                "n1": rec["n1"],
                "n0": rec["n0"],
            }
        )
        descriptions[key] = description
    pd.DataFrame(rows).sort_values("openml_id").to_csv(out / "metadata.csv", index=False)
    json_dump_atomic(out / "metadata_descriptions.json", descriptions)
    old_ids, new_ids = set(old_inc.openml_id.astype(int)), set(new_inc.openml_id.astype(int))
    pairs: list[dict[str, Any]] = []
    recs = {int(k): v for k, v in records.items()}
    ids = sorted(recs)
    for pos, a in enumerate(ids):
        for b in ids[pos + 1 :]:
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
                desc_ratio = difflib.SequenceMatcher(None, da, db).ratio()
                if desc_ratio >= 0.92:
                    reasons.append(f"high_description_similarity:{desc_ratio:.3f}")
            if reasons:
                pairs.append(
                    {"openml_id_a": a, "name_a": ra["name"], "openml_id_b": b, "name_b": rb["name"], "reason": ";".join(reasons)}
                )
    pd.DataFrame(pairs, columns=["openml_id_a", "name_a", "openml_id_b", "name_b", "reason"]).to_csv(
        out / "alias_candidates.csv", index=False
    )


def transformed_groups(pipeline: Any, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    preprocess = pipeline.named_steps["preprocess"]
    groups: list[str] = []
    if numeric_cols:
        imp = preprocess.named_transformers_["numeric"].named_steps["imputer"]
        groups.extend([name for name, stat in zip(numeric_cols, imp.statistics_) if not pd.isna(stat)])
    if categorical_cols:
        catpipe = preprocess.named_transformers_["categorical"]
        imp = catpipe.named_steps["imputer"]
        retained = [name for name, stat in zip(categorical_cols, imp.statistics_) if not pd.isna(stat)]
        onehot = catpipe.named_steps["onehot"]
        for name, levels in zip(retained, onehot.categories_):
            groups.extend([name] * len(levels))
    return groups


def run_alt(root: Path, phase0b_dir: Path, old_script: Path) -> None:
    out = root / "phase0c_outputs"
    clean_dir = root / "cleaned_data"
    state = root / "run_state"
    verify_phase0b(phase0b_dir)
    old = get_old(str(old_script))
    phase0c = load_module(root / "project" / "code" / "phase0c_functions.py", "phase0c_functions_frozen_alt")
    old_table = pd.read_csv(phase0b_dir / "datasets_stratifiers.csv", float_precision="round_trip")
    ext = pd.read_csv(out / "datasets_stratifiers_ext.csv", float_precision="round_trip")
    inc = pd.concat(
        [old_table.loc[old_table.excluded_reason.fillna("").eq("")], ext.loc[ext.excluded_reason.fillna("").eq("")]],
        ignore_index=True,
    )
    marginals = {
        **json.loads((phase0b_dir / "marginal_aucs.json").read_text(encoding="utf-8")),
        **json.loads((out / "marginal_aucs_ext.json").read_text(encoding="utf-8")),
    }
    alt_checkpoint = state / "alt_checkpoint.csv"
    alt_rows = pd.read_csv(alt_checkpoint).to_dict("records") if alt_checkpoint.exists() else []
    done = {int(r["openml_id"]) for r in alt_rows}
    clusters_all = json.loads((state / "clusters_checkpoint.json").read_text(encoding="utf-8")) if (state / "clusters_checkpoint.json").exists() else {}
    coef_all = json.loads((state / "lr_coefficients_checkpoint.json").read_text(encoding="utf-8")) if (state / "lr_coefficients_checkpoint.json").exists() else {}
    for row in inc.itertuples(index=False):
        did = int(row.openml_id)
        if did in done:
            continue
        started = time.perf_counter()
        try:
            df = pd.read_parquet(clean_dir / f"{did}.parquet")
            schema = json.loads((clean_dir / f"{did}.schema.json").read_text(encoding="utf-8"))
            names = list(marginals[str(did)].keys())
            X = df[names].copy()
            y = df["y"].to_numpy(dtype=int)
            folds = df["fold"].to_numpy(dtype=int)
            categorical_map = {name: schema["columns"][name]["type"] == "categorical" for name in names}
            clusters = phase0c.feature_clusters(X.to_numpy(dtype=float), names)
            n1, n0 = int(y.sum()), int(len(y) - y.sum())
            neff_cluster = float(phase0c.neff_cluster(marginals[str(did)], clusters, n1, n0))
            X_prepared, numeric_cols, categorical_cols = old.prepare_feature_frame(X, categorical_map)
            per_fold: list[dict[str, np.ndarray]] = []
            audit_folds: list[dict[str, Any]] = []
            for fold in range(5):
                train = folds != fold
                held = folds == fold
                pipeline = old.build_lr_pipeline(numeric_cols, categorical_cols)
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    pipeline.fit(X_prepared.loc[train], y[train])
                preprocess = pipeline.named_steps["preprocess"]
                Xt = preprocess.transform(X_prepared.loc[held])
                if hasattr(Xt, "toarray"):
                    Xt = Xt.toarray()
                Xt = np.asarray(Xt, dtype=float)
                coef = np.asarray(pipeline.named_steps["model"].coef_, dtype=float).ravel()
                groups = transformed_groups(pipeline, numeric_cols, categorical_cols)
                if len(groups) != Xt.shape[1] or len(coef) != Xt.shape[1]:
                    raise RuntimeError(f"transformed group mismatch groups={len(groups)} Xt={Xt.shape} coef={len(coef)}")
                per_fold.append(phase0c.feature_contributions(coef, Xt, groups))
                audit_folds.append(
                    {"fold": fold, "coef": coef.tolist(), "intercept": np.asarray(pipeline.named_steps["model"].intercept_).ravel().tolist(), "groups": groups}
                )
            neff_cond = float(phase0c.neff_conditional(per_fold, names))
            alt_rows.append(
                {
                    "openml_id": did,
                    "neff_cluster": neff_cluster,
                    "neff_cond": neff_cond,
                    "n_clusters": len(set(clusters.values())),
                    "p_used": len(names),
                }
            )
            clusters_all[str(did)] = clusters
            coef_all[str(did)] = audit_folds
            pd.DataFrame(alt_rows).sort_values("openml_id").to_csv(alt_checkpoint, index=False)
            json_dump_atomic(state / "clusters_checkpoint.json", clusters_all)
            json_dump_atomic(state / "lr_coefficients_checkpoint.json", coef_all)
            append_log(out / "run_log_ext.txt", f"alt\t{did}\tok\tseconds={time.perf_counter()-started:.3f}")
            print(f"ALT {did} OK", flush=True)
        except Exception as exc:
            append_log(out / "run_log_ext.txt", f"alt\t{did}\tfailed\t{type(exc).__name__}:{exc}")
            (state / f"alt_error_{did}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
    alt = pd.DataFrame(alt_rows).sort_values("openml_id")
    if len(alt) != len(inc):
        raise RuntimeError(f"alt incomplete {len(alt)} != {len(inc)}")
    if not np.isfinite(alt[["neff_cluster", "neff_cond"]].to_numpy(dtype=float)).all() or not (alt[["neff_cluster", "neff_cond"]] > 0).all().all():
        raise RuntimeError("STOP non-finite or non-positive alternative stratifier")
    alt.to_csv(out / "alt_stratifiers.csv", index=False)
    json_dump_atomic(out / "clusters.json", clusters_all)
    json_dump_atomic(out / "lr_coefficients.json", coef_all)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["scan", "clean", "alt"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase0b-dir", type=Path, required=True)
    parser.add_argument("--old-script", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.root = args.root.resolve()
    verify_phase0b(args.phase0b_dir.resolve())
    if args.phase == "scan":
        run_scan(args.root, args.old_script.resolve(), args.workers)
    elif args.phase == "clean":
        run_clean_reproduce(args.root, args.phase0b_dir.resolve(), args.old_script.resolve())
    else:
        run_alt(args.root, args.phase0b_dir.resolve(), args.old_script.resolve())


if __name__ == "__main__":
    main()
