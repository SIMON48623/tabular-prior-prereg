from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .execution import _generator, _load, _schema, prepare_phase0b_marginal_frame


_WORKER_ROOT: Path | None = None
_WORKER_PHASE0B: Any = None
_WORKER_GENERATOR: Any = None


def _initialize_marginal_worker(root_text: str) -> None:
    global _WORKER_ROOT, _WORKER_PHASE0B
    _WORKER_ROOT = Path(root_text)
    _WORKER_PHASE0B = _load(
        _WORKER_ROOT / "pipeline_code" / "phase0b" / "phase0b_experiment.py",
        "phase0b_full_marginal_check_worker",
    )


def _marginal_worker(openml_id: int) -> tuple[int, dict[str, float]]:
    if _WORKER_ROOT is None or _WORKER_PHASE0B is None:
        raise RuntimeError("marginal worker was not initialized")
    return openml_id, recompute_dataset_marginals(_WORKER_ROOT, openml_id, _WORKER_PHASE0B)


def _initialize_injection_worker(root_text: str) -> None:
    global _WORKER_ROOT, _WORKER_GENERATOR
    _WORKER_ROOT = Path(root_text)
    _WORKER_GENERATOR = _generator(_WORKER_ROOT)


def _injection_worker(openml_id: int) -> list[dict[str, Any]]:
    if _WORKER_ROOT is None or _WORKER_GENERATOR is None:
        raise RuntimeError("injection worker was not initialized")
    generator = _WORKER_GENERATOR
    _features, labels, _folds, _row_index, _categorical = _schema(_WORKER_ROOT, openml_id)
    rows: list[dict[str, Any]] = []
    for rep in range(int(generator.INJECT_REPS)):
        position = concentrated_signal_position(generator, openml_id, rep)
        for level in generator.INJECT_LEVELS:
            block, names = generator.injection_block(
                labels, openml_id, "concentrated", float(level), rep=rep
            )
            observed = float(roc_auc_score(labels, block[:, position]))
            difference = abs(observed - float(level))
            rows.append({
                "openml_id": openml_id, "rep": rep, "level": float(level),
                "signal_column": names[position], "observed_auroc": observed,
                "target_auroc": float(level), "abs_diff": difference,
                "status": "ok" if difference < 0.06 else "outside_0.06_recorded",
            })
    return rows


def target_region_ids(root: Path) -> list[int]:
    frame = pd.read_csv(root / "frozen" / "stratifier_v2.csv")
    required = {"openml_id", "target_region"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"stratifier_v2.csv missing columns: {sorted(required - set(frame.columns))}")
    flag = frame["target_region"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    identifiers = sorted(frame.loc[flag, "openml_id"].astype(int).tolist())
    if len(identifiers) != 238 or len(set(identifiers)) != 238:
        raise RuntimeError(f"target_region must contain exactly 238 unique IDs, got {len(identifiers)}")
    return identifiers


def recompute_dataset_marginals(root: Path, openml_id: int, phase0b: Any) -> dict[str, float]:
    features, labels, folds, categorical_map = prepare_phase0b_marginal_frame(
        root, openml_id, phase0b
    )
    unique_folds = sorted(np.unique(folds).tolist())
    if unique_folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"dataset {openml_id} fold values are {unique_folds}, expected 0..4")
    oof = np.full((len(features), len(features.columns)), np.nan, dtype=np.float64)
    for fold in unique_folds:
        train_position = np.flatnonzero(folds != fold)
        test_position = np.flatnonzero(folds == fold)
        for position, column in enumerate(features.columns):
            train_score, test_score = phase0b.single_feature_percentile_scores(
                features.iloc[train_position][column],
                features.iloc[test_position][column],
                categorical_map[column],
            )
            train_auc = float(roc_auc_score(labels[train_position], train_score))
            oof[test_position, position] = test_score if train_auc >= 0.5 else 1.0 - test_score
    if not np.isfinite(oof).all():
        raise RuntimeError(f"incomplete marginal predictions for dataset {openml_id}")
    return {
        column: float(roc_auc_score(labels, oof[:, position]))
        for position, column in enumerate(features.columns)
    }


def run_marginal_check(root: Path, output_path: Path, workers: int = 1) -> pd.DataFrame:
    expected = json.loads((root / "final_pool" / "marginal_aucs.json").read_text(encoding="utf-8"))
    pool = pd.read_csv(root / "final_pool" / "datasets_stratifiers.csv")
    pool = pool.loc[pool["excluded_reason"].fillna("").eq("")]
    identifiers = sorted(pool["openml_id"].astype(int).tolist())
    if len(identifiers) != 478 or set(map(str, identifiers)) != set(expected):
        raise RuntimeError("final pool and marginal_aucs.json do not contain the same 478 datasets")
    rows: list[dict[str, Any]] = []
    if workers == 1:
        phase0b = _load(
            root / "pipeline_code" / "phase0b" / "phase0b_experiment.py",
            "phase0b_full_marginal_check",
        )
        calculations = [
            (openml_id, recompute_dataset_marginals(root, openml_id, phase0b))
            for openml_id in identifiers
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_marginal_worker,
            initargs=(str(root),),
        ) as executor:
            calculations = list(executor.map(_marginal_worker, identifiers))
    for openml_id, actual in calculations:
        registered = {str(key): float(value) for key, value in expected[str(openml_id)].items()}
        if list(actual) != list(registered):
            raise RuntimeError(f"marginal feature order mismatch for dataset {openml_id}")
        for feature in actual:
            difference = abs(actual[feature] - registered[feature])
            rows.append({
                "openml_id": openml_id,
                "feature": feature,
                "registered_auroc": registered[feature],
                "recomputed_auroc": actual[feature],
                "abs_diff": difference,
                "status": "ok" if difference < 1e-12 else "failed",
            })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    failed = result.loc[result["abs_diff"] >= 1e-12]
    if not failed.empty:
        first = failed.iloc[0]
        raise RuntimeError(
            f"marginal reproduction failed for {int(first.openml_id)} {first.feature}: "
            f"abs_diff={first.abs_diff}"
        )
    return result


def concentrated_signal_position(generator: Any, openml_id: int, rep: int) -> int:
    return int(
        np.random.default_rng(generator.seed_for("inject-position", openml_id, rep)).integers(
            generator.INJECT_K
        )
    )


def run_injection_sanity(root: Path, output_path: Path, workers: int = 1) -> pd.DataFrame:
    generator = _generator(root)
    identifiers = target_region_ids(root)
    if workers == 1:
        _initialize_injection_worker(str(root))
        nested = [_injection_worker(openml_id) for openml_id in identifiers]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_injection_worker,
            initargs=(str(root),),
        ) as executor:
            nested = list(executor.map(_injection_worker, identifiers))
    rows = [row for dataset_rows in nested for row in dataset_rows]
    result = pd.DataFrame(rows)
    expected_rows = 238 * int(generator.INJECT_REPS) * len(generator.INJECT_LEVELS)
    if len(result) != expected_rows:
        raise RuntimeError(f"injection sanity produced {len(result)} rows, expected {expected_rows}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root = args.registered_root.resolve()
    output = args.output_dir.resolve()
    if args.workers < 1 or args.workers > 32:
        raise RuntimeError("model-free checks require 1..32 workers")
    run_marginal_check(root, output / "marginal_check.csv", args.workers)
    run_injection_sanity(root, output / "injection_sanity.csv", args.workers)


if __name__ == "__main__":
    main()
