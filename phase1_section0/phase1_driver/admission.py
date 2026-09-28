from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ["PYTHONHASHSEED"] = "13"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


CPU_MODELS = ("lr", "ebm", "catboost", "lgbm", "xgb", "rf")
SIZES = (100, 250, 500, 1000)
ADMISSION_COLUMNS = [
    "openml_id",
    "lr_auc",
    "abs_diff_vs_phase0c_reference",
    "tie_units",
    "cpu_model",
    "n_threads",
    "abs_diff_vs_preflight",
]

_PHASE0B: Any | None = None
_PHASE0B_PATH: str | None = None


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _phase0b(path: Path) -> Any:
    global _PHASE0B, _PHASE0B_PATH
    resolved = str(path.resolve())
    if _PHASE0B is None or _PHASE0B_PATH != resolved:
        _PHASE0B = _load_module(path, f"phase0b_admission_{os.getpid()}")
        _PHASE0B_PATH = resolved
    return _PHASE0B


def _cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def validate_admission(frame: pd.DataFrame) -> None:
    if len(frame) != 478 or frame["openml_id"].nunique() != 478:
        raise RuntimeError(
            f"214 admission failed: expected 478 unique datasets, got {len(frame)} rows"
        )
    bad = frame.loc[~frame["abs_diff_vs_preflight"].lt(1e-12)]
    if not bad.empty:
        identifiers = ",".join(str(int(value)) for value in bad["openml_id"].head(10))
        raise RuntimeError(
            f"214 admission failed: {len(bad)} differences are not < 1e-12; IDs={identifiers}"
        )
    if set(frame["n_threads"].astype(int)) != {1}:
        raise RuntimeError("214 admission failed: n_threads must be 1 for every row")


def _admission_dataset(task: tuple[str, int, float, float, str]) -> dict[str, Any]:
    root_text, openml_id, phase0c_auc, preflight_auc, checkpoint_text = task
    root = Path(root_text)
    checkpoint = Path(checkpoint_text)
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    phase0b = _phase0b(root / "pipeline_code" / "phase0b" / "phase0b_experiment.py")
    marginals = json.loads(
        (root / "final_pool" / "marginal_aucs.json").read_text(encoding="utf-8")
    )
    frame = pd.read_parquet(root / "cleaned_data" / f"{openml_id}.parquet")
    schema = json.loads(
        (root / "cleaned_data" / f"{openml_id}.schema.json").read_text(encoding="utf-8")
    )
    names = list(marginals[str(openml_id)].keys())
    features = frame[names].copy()
    labels = frame["y"].to_numpy(dtype=int)
    folds = frame["fold"].to_numpy(dtype=int)
    categorical = {
        name: schema["columns"][name]["type"] == "categorical" for name in names
    }
    prepared, numeric_columns, categorical_columns = phase0b.prepare_feature_frame(
        features, categorical
    )
    out_of_fold = np.full(len(labels), np.nan, dtype=float)
    for fold in range(5):
        train = folds != fold
        held_out = folds == fold
        pipeline = phase0b.build_lr_pipeline(numeric_columns, categorical_columns)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            pipeline.fit(prepared.loc[train], labels[train])
            out_of_fold[held_out] = pipeline.predict_proba(prepared.loc[held_out])[:, 1]
    if not np.isfinite(out_of_fold).all():
        raise RuntimeError(f"non-finite LR admission prediction for {openml_id}")
    auc = float(roc_auc_score(labels, out_of_fold))
    phase0c_difference = abs(auc - float(phase0c_auc))
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    tie_unit = 0.5 / (n_positive * n_negative)
    row = {
        "openml_id": int(openml_id),
        "lr_auc": auc,
        "abs_diff_vs_phase0c_reference": phase0c_difference,
        "tie_units": phase0c_difference / tie_unit,
        "cpu_model": _cpu_model_name(),
        "n_threads": 1,
        "abs_diff_vs_preflight": abs(auc - float(preflight_auc)),
    }
    _atomic_json(checkpoint, row)
    return row


def run_admission(root: Path, workers: int, output: Path, checkpoint_dir: Path) -> pd.DataFrame:
    pool = pd.read_csv(root / "final_pool" / "datasets_stratifiers.csv")
    pool = pool.loc[pool["excluded_reason"].fillna("").eq("")]
    phase0c = pd.read_csv(root / "phase0c_outputs" / "lr_reference_phase1.csv")
    preflight = pd.read_csv(
        root / "preflight_outputs" / "lr_reference_phase1_preflight.csv"
    )
    phase0c_map = dict(
        zip(phase0c["openml_id"].astype(int), phase0c["lr_auc"].astype(float), strict=True)
    )
    preflight_map = dict(
        zip(
            preflight["openml_id"].astype(int),
            preflight["lr_auc"].astype(float),
            strict=True,
        )
    )
    identifiers = sorted(pool["openml_id"].astype(int).tolist())
    if len(identifiers) != 478 or len(set(identifiers)) != 478:
        raise RuntimeError("214 admission failed: final pool is not 478 unique datasets")
    if set(identifiers) != set(phase0c_map) or set(identifiers) != set(preflight_map):
        raise RuntimeError("214 admission failed: reference IDs differ from final pool")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pending: list[tuple[str, int, float, float, str]] = []
    for openml_id in identifiers:
        checkpoint = checkpoint_dir / f"{openml_id}.json"
        if checkpoint.exists():
            rows.append(json.loads(checkpoint.read_text(encoding="utf-8")))
        else:
            pending.append(
                (
                    str(root),
                    openml_id,
                    phase0c_map[openml_id],
                    preflight_map[openml_id],
                    str(checkpoint),
                )
            )
    print(f"214 LR admission total=478 pending={len(pending)} workers={workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_admission_dataset, task): task[1] for task in pending}
            for number, future in enumerate(as_completed(future_map), 1):
                row = future.result()
                rows.append(row)
                print(
                    f"214 LR {number}/{len(pending)} openml_id={row['openml_id']} "
                    f"diff_vs_preflight={row['abs_diff_vs_preflight']:.17g}",
                    flush=True,
                )
                if not float(row["abs_diff_vs_preflight"]) < 1e-12:
                    for queued in future_map:
                        queued.cancel()
                    raise RuntimeError(
                        f"214 admission failed: openml_id={row['openml_id']} is not < 1e-12"
                    )
    result = pd.DataFrame(rows, columns=ADMISSION_COLUMNS).sort_values("openml_id")
    validate_admission(result)
    _atomic_csv(output, result)
    return result


def _fit_log_models(timing: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    data = timing.copy()
    data["total_seconds"] = data["fit_seconds"] + data["predict_seconds"]
    models: dict[tuple[str, str], np.ndarray] = {}
    for (model, stage), group in data.groupby(["model", "stage"]):
        predictors = np.column_stack(
            [
                np.ones(len(group)),
                np.log(np.maximum(group["n_train"].to_numpy(float), 2)),
                np.log(np.maximum(group["p_train"].to_numpy(float), 1)),
            ]
        )
        response = np.log(np.maximum(group["total_seconds"].to_numpy(float), 1e-6))
        models[(str(model), str(stage))] = np.linalg.lstsq(
            predictors, response, rcond=None
        )[0]
    return models


def _predict_cost(
    models: dict[tuple[str, str], np.ndarray],
    model: str,
    stage: str,
    n_train: float,
    p_train: float,
) -> float:
    row = np.array(
        [1.0, math.log(max(float(n_train), 2.0)), math.log(max(float(p_train), 1.0))]
    )
    return float(math.exp(float(row @ models[(model, stage)])))


def _standard_cpu_seconds(
    models: dict[tuple[str, str], np.ndarray],
    n_rows: float,
    n_features: float,
    variants: int = 1,
) -> float:
    return float(
        sum(
            variants
            * (
                5 * _predict_cost(models, model, "outer", 0.8 * n_rows, n_features)
                + 25
                * _predict_cost(models, model, "inner", 0.64 * n_rows, n_features)
            )
            for model in CPU_MODELS
        )
    )


def estimate_dataset_cpu_work(
    pool: pd.DataFrame,
    timing: pd.DataFrame,
    eligible_sizes: dict[int, list[int]],
) -> pd.DataFrame:
    models = _fit_log_models(timing)
    rows: list[dict[str, float | int]] = []
    for item in pool.itertuples(index=False):
        openml_id = int(item.openml_id)
        n_rows = float(item.n)
        n_features = float(item.p_used)
        seconds = _standard_cpu_seconds(models, n_rows, n_features)
        target_region = 0.60 <= float(item.auc_lr) <= 0.85
        if target_region:
            seconds += _standard_cpu_seconds(models, n_rows, n_features + 8, variants=27)
            seconds += _standard_cpu_seconds(models, n_rows, n_features + 15, variants=12)
            for size in eligible_sizes[openml_id]:
                seconds += sum(
                    5
                    * (
                        5 * _predict_cost(models, model, "outer", size, n_features)
                        + 25
                        * _predict_cost(
                            models, model, "inner", 0.8 * size, n_features
                        )
                    )
                    for model in CPU_MODELS
                )
        rows.append(
            {"openml_id": openml_id, "estimated_cpu_seconds": float(seconds)}
        )
    return pd.DataFrame(rows, columns=["openml_id", "estimated_cpu_seconds"]).sort_values(
        "openml_id"
    )


def build_host_assignment(costs: pd.DataFrame, hosts: list[str]) -> pd.DataFrame:
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError("hosts must be a non-empty unique list")
    required = {"openml_id", "estimated_cpu_seconds"}
    if not required.issubset(costs.columns):
        raise ValueError(f"costs missing columns {sorted(required - set(costs.columns))}")
    ordered_hosts = sorted(str(host) for host in hosts)
    loads = {host: 0.0 for host in ordered_hosts}
    assignments: list[dict[str, int | str]] = []
    ordered = costs.sort_values(
        ["estimated_cpu_seconds", "openml_id"], ascending=[False, True]
    )
    for row in ordered.itertuples(index=False):
        host = min(ordered_hosts, key=lambda candidate: (loads[candidate], candidate))
        assignments.append({"openml_id": int(row.openml_id), "host": host})
        loads[host] += float(row.estimated_cpu_seconds)
    return pd.DataFrame(assignments, columns=["openml_id", "host"]).sort_values(
        "openml_id"
    ).reset_index(drop=True)


def _eligible_sizes(root: Path, pool: pd.DataFrame) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for row in pool.itertuples(index=False):
        openml_id = int(row.openml_id)
        folds = pd.read_parquet(
            root / "cleaned_data" / f"{openml_id}.parquet", columns=["fold"]
        )["fold"]
        minimum_training = int(row.n) - int(folds.value_counts().max())
        result[openml_id] = [
            size
            for size in SIZES
            if size < minimum_training and float(row.minority_rate) * size >= 10
        ]
    return result


def generate_host_assignment(root: Path, output: Path) -> pd.DataFrame:
    pool = pd.read_csv(root / "final_pool" / "datasets_stratifiers.csv")
    pool = pool.loc[pool["excluded_reason"].fillna("").eq("")].copy()
    if len(pool) != 478 or pool["openml_id"].nunique() != 478:
        raise RuntimeError("host assignment requires 478 unique included datasets")
    timing = pd.read_csv(root / "phase0c_outputs" / "pilot_timing.csv")
    costs = estimate_dataset_cpu_work(pool, timing, _eligible_sizes(root, pool))
    assignment = build_host_assignment(costs, ["686", "214"])
    _atomic_csv(output, assignment)
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    admission = subparsers.add_parser("admission")
    admission.add_argument("--root", required=True, type=Path)
    admission.add_argument("--workers", type=int, default=32)
    admission.add_argument("--output", required=True, type=Path)
    admission.add_argument("--checkpoint-dir", required=True, type=Path)
    assignment = subparsers.add_parser("assignment")
    assignment.add_argument("--root", required=True, type=Path)
    assignment.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "admission":
        run_admission(args.root.resolve(), args.workers, args.output, args.checkpoint_dir)
    else:
        generate_host_assignment(args.root.resolve(), args.output)


if __name__ == "__main__":
    main()
