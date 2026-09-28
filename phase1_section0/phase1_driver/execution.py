from __future__ import annotations

from . import config

import importlib.util
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .checkpoints import CheckpointIdentity, read_checkpoint, rows_to_columnar, write_checkpoint
from .cpu_models import fit_predict as cpu_fit_predict
from .gpu_models import WeightAttestationFailure, foundation_fit_predict, ftt_fit_predict
from .runtime import RegisteredCheckpoint, registered_checkpoint_paths


class InfrastructureFailure(RuntimeError):
    def __init__(self, message: str, error_type: str = "cuda_context_fatal") -> None:
        super().__init__(message)
        self.error_type = error_type


def seed_cpu_process() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def validate_probability(value: np.ndarray, expected_rows: int) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float64)
    if (
        probability.shape != (expected_rows,)
        or not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise RuntimeError(
            f"invalid probability output: shape={probability.shape}, expected={(expected_rows,)}"
        )
    return probability


def classify_error(exc: BaseException) -> str:
    if isinstance(exc, WeightAttestationFailure):
        return "weight_attestation_fatal"
    text = f"{type(exc).__name__}: {exc}".lower()
    if "network disabled for offline phase 1 execution" in text:
        return "network_disabled"
    if "deterministic" in text and ("operator" in text or "implementation" in text):
        return "nondeterministic_operator"
    if isinstance(exc, MemoryError):
        return "memory_error"
    if any(marker in text for marker in (
        "cuda out of memory", "outofmemoryerror", "cublas_status_alloc_failed",
    )):
        return "out_of_memory"
    if any(marker in text for marker in (
        "cuda context", "context is destroyed", "device-side assert",
        "illegal memory access", "driver shutting down", "cuda initialization error",
        "cuda error: unknown error",
    )):
        return "cuda_context_fatal"
    return "model_error"


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _retryable_error(error_type: str) -> bool:
    return error_type in {
        "out_of_memory", "memory_error", "model_error",
        "network_disabled", "nondeterministic_operator",
    }


def run_with_retry(operation: Any, expected_rows: int, attempts: int = 2) -> tuple[np.ndarray | None, str, int, str]:
    failure = ""
    for attempt in range(attempts):
        try:
            probability = validate_probability(operation(), expected_rows)
            return probability, "", attempt + 1, ""
        except Exception as exc:
            error_type = classify_error(exc)
            if error_type in {"cuda_context_fatal", "weight_attestation_fatal"}:
                raise InfrastructureFailure(str(exc), error_type) from exc
            if error_type == "out_of_memory":
                _clear_cuda_cache()
            failure = f"attempt {attempt + 1} [{error_type}]: {type(exc).__name__}: {exc}"
            if not _retryable_error(error_type):
                break
    return None, failure, min(attempts, attempt + 1), error_type


def _run_model_with_retry(operation: Any, expected_rows: int,
                          attempts: int) -> tuple[np.ndarray | None, dict[str, Any], str, int, str]:
    failure = ""
    for attempt in range(attempts):
        try:
            raw, detail = operation()
            probability = validate_probability(raw, expected_rows)
            return probability, dict(detail), "", attempt + 1, ""
        except Exception as exc:
            error_type = classify_error(exc)
            if error_type in {"cuda_context_fatal", "weight_attestation_fatal"}:
                raise InfrastructureFailure(str(exc), error_type) from exc
            if error_type == "out_of_memory":
                _clear_cuda_cache()
            failure = f"attempt {attempt + 1} [{error_type}]: {type(exc).__name__}: {exc}"
            if not _retryable_error(error_type):
                break
    return None, {}, failure, min(attempts, attempt + 1), error_type


@dataclass(frozen=True)
class WorkUnit:
    module: str
    identifier: str
    variant: str
    fold: int
    group: str
    host: str

    @property
    def key(self) -> str:
        return "|".join((self.module, self.identifier, self.variant, str(self.fold), self.group))


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import registered module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _generator(root: Path) -> Any:
    return _load(root / "code" / "generators.py", "phase1_registered_generators")


def _schema(root: Path, openml_id: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_parquet(root / "cleaned_data" / f"{openml_id}.parquet")
    schema = json.loads((root / "cleaned_data" / f"{openml_id}.schema.json").read_text(encoding="utf-8"))
    if "row_index" in frame.columns:
        raise RuntimeError(
            f"cleaned_data/{openml_id}.parquet unexpectedly contains row_index; "
            "registered row_index is defined as parquet row position"
        )
    reserved = {"y", "fold"}
    columns = [column for column in frame.columns if column not in reserved]
    categorical = [column for column in columns if schema["columns"][column]["type"] == "categorical"]
    row_index = np.arange(len(frame), dtype=int)
    return frame[columns].copy(), frame["y"].to_numpy(dtype=int), frame["fold"].to_numpy(dtype=int), row_index, categorical


def prepare_phase0b_marginal_frame(
    root: Path, openml_id: int, phase0b: Any
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, bool]]:
    """Reconstruct the categorical values seen by frozen Phase 0b.

    Cleaned parquet files contain registered integer category codes. Phase 0b's
    percentile scorer sorts the original string levels, so the schema mapping
    must be inverted before calling its frozen preparation/scoring functions.
    """
    features, labels, folds, _row_index, categorical = _schema(root, openml_id)
    schema = json.loads(
        (root / "cleaned_data" / f"{openml_id}.schema.json").read_text(encoding="utf-8")
    )
    positive_encoded_class = int(schema["label"]["positive_phase0b_encoded_class"])
    if positive_encoded_class not in (0, 1):
        raise RuntimeError(
            f"invalid positive_phase0b_encoded_class for {openml_id}: {positive_encoded_class}"
        )
    if positive_encoded_class == 0:
        labels = 1 - labels
    registered_counts = [int(value) for value in schema["label"]["counts_before_flip"]]
    if np.bincount(labels, minlength=2).tolist() != registered_counts:
        raise RuntimeError(f"Phase 0b label reconstruction failed for {openml_id}")
    for column in categorical:
        mapping = schema["columns"][column].get("encoding")
        if not isinstance(mapping, dict):
            raise RuntimeError(f"categorical schema lacks encoding: {openml_id} {column}")
        inverse = {int(code): str(level) for level, code in mapping.items()}
        def decode(value: Any) -> Any:
            if pd.isna(value):
                return np.nan
            numeric = float(value)
            code = int(numeric)
            if numeric != code or code not in inverse:
                raise RuntimeError(
                    f"invalid categorical code for {openml_id} {column}: {value}"
                )
            return inverse[code]
        features[column] = features[column].map(decode).astype(object)
    categorical_map = {column: column in categorical for column in features.columns}
    prepared, _numeric_columns, _categorical_columns = phase0b.prepare_feature_frame(
        features, categorical_map
    )
    return prepared, labels, folds, categorical_map


def _m2_variant(root: Path, features: pd.DataFrame, labels: np.ndarray,
                openml_id: int, variant: str) -> pd.DataFrame:
    arm, level, rep = None, None, None
    parts = variant.split("_")
    if parts[0] == "noise":
        arm, rep = "noise", int(parts[1][1:])
    else:
        arm = {"conc": "concentrated", "disp": "dispersed", "dispn": "dispersed_noise"}[parts[0]]
        level, rep = float(parts[1]), int(parts[2][1:])
    block, names = _generator(root).injection_block(labels, openml_id, arm, level, rep=rep)
    return pd.concat([features.reset_index(drop=True), pd.DataFrame(block, columns=names)], axis=1)


def _probability_metrics(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return (
        float(roc_auc_score(labels, probability)),
        float(log_loss(labels, clipped, labels=[0, 1])),
        float(brier_score_loss(labels, probability)),
    )


def _cpu_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _gpu_name() -> str:
    import torch
    return str(torch.cuda.get_device_name(0))


def _selection_rows(identifier: str, variant: str, fold: int,
                    scores: dict[str, list[float] | None], models: tuple[str, ...],
                    select: bool) -> list[dict[str, Any]]:
    eligible = {
        model: float(np.mean(value))
        for model, value in scores.items()
        if value is not None and len(value) == 5
    }
    chosen = max(eligible, key=lambda model: (eligible[model], -models.index(model))) if eligible and select else None
    rows = []
    for model in models:
        values = scores.get(model)
        complete = values is not None and len(values) == 5
        row: dict[str, Any] = {"id": identifier, "variant": variant, "fold": fold, "model": model}
        for number in range(5):
            row[f"inner_auroc_{number + 1}"] = float(values[number]) if complete else None
        row["inner_auroc_mean"] = float(np.mean(values)) if complete else None
        row["inner_status"] = "ok" if complete else "failed"
        row["selected"] = model == chosen
        rows.append(row)
    return rows


def _metric_row(identifier: str, variant: str, fold: int, model: str,
                labels: np.ndarray, probability: np.ndarray | None, n_train: int,
                elapsed: float, host: str, failure: str = "") -> dict[str, Any]:
    if probability is None:
        auc = loss = brier = None
        status = "failed"
    else:
        auc, loss, brier = _probability_metrics(labels, probability)
        status = "ok"
    is_cpu = model in config.CPU_MODELS
    return {
        "id": identifier, "variant": variant, "fold": fold, "model": model,
        "auroc": auc, "logloss": loss, "brier": brier, "n_test": len(labels),
        "n_pos_test": int(labels.sum()), "n_train": n_train, "fit_seconds": elapsed,
        "host": host, "cpu_model": _cpu_name() if is_cpu else "",
        "gpu_model": "" if is_cpu else _gpu_name(), "n_threads": 1 if is_cpu else None,
        "status": status, "failure_reason": failure,
    }


def _prediction_rows(openml_id: int, variant: str, fold: int, row_index: np.ndarray,
                     labels: np.ndarray, model: str, probability: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"openml_id": openml_id, "variant": variant, "fold": fold,
         "row_index": int(index), "y": int(label), "model": model, "p": float(value)}
        for index, label, value in zip(row_index, labels, probability, strict=True)
    ]


def _inner_scores_cpu(root: Path, train: pd.DataFrame, labels: np.ndarray,
                      categorical: list[str]) -> tuple[dict[str, list[float] | None], dict[str, str]]:
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    result: dict[str, list[float] | None] = {}
    failures: dict[str, str] = {}
    for model in config.CPU_MODELS:
        scores: list[float] = []
        try:
            for inner_train, inner_test in split.split(train, labels):
                probability, _detail, failure, _attempt_count, _error_type = _run_model_with_retry(
                    lambda: cpu_fit_predict(
                        model, train.iloc[inner_train], labels[inner_train], train.iloc[inner_test],
                        categorical, root,
                    ),
                    len(inner_test),
                    2,
                )
                if probability is None:
                    raise RuntimeError(failure)
                scores.append(float(roc_auc_score(labels[inner_test], probability)))
            result[model] = scores
        except Exception as exc:
            result[model] = None
            failures[model] = f"{type(exc).__name__}: {exc}"
    return result, failures


def _inner_scores_foundation(root: Path, train: pd.DataFrame, labels: np.ndarray,
                             categorical: list[str],
                             checkpoints: dict[str, RegisteredCheckpoint]) -> tuple[dict[str, list[float] | None], dict[str, str]]:
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    result: dict[str, list[float] | None] = {}
    failures: dict[str, str] = {}
    for model in config.FOUNDATION_MODELS:
        scores: list[float] = []
        try:
            for inner_train, inner_test in split.split(train, labels):
                probability, _detail, failure, _attempt_count, _error_type = _run_model_with_retry(
                    lambda: foundation_fit_predict(
                        model, train.iloc[inner_train], labels[inner_train],
                        train.iloc[inner_test], categorical, checkpoints,
                    ),
                    len(inner_test),
                    2,
                )
                if probability is None:
                    raise RuntimeError(failure)
                scores.append(float(roc_auc_score(labels[inner_test], probability)))
            result[model] = scores
        except InfrastructureFailure:
            raise
        except Exception as exc:
            result[model] = None
            failures[model] = f"{type(exc).__name__}: {exc}"
    return result, failures


def _run_log(unit: WorkUnit, elapsed: float, metrics: list[dict[str, Any]],
             selection: list[dict[str, Any]], inner_failures: dict[str, str],
             attempt_counts: dict[str, int] | None = None,
             error_types: dict[str, str] | None = None,
             fatal_attempts: list[dict[str, Any]] | None = None,
             actual_host: str | None = None) -> dict[str, Any]:
    failures = {
        str(row["model"]): str(row["failure_reason"])
        for row in metrics
        if row.get("status") == "failed"
    }
    s_undefined = bool(
        unit.group == "cpu"
        and unit.module in {"M1", "M2", "M4", "M5"}
        and selection
        and not any(bool(row.get("selected")) for row in selection)
    )
    return {
        "task_key": unit.key,
        "module": unit.module,
        "id": unit.identifier,
        "variant": unit.variant,
        "fold": unit.fold,
        "group": unit.group,
        "host": actual_host or unit.host,
        "elapsed_seconds": float(elapsed),
        "status": "ok" if not failures else "completed_with_failures",
        "failure_reasons": failures,
        "s_undefined": s_undefined,
        "inner_failures": dict(sorted(inner_failures.items())),
        "attempt_counts": dict(sorted((attempt_counts or {}).items())),
        "error_types": dict(sorted((error_types or {}).items())),
        "fatal_attempts": list(fatal_attempts or []),
    }


def _pool_views(root: Path, unit: WorkUnit) -> tuple[int, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    openml_id = int(unit.identifier)
    features, labels, folds, row_index, categorical = _schema(root, openml_id)
    if unit.module == "M2":
        features = _m2_variant(root, features, labels, openml_id, unit.variant)
    outer_train = np.flatnonzero(folds != unit.fold)
    outer_test = np.flatnonzero(folds == unit.fold)
    if unit.module == "M4":
        size_text, rep_text = unit.variant.split("_")
        size, rep = int(size_text[1:]), int(rep_text[1:])
        relative = _generator(root).subsample_indices(labels[outer_train], openml_id, unit.fold, size, rep)
        outer_train = outer_train[np.asarray(relative, dtype=int)]
    return (
        openml_id, features.iloc[outer_train].copy(), labels[outer_train],
        row_index[outer_train], features.iloc[outer_test].copy(), labels[outer_test],
        row_index[outer_test], categorical,
    )


def run_pool_unit(
    root: Path,
    unit: WorkUnit,
    registered_checkpoints: dict[str, RegisteredCheckpoint] | None = None,
    actual_host: str | None = None,
) -> dict[str, Any]:
    unit_started = time.perf_counter()
    seed_cpu_process()
    openml_id, train, y_train, _train_index, test, y_test, test_index, categorical = _pool_views(root, unit)
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    ftt_log: list[dict[str, Any]] = []
    inner_failures: dict[str, str] = {}
    attempt_counts: dict[str, int] = {}
    error_types: dict[str, str] = {}
    checkpoints = (
        registered_checkpoints if registered_checkpoints is not None
        else (registered_checkpoint_paths(root) if unit.group == "gpu" else {})
    )
    if unit.group == "cpu":
        scores, inner_failures = _inner_scores_cpu(root, train, y_train, categorical)
        selection = _selection_rows(unit.identifier, unit.variant, unit.fold, scores, config.CPU_MODELS, True)
        models = config.CPU_MODELS
    elif unit.group == "gpu":
        if unit.module == "M1":
            scores, inner_failures = _inner_scores_foundation(
                root, train, y_train, categorical, checkpoints
            )
        else:
            scores, inner_failures = {}, {}
        selection = _selection_rows(unit.identifier, unit.variant, unit.fold, scores, config.FOUNDATION_MODELS, False) if unit.module == "M1" else []
        models = config.FOUNDATION_MODELS
    elif unit.group == "ftt":
        scores, inner_failures = {}, {}
        selection = []
        models = ("ftt",)
    else:
        raise ValueError(unit.group)
    for model in models:
        started = time.perf_counter()
        attempts = 2
        def operation() -> tuple[np.ndarray, dict[str, Any]]:
            if model in config.CPU_MODELS:
                return cpu_fit_predict(model, train, y_train, test, categorical, root)
            if model in config.FOUNDATION_MODELS:
                return foundation_fit_predict(
                    model, train, y_train, test, categorical, checkpoints
                )
            return ftt_fit_predict(train, y_train, test, categorical)
        probability, detail, failure, attempt_count, error_type = _run_model_with_retry(
            operation, len(test), attempts
        )
        attempt_counts[model] = attempt_count
        if error_type:
            error_types[model] = error_type
        elapsed = time.perf_counter() - started
        metrics.append(_metric_row(unit.identifier, unit.variant, unit.fold, model, y_test, probability,
                                   len(y_train), elapsed, actual_host or unit.host, failure))
        if probability is not None:
            predictions.extend(_prediction_rows(openml_id, unit.variant, unit.fold, test_index, y_test, model, probability))
            if model == "ftt":
                ftt_log.append({"openml_id": openml_id, "fold": unit.fold,
                                "best_epoch": int(detail["best_epoch"]), "n_params": int(detail["n_params"])})
    log = _run_log(
        unit, time.perf_counter() - unit_started, metrics, selection,
        inner_failures, attempt_counts, error_types, actual_host=actual_host,
    )
    return {"unit": asdict(unit), "metrics": metrics, "predictions": rows_to_columnar(predictions),
            "selection": selection, "ftt_log": ftt_log, "log": log}


def run_m5_unit(
    root: Path,
    unit: WorkUnit,
    registered_checkpoints: dict[str, RegisteredCheckpoint] | None = None,
    actual_host: str | None = None,
) -> dict[str, Any]:
    unit_started = time.perf_counter()
    generator = _generator(root)
    gen, k_text, rho_text, auc_text = unit.identifier.split("|")
    k, rho, auc = int(k_text), float(rho_text), float(auc_text)
    size_text, rep_text = unit.variant.split("_")
    size, rep = int(size_text[1:]), int(rep_text[1:])
    train_array, y_train, _truth, _informative = generator.synth_draw(gen, k, rho, auc, size, rep, "train")
    test_array, y_test, _truth_test, _informative_test = generator.synth_draw(gen, k, rho, auc, generator.SYN_N_TEST, rep, "test")
    train = pd.DataFrame(train_array, columns=[f"x{i}" for i in range(train_array.shape[1])])
    test = pd.DataFrame(test_array, columns=train.columns)
    metrics: list[dict[str, Any]] = []
    checkpoints = (
        registered_checkpoints if registered_checkpoints is not None
        else (registered_checkpoint_paths(root) if unit.group == "gpu" else {})
    )
    inner_failures: dict[str, str] = {}
    attempt_counts: dict[str, int] = {}
    error_types: dict[str, str] = {}
    if unit.group == "cpu":
        scores, inner_failures = _inner_scores_cpu(root, train, y_train, [])
        selection = _selection_rows(unit.identifier, unit.variant, 0, scores, config.CPU_MODELS, True)
        models = config.CPU_MODELS
    else:
        selection = []
        models = config.FOUNDATION_MODELS
    for model in models:
        started = time.perf_counter()
        attempts = 2
        def operation() -> tuple[np.ndarray, dict[str, Any]]:
            if unit.group == "cpu":
                return cpu_fit_predict(model, train, y_train, test, [], root)
            return foundation_fit_predict(model, train, y_train, test, [], checkpoints)
        probability, _detail, failure, attempt_count, error_type = _run_model_with_retry(
            operation, len(test), attempts
        )
        attempt_counts[model] = attempt_count
        if error_type:
            error_types[model] = error_type
        metrics.append(_metric_row(unit.identifier, unit.variant, 0, model, y_test, probability,
                                   len(y_train), time.perf_counter() - started, actual_host or unit.host, failure))
    log = _run_log(
        unit, time.perf_counter() - unit_started, metrics, selection,
        inner_failures, attempt_counts, error_types, actual_host=actual_host,
    )
    return {"unit": asdict(unit), "metrics": metrics, "predictions": {},
            "selection": selection, "ftt_log": [], "log": log}


def run_m6(root: Path, openml_id: int) -> dict[str, dict[str, float]]:
    phase0b = _load(root / "pipeline_code" / "phase0b" / "phase0b_experiment.py", "phase0b_m6_frozen")
    features, labels, folds, categorical_map = prepare_phase0b_marginal_frame(
        root, openml_id, phase0b
    )
    result: dict[str, dict[str, float]] = {}
    for fold in range(5):
        train_position = np.flatnonzero(folds != fold)
        train = features.iloc[train_position].reset_index(drop=True)
        y_train = labels[train_position]
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        out_of_fold = np.full((len(train), train.shape[1]), np.nan, dtype=float)
        for inner_train, inner_test in inner.split(train, y_train):
            for position, column in enumerate(train.columns):
                train_score, test_score = phase0b.single_feature_percentile_scores(
                    train.iloc[inner_train][column], train.iloc[inner_test][column], categorical_map[column]
                )
                train_auc = float(roc_auc_score(y_train[inner_train], train_score))
                out_of_fold[inner_test, position] = test_score if train_auc >= 0.5 else 1.0 - test_score
        if np.isnan(out_of_fold).any():
            raise RuntimeError(f"incomplete M6 marginal predictions for {openml_id} fold {fold}")
        result[str(fold)] = {
            column: float(roc_auc_score(y_train, out_of_fold[:, position]))
            for position, column in enumerate(train.columns)
        }
    return result


def execute_checkpoint(root: Path, unit: WorkUnit, checkpoint: Path,
                       identity: CheckpointIdentity,
                       registered_checkpoints: dict[str, RegisteredCheckpoint] | None = None,
                       environment_record: dict[str, Any] | None = None,
                       fatal_attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if identity.task_key != unit.key:
        raise RuntimeError("checkpoint identity task key does not match work unit")
    if checkpoint.exists():
        return read_checkpoint(checkpoint, identity)
    actual_host = str(environment_record.get("host")) if environment_record else None
    if unit.module in {"M1", "M2", "M4"}:
        payload = run_pool_unit(root, unit, registered_checkpoints, actual_host)
    elif unit.module == "M5":
        payload = run_m5_unit(root, unit, registered_checkpoints, actual_host)
    elif unit.module == "M6":
        started = time.perf_counter()
        marginal = run_m6(root, int(unit.identifier))
        payload = {"unit": asdict(unit), "metrics": [], "predictions": {}, "selection": [],
                   "ftt_log": [], "marginal": marginal,
                   "log": _run_log(
                       unit, time.perf_counter() - started, [], [], {},
                       actual_host=actual_host,
                   )}
    else:
        raise ValueError(f"unsupported checkpoint module {unit.module}")
    payload["environment"] = dict(environment_record or {})
    if fatal_attempts:
        payload["log"]["fatal_attempts"] = list(fatal_attempts)
    write_checkpoint(checkpoint, identity, payload)
    return payload


def failed_unit_payload(
    root: Path,
    unit: WorkUnit,
    reason: str,
    error_type: str,
    environment_record: dict[str, Any],
    fatal_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a terminal, schema-complete checkpoint without fitting a model."""
    if unit.group == "cpu":
        models = config.CPU_MODELS
    elif unit.group == "gpu":
        models = config.FOUNDATION_MODELS
    elif unit.group == "ftt":
        models = ("ftt",)
    else:
        models = ()

    if unit.module in {"M1", "M2", "M4"}:
        _openml_id, _train, y_train, _train_index, _test, y_test, _test_index, _categorical = _pool_views(root, unit)
        n_train = len(y_train)
    elif unit.module == "M5":
        generator = _generator(root)
        gen, k_text, rho_text, auc_text = unit.identifier.split("|")
        size_text, rep_text = unit.variant.split("_")
        size, rep = int(size_text[1:]), int(rep_text[1:])
        _train, y_train, _truth, _informative = generator.synth_draw(
            gen, int(k_text), float(rho_text), float(auc_text), size, rep, "train"
        )
        _test, y_test, _truth_test, _informative_test = generator.synth_draw(
            gen, int(k_text), float(rho_text), float(auc_text),
            generator.SYN_N_TEST, rep, "test",
        )
        n_train = len(y_train)
    else:
        y_test = np.asarray([], dtype=int)
        n_train = 0

    host = str(environment_record.get("host", unit.host))
    cpu_name = str(environment_record.get("cpu_model", ""))
    gpu_name = str(environment_record.get("gpu_model", ""))
    metrics = []
    for model in models:
        is_cpu = model in config.CPU_MODELS
        metrics.append({
            "id": unit.identifier, "variant": unit.variant, "fold": unit.fold,
            "model": model, "auroc": None, "logloss": None, "brier": None,
            "n_test": int(len(y_test)), "n_pos_test": int(np.asarray(y_test).sum()),
            "n_train": int(n_train), "fit_seconds": 0.0, "host": host,
            "cpu_model": cpu_name if is_cpu else "",
            "gpu_model": "" if is_cpu else gpu_name,
            "n_threads": 1 if is_cpu else None,
            "status": "failed", "failure_reason": reason,
        })
    selection: list[dict[str, Any]] = []
    if unit.module == "M1" and unit.group in {"cpu", "gpu"}:
        selection = _selection_rows(
            unit.identifier, unit.variant, unit.fold,
            {model: None for model in models}, tuple(models), False,
        )
    log = {
        "task_key": unit.key, "module": unit.module, "id": unit.identifier,
        "variant": unit.variant, "fold": unit.fold, "group": unit.group,
        "host": host, "elapsed_seconds": 0.0, "status": "failed",
        "failure_reasons": {model: reason for model in models} or {"unit": reason},
        "s_undefined": bool(unit.group == "cpu" and selection),
        "inner_failures": {model: reason for model in models} if selection else {},
        "attempt_counts": {model: 0 for model in models},
        "error_types": {model: error_type for model in models} or {"unit": error_type},
        "fatal_attempts": list(fatal_attempts),
    }
    return {
        "unit": asdict(unit), "metrics": metrics, "predictions": {},
        "selection": selection, "ftt_log": [], "environment": dict(environment_record),
        "unit_failed": True, "log": log,
    }
