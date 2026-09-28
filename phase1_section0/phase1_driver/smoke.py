from __future__ import annotations

from . import config

import argparse
import importlib.util
import json
import os
import socket
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from .cpu_models import fit_predict as cpu_fit_predict, xgb_d1_fit_predict
from .gpu_models import foundation_fit_predict, ftt_fit_predict
from .runtime import registered_checkpoint_paths, sha256_file

ALL_MODELS = config.ALL_MODELS


@dataclass(frozen=True)
class SmokeTask:
    variant: str
    model: str


@dataclass(frozen=True)
class SmokeRecord:
    variant: str
    model: str
    status: str
    train_shape: list[int]
    test_shape: list[int]
    categorical_positions: list[int]
    categorical_parameter: str
    nan_free_input: bool
    boundary_audit_count: int
    xgb_equivalence: str
    unseen_level_handling: str
    device: str
    determinism: dict[str, Any]
    output_shape: list[int]
    failure_reason: str
    network_guard: str
    driver_manifest_sha256: str
    loaded_parameter_sha256: str


def make_toy_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    values, labels = make_classification(n_samples=600, n_features=10, random_state=0)
    frame = pd.DataFrame(values, columns=[f"x{i}" for i in range(10)])
    rng = np.random.default_rng(config.SEED)
    frame["cat3"] = np.asarray([f"a{i}" for i in rng.integers(0, 3, size=len(frame))], dtype=object)
    frame["cat5"] = np.asarray([f"b{i}" for i in rng.integers(0, 5, size=len(frame))], dtype=object)
    for column in ("x0", "x1", "cat3"):
        rows = rng.choice(len(frame), size=30, replace=False)
        frame.loc[rows, column] = np.nan
    return frame, np.asarray(labels, dtype=int), ["cat3", "cat5"]


def _registered_root(explicit: Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("PHASE1_REGISTERED_ROOT"):
        candidates.append(Path(os.environ["PHASE1_REGISTERED_ROOT"]))
    candidates.extend([
        Path("/root/autodl-tmp/tabular_prior_phase1_preflight/tabular_prior_v2"),
        Path("/root/autodl-tmp/tabular_prior_phase0c/tabular_prior_v2"),
        Path("/root/autodl-tmp/tabular_prior_phase1_setup/tabular_prior_v2"),
    ])
    for candidate in candidates:
        if (candidate / "code" / "generators.py").is_file():
            return candidate.resolve()
    raise RuntimeError("final registered root containing code/generators.py was not found")


def _generator(root: Path) -> Any:
    path = root / "code" / "generators.py"
    spec = importlib.util.spec_from_file_location("phase1_frozen_generators", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen generator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_smoke_variant(features: pd.DataFrame, labels: np.ndarray, variant: str,
                       registered_root: Path | None = None) -> pd.DataFrame:
    if variant == "raw":
        return features.copy()
    if variant != "dispn_0.85_r0":
        raise ValueError(variant)
    frozen = _generator(_registered_root(registered_root))
    block, names = frozen.injection_block(
        labels, "phase1_smoke_toy", "dispersed_noise", 0.85, rep=0,
    )
    injection = pd.DataFrame(np.asarray(block), columns=list(names))
    if injection.shape != (len(features), 15):
        raise RuntimeError(f"frozen injection_block returned {injection.shape}, expected {(len(features), 15)}")
    return pd.concat([features.reset_index(drop=True), injection.reset_index(drop=True)], axis=1)


def smoke_tasks() -> list[SmokeTask]:
    return [SmokeTask(variant, model) for variant in ("raw", "dispn_0.85_r0") for model in ALL_MODELS]


def edge_case_split(features: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_rows, test_rows = next(iter(splitter.split(features, labels)))
    train = features.iloc[train_rows].copy()
    test = features.iloc[test_rows].copy()
    train["cat3"] = [f"a{position % 3}" for position in range(len(train))]
    test["cat3"] = [f"a{position % 2}" for position in range(len(test))]
    test.iloc[0, test.columns.get_loc("cat3")] = "unseen_outer_level"
    return train, labels[train_rows], test, labels[test_rows]


def audit_model_input(value: Any) -> None:
    if isinstance(value, pd.DataFrame):
        has_nan = value.isna().any().any()
    else:
        has_nan = np.isnan(np.asarray(value, dtype=float)).any()
    if has_nan:
        raise RuntimeError("NaN reached model boundary")


@contextmanager
def offline_network_guard() -> Iterator[None]:
    original_socket = socket.socket
    original_connection = socket.create_connection

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network disabled for offline smoke test")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_connection  # type: ignore[assignment]


def _record_failure(task: SmokeTask, train: pd.DataFrame, test: pd.DataFrame,
                    exc: Exception, driver_manifest_sha256: str) -> SmokeRecord:
    return SmokeRecord(task.variant, task.model, "failed", list(train.shape), list(test.shape), [], "", False, 0, "not_checked", "",
                       "cuda" if task.model in config.FOUNDATION_MODELS + ("ftt",) else "cpu",
                       {}, [], f"{type(exc).__name__}: {exc}", "enabled",
                       driver_manifest_sha256, "")


def xgb_registered_reference_equivalence(
    train: pd.DataFrame,
    labels: np.ndarray,
    test: pd.DataFrame,
    categorical: list[str],
    registered_root: Path,
    input_audit: Any,
) -> None:
    pilot = _generator_module(
        registered_root / "pipeline_code" / "phase0c" / "phase1_pilot.py",
        "phase1_pilot_xgb_equivalence",
    )
    registered, _fit_seconds, _predict_seconds = pilot.cpu_fit_predict(
        "xgb", train, labels, test, categorical
    )
    d1, unseen_cells = xgb_d1_fit_predict(
        pilot, train, labels, test, categorical, input_audit=input_audit
    )
    if unseen_cells != 0:
        raise RuntimeError(
            f"D1 no-unseen equivalence fixture unexpectedly has {unseen_cells} unseen cells"
        )
    if not np.array_equal(np.asarray(registered), d1):
        maximum = float(np.max(np.abs(np.asarray(registered) - d1)))
        raise RuntimeError(
            f"D1 XGBoost differs from registered path without unseen levels; max_abs_diff={maximum}"
        )


def _generator_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_smoke(registered_root: Path, log_path: Path) -> list[SmokeRecord]:
    features, labels, categorical = make_toy_data()
    records: list[SmokeRecord] = []
    checkpoints = registered_checkpoint_paths(registered_root)
    driver_manifest_sha256 = sha256_file(
        Path(__file__).resolve().parent.parent / "phase1_driver_sha256.txt"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    equivalence_train, equivalence_y, equivalence_test, _ = edge_case_split(features, labels)
    equivalence_test.loc[
        equivalence_test["cat3"].eq("unseen_outer_level"), "cat3"
    ] = "a0"
    xgb_registered_reference_equivalence(
        equivalence_train, equivalence_y, equivalence_test, categorical,
        registered_root, audit_model_input,
    )
    for task in smoke_tasks():
        variant = make_smoke_variant(features, labels, task.variant, registered_root)
        train, y_train, test, _y_test = edge_case_split(variant, labels)
        audited = 0
        def boundary_audit(value: Any) -> None:
            nonlocal audited
            audit_model_input(value)
            audited += 1
        try:
            if task.model in config.CPU_MODELS:
                with offline_network_guard():
                    output, detail = cpu_fit_predict(
                        task.model, train, y_train, test, categorical,
                        registered_root, input_audit=boundary_audit,
                    )
                xgb_equivalence = (
                    "no-unseen: bitwise identical to registered; unseen: D1 completed"
                    if task.model == "xgb" else "not_applicable"
                )
                device, determinism = "cpu", {
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                }
            elif task.model in config.FOUNDATION_MODELS:
                with offline_network_guard():
                    output, detail = foundation_fit_predict(
                        task.model, train, y_train, test, categorical, checkpoints,
                        input_audit=boundary_audit,
                    )
                xgb_equivalence = "not_applicable"
                device, determinism = "cuda", detail.pop("determinism")
            else:
                with offline_network_guard():
                    output, detail = ftt_fit_predict(
                        train, y_train, test, categorical, input_audit=boundary_audit
                    )
                xgb_equivalence = "not_applicable"
                device, determinism = "cuda", detail.pop("determinism")
            if audited < 2:
                raise RuntimeError(f"model boundary input audit ran only {audited} times")
            if output.shape != (len(test),):
                raise RuntimeError(f"unexpected model output shape {output.shape}")
            record = SmokeRecord(
                task.variant, task.model, "ok", list(train.shape), list(test.shape),
                list(detail["categorical_positions"]), str(detail["categorical_parameter"]),
                True, audited, xgb_equivalence,
                str(detail["unseen_category_handling"]),
                device, determinism, list(output.shape), "", "enabled",
                driver_manifest_sha256,
                str(detail.get("loaded_parameter_sha256", "")),
            )
        except Exception as exc:
            record = _record_failure(
                task, train, test, exc, driver_manifest_sha256
            )
        records.append(record)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
        print(f"smoke {len(records)}/20 {task.variant} {task.model} {record.status}", flush=True)
        if record.status != "ok":
            raise RuntimeError(record.failure_reason)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()
    if args.log.exists():
        args.log.unlink()
    run_smoke(args.registered_root.resolve(), args.log.resolve())


if __name__ == "__main__":
    main()
