from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any

from . import config
from .checkpoints import checkpoint_path
from .execution import WorkUnit, execute_checkpoint
from .planner import build_plan
from .runtime import (
    assert_process_environment,
    assert_unit_in_plan,
    cpu_model_name,
    derive_runtime_identity,
    gpu_model_name,
    offline_socket_guard,
    registered_checkpoint_paths,
    resolve_registered_host,
    set_offline_environment,
    validate_hardware,
    verify_registered_inputs,
)


def _authorized(path: Path) -> None:
    if not path.is_file() or path.read_text(encoding="utf-8").strip() != "PHASE1_AUTHORIZED":
        raise RuntimeError(
            "Phase 1 pool fitting is locked until an authorization file containing "
            "PHASE1_AUTHORIZED is supplied"
        )


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
        "catboost", "interpret", "torch", "tabpfn", "tabicl",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _environment_record(
    root: Path,
    unit: WorkUnit,
    actual_host: str,
    cpu_name: str,
    gpu_name: str,
    registered_hashes: dict[str, str],
    driver_hash: str,
    weights: dict[str, Any],
) -> dict[str, Any]:
    determinism = config.gpu_determinism_record() if unit.group in {"gpu", "ftt"} else {
        "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        "seed": config.SEED,
    }
    lock_key = "phase0c_outputs/environment_phase1.lock.txt"
    return {
        "host": actual_host,
        "hostname": socket.gethostname(),
        "cpu_model": cpu_name,
        "gpu_model": gpu_name,
        "python": platform.python_version(),
        "packages": _package_versions(),
        "lock_sha256": registered_hashes[lock_key],
        "registered_hashes": registered_hashes,
        "driver_manifest_sha256": driver_hash,
        "weights": {name: item.sha256 for name, item in sorted(weights.items())},
        "weight_paths": {name: str(item.path) for name, item in sorted(weights.items())},
        "offline": {
            name: os.environ[name]
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        },
        "determinism": determinism,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--host-assignment", required=True, type=Path)
    parser.add_argument("--unit-json", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--authorization-file", required=True, type=Path)
    parser.add_argument("--host-registry", type=Path)
    args = parser.parse_args()

    assert_process_environment()
    set_offline_environment()
    if "TABPFN_TOKEN" in os.environ:
        raise RuntimeError("TABPFN_TOKEN must be unset during Phase 1 execution")
    _authorized(args.authorization_file.resolve())

    root = args.registered_root.resolve()
    assignment = args.host_assignment.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    runtime_root = args.runtime_root.resolve()
    unit = WorkUnit(**json.loads(args.unit_json))
    driver_dir = Path(__file__).resolve().parent
    role, actual_host = resolve_registered_host(
        (args.host_registry or (driver_dir / "host_registry.json")).resolve()
    )
    identity = derive_runtime_identity(root, driver_dir, assignment, unit.key)
    registered_hashes = verify_registered_inputs(root)
    plan = build_plan(root, assignment)
    assert_unit_in_plan(unit, plan)
    cpu_name = cpu_model_name()
    gpu_name = gpu_model_name() if unit.group in {"gpu", "ftt"} else ""
    validate_hardware(unit, role, actual_host, cpu_name, gpu_name)
    expected_checkpoint = checkpoint_path(checkpoint_root, unit).resolve()

    # Hash all local weights before any model constructor is reached. A failure
    # here exits without creating a checkpoint.
    weights = registered_checkpoint_paths(root) if unit.group in {"gpu", "ftt"} else {}
    record = _environment_record(
        root, unit, actual_host, cpu_name, gpu_name, registered_hashes,
        identity.driver_sha256, weights,
    )
    with offline_socket_guard():
        execute_checkpoint(
            root, unit, expected_checkpoint, identity, weights, record, []
        )


if __name__ == "__main__":
    main()
