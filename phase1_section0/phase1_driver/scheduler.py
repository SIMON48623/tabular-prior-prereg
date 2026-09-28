from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from . import config
from .checkpoints import (
    CheckpointIdentity, checkpoint_path, exclusive_lock, load_migrations,
    quarantine_checkpoint, read_checkpoint_with_migration, write_checkpoint,
)
from .execution import (
    InfrastructureFailure, WorkUnit, execute_checkpoint, failed_unit_payload,
)
from .planner import build_plan, execution_stage
from .runtime import (
    assert_process_environment, cpu_model_name, ensure_cleaned_data_manifest,
    gpu_model_name, offline_socket_guard, registered_checkpoint_paths,
    resolve_registered_host, set_offline_environment, sha256_file,
    validate_hardware, verify_driver_manifest, verify_registered_inputs,
)
from .timeouts import unit_timeout_seconds
from .worker_supervisor import SupervisedTask, run_supervised_tasks


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _authorized(path: Path) -> None:
    if not path.is_file() or path.read_text(encoding="utf-8").strip() != "PHASE1_AUTHORIZED":
        raise RuntimeError("Phase 1 execution is locked: invalid authorization file")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fatal_path(runtime_root: Path, unit: WorkUnit) -> Path:
    return runtime_root / "fatal_attempts" / f"{unit.key.replace('|', '__')}.json"


def _load_fatal_attempts(runtime_root: Path, unit: WorkUnit) -> list[dict[str, Any]]:
    path = _fatal_path(runtime_root, unit)
    if not path.exists():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise RuntimeError(f"invalid fatal attempt ledger: {path}")
    return document


def _save_fatal_attempts(runtime_root: Path, unit: WorkUnit,
                         attempts: list[dict[str, Any]]) -> None:
    path = _fatal_path(runtime_root, unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(attempts, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _failed_after_fatal(unit: WorkUnit, attempts: list[dict[str, Any]],
                        environment: dict[str, Any], root: Path) -> dict[str, Any]:
    reason = str(attempts[-1].get("reason", "two consecutive process-fatal attempts"))
    error_type = str(attempts[-1].get("error_type", "process_fatal"))
    return failed_unit_payload(root, unit, reason, error_type, environment, attempts)


def _worker_execute(message: dict[str, Any]) -> dict[str, Any]:
    unit = WorkUnit(**message["unit"])
    checkpoint = Path(message["checkpoint"])
    identity = CheckpointIdentity(**message["identity"])
    lock = checkpoint.with_suffix(checkpoint.suffix + ".lock")
    try:
        with exclusive_lock(lock), offline_socket_guard():
            payload = execute_checkpoint(
                Path(message["root"]), unit, checkpoint, identity,
                message["registered_checkpoints"], message["environment"],
                message["fatal_attempts"],
            )
        return {"status": payload["log"]["status"], "log": payload["log"]}
    except InfrastructureFailure as exc:
        return {
            "status": "fatal", "error_type": exc.error_type,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _identity_factory(root: Path, driver_dir: Path, assignment: Path):
    registration = sha256_file(root / "MANIFEST.sha256")
    driver = sha256_file(driver_dir.parent / "phase1_driver_sha256.txt")
    assignment_hash = sha256_file(assignment)

    def make(task_key: str) -> CheckpointIdentity:
        return CheckpointIdentity(registration, driver, assignment_hash, task_key)
    return make


def _checkpoint_state(path: Path, identity: CheckpointIdentity,
                      migrations: list[dict[str, Any]], quarantine: Path) -> tuple[bool, dict[str, Any] | None]:
    if not path.exists():
        return False, None
    try:
        _payload, migration = read_checkpoint_with_migration(path, identity, migrations)
        return True, migration
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        quarantine_checkpoint(path, quarantine, f"{type(exc).__name__}: {exc}")
        return False, None


def _environment_record(
    actual_host: str, role: str, cpu_name: str, gpu_name: str,
    registered_hashes: dict[str, str], cleaned_hashes: dict[str, str],
    driver_hash: str, weights: dict[str, Any],
) -> dict[str, Any]:
    return {
        "host": actual_host, "registered_role": role,
        "cpu_model": cpu_name, "gpu_model": gpu_name,
        "registered_manifest_entries": len(registered_hashes),
        "cleaned_data_entries": len(cleaned_hashes),
        "driver_manifest_sha256": driver_hash,
        "weights": {name: item.sha256 for name, item in sorted(weights.items())},
        "weight_paths": {name: str(item.path) for name, item in sorted(weights.items())},
        "offline": {name: os.environ[name] for name in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
        )},
        "determinism": config.gpu_determinism_record() if role == "4090" else {
            "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
            "seed": config.SEED,
        },
    }


def _supervised_task(
    root: Path, unit: WorkUnit, checkpoint: Path, identity: CheckpointIdentity,
    weights: dict[str, Any], environment: dict[str, Any],
    fatal_attempts: list[dict[str, Any]],
) -> SupervisedTask:
    return SupervisedTask(
        key=unit.key,
        payload={
            "root": str(root), "unit": asdict(unit), "checkpoint": str(checkpoint),
            "identity": asdict(identity), "registered_checkpoints": weights,
            "environment": environment, "fatal_attempts": fatal_attempts,
        },
        timeout_seconds=unit_timeout_seconds(root, unit),
    )


def _run_pending_units(
    pending: list[WorkUnit], root: Path, checkpoint_root: Path,
    identity_for: Any, weights: dict[str, Any], environment: dict[str, Any],
    runtime_root: Path, workers: int,
) -> None:
    failures_path = runtime_root / "scheduler_failures.jsonl"
    initial = {unit.key: _load_fatal_attempts(runtime_root, unit) for unit in pending}
    runnable: list[WorkUnit] = []
    for unit in pending:
        if len(initial[unit.key]) >= 2:
            write_checkpoint(
                checkpoint_path(checkpoint_root, unit), identity_for(unit.key),
                _failed_after_fatal(unit, initial[unit.key], environment, root),
            )
        else:
            runnable.append(unit)
    tasks = [
        _supervised_task(
            root, unit, checkpoint_path(checkpoint_root, unit), identity_for(unit.key),
            weights, environment, initial[unit.key],
        )
        for unit in runnable
    ]
    by_key = {unit.key: unit for unit in runnable}

    def on_fatal(task: SupervisedTask, event: dict[str, Any]) -> None:
        unit = by_key[task.key]
        attempts = _load_fatal_attempts(runtime_root, unit)
        attempts.append(event)
        _save_fatal_attempts(runtime_root, unit, attempts)
        _append_jsonl(failures_path, event)

    def on_result(task: SupervisedTask, result: dict[str, Any]) -> None:
        if result.get("status") != "failed" or not result.get("fatal_attempts"):
            return
        unit = by_key[task.key]
        path = checkpoint_path(checkpoint_root, unit)
        if not path.exists():
            write_checkpoint(
                path, identity_for(unit.key),
                _failed_after_fatal(unit, list(result["fatal_attempts"]), environment, root),
            )

    run_supervised_tasks(
        tasks, _worker_execute, workers, recycle_after=200,
        initial_fatal_attempts={key: initial[key] for key in by_key},
        on_fatal=on_fatal, on_result=on_result,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--host-assignment", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--authorization-file", required=True, type=Path)
    parser.add_argument("--host-registry", type=Path)
    parser.add_argument("--migration-file", type=Path)
    parser.add_argument("--group", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--stages", nargs="+", choices=config.EXECUTION_STAGES, required=True)
    args = parser.parse_args(argv)
    ranks = [config.EXECUTION_STAGES.index(stage) for stage in args.stages]
    if ranks != sorted(set(ranks)):
        parser.error("--stages must be unique and follow the registered Section 1 order")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    assert_process_environment()
    set_offline_environment()
    if "TABPFN_TOKEN" in os.environ:
        raise RuntimeError("TABPFN_TOKEN must be unset during Phase 1 execution")
    _authorized(args.authorization_file.resolve())
    expected_workers = 1 if args.group == "gpu" else 32
    if args.workers != expected_workers:
        raise RuntimeError(f"{args.group} scheduler requires {expected_workers} workers")

    root = args.registered_root.resolve()
    assignment = args.host_assignment.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    runtime_root = args.runtime_root.resolve()
    driver_dir = Path(__file__).resolve().parent
    host_registry = (args.host_registry or (driver_dir / "host_registry.json")).resolve()
    migrations = load_migrations(args.migration_file.resolve() if args.migration_file else None)

    registered_hashes = verify_registered_inputs(root)
    driver_hash = verify_driver_manifest(driver_dir)
    cleaned_hashes = ensure_cleaned_data_manifest(root, runtime_root)
    plan = build_plan(root, assignment)
    role, actual_host = resolve_registered_host(host_registry)
    cpu_name = cpu_model_name()
    gpu_name = gpu_model_name() if args.group == "gpu" else ""
    validate_hardware(
        WorkUnit("M1", "probe", "raw", 0, "gpu" if args.group == "gpu" else "cpu", role),
        role, actual_host, cpu_name, gpu_name,
    )
    if (args.group == "gpu") != (role == "4090"):
        raise RuntimeError(f"scheduler group {args.group} cannot run on registered role {role}")
    weights = registered_checkpoint_paths(root) if args.group == "gpu" else {}
    identity_for = _identity_factory(root, driver_dir, assignment)
    environment = _environment_record(
        actual_host, role, cpu_name, gpu_name, registered_hashes,
        cleaned_hashes, driver_hash, weights,
    )

    groups = {"gpu", "ftt"} if args.group == "gpu" else {"cpu", "none"}
    selected_stages = set(args.stages)
    chosen = [
        unit for unit in plan
        if unit.group in groups and unit.host == role and execution_stage(unit) in selected_stages
    ]
    singleton = runtime_root / f"scheduler.{actual_host}.lock"
    with exclusive_lock(singleton):
        quarantine = runtime_root / "quarantine"
        pending: list[WorkUnit] = []
        for unit in chosen:
            complete, migration = _checkpoint_state(
                checkpoint_path(checkpoint_root, unit), identity_for(unit.key),
                migrations, quarantine,
            )
            if migration is not None:
                _append_jsonl(runtime_root / "checkpoint_migrations.jsonl", migration)
            if not complete:
                pending.append(unit)
        _run_pending_units(
            pending, root, checkpoint_root, identity_for, weights,
            environment, runtime_root, args.workers,
        )


if __name__ == "__main__":
    main()
