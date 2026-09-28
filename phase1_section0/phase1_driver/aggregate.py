from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Iterator
from itertools import groupby
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .checkpoints import (
    CheckpointIdentity, checkpoint_path, columnar_to_rows, load_migrations,
    read_checkpoint_with_migration,
)
from .execution import WorkUnit
from .planner import build_plan
from .runtime import sha256_file, verify_driver_manifest, verify_registered_inputs


def iter_plan_groups(units: Iterable[WorkUnit]) -> Iterator[tuple[tuple[str, str], list[WorkUnit]]]:
    def identifier_key(unit: WorkUnit) -> tuple[Any, ...]:
        try:
            return (0, int(unit.identifier))
        except ValueError:
            return (1, unit.identifier)
    ordered = sorted(
        units,
        key=lambda unit: (unit.module, identifier_key(unit), unit.variant, unit.fold, unit.group),
    )
    for key, values in groupby(ordered, key=lambda unit: (unit.module, unit.identifier)):
        yield key, list(values)


def load_complete_checkpoints(
    checkpoint_root: Path,
    units: Iterable[WorkUnit],
    identity_factory: Callable[[str], CheckpointIdentity],
    reject_extras: bool = True,
    migrations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    unit_list = list(units)
    expected_paths = {checkpoint_path(checkpoint_root, unit).resolve(): unit for unit in unit_list}
    if reject_extras:
        actual_paths = {path.resolve() for path in checkpoint_root.rglob("*.json")}
        extras = actual_paths - set(expected_paths)
        if extras:
            raise RuntimeError(f"unexpected checkpoint: {sorted(map(str, extras))[0]}")
    payloads: list[dict[str, Any]] = []
    for path, unit in expected_paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing checkpoint for {unit.key}: {path}")
        payload, migration = read_checkpoint_with_migration(
            path, identity_factory(unit.key), migrations or []
        )
        if migration is not None:
            payload.setdefault("log", {})["checkpoint_migration"] = migration
        expected_unit = {
            "module": unit.module, "identifier": unit.identifier,
            "variant": unit.variant, "fold": unit.fold,
            "group": unit.group, "host": unit.host,
        }
        if payload.get("unit") != expected_unit:
            raise RuntimeError(f"checkpoint unit mismatch for {unit.key}")
        payloads.append(payload)
    validate_payload_uniqueness(payloads)
    validate_prediction_counts(payloads)
    return payloads


def validate_payload_uniqueness(payloads: Iterable[dict[str, Any]]) -> None:
    metric_keys: set[tuple[Any, ...]] = set()
    prediction_keys: set[tuple[Any, ...]] = set()
    for payload in payloads:
        for row in payload.get("metrics", []):
            key = (row["id"], row["variant"], int(row["fold"]), row["model"])
            if key in metric_keys:
                raise RuntimeError(f"duplicate metric key: {key}")
            metric_keys.add(key)
        for row in columnar_to_rows(payload.get("predictions", {})):
            key = (row["variant"], int(row["fold"]), int(row["row_index"]), row["model"])
            if key in prediction_keys:
                raise RuntimeError(f"duplicate prediction key: {key}")
            prediction_keys.add(key)


def validate_prediction_counts(payloads: Iterable[dict[str, Any]]) -> None:
    for payload in payloads:
        if payload.get("unit_failed"):
            continue
        if payload["unit"]["module"] not in {"M1", "M2", "M4"}:
            continue
        predictions = columnar_to_rows(payload.get("predictions", {}))
        counts: dict[str, int] = {}
        for row in predictions:
            model = str(row["model"])
            counts[model] = counts.get(model, 0) + 1
        for metric in payload.get("metrics", []):
            model = str(metric["model"])
            actual = counts.get(model, 0)
            if metric["status"] == "failed":
                if actual:
                    raise RuntimeError(
                        f"failed model has prediction rows: {model} count={actual}"
                    )
            elif actual != int(metric["n_test"]):
                raise RuntimeError(
                    f"prediction row count mismatch for {model}: "
                    f"expected={metric['n_test']} actual={actual}"
                )


def _write_group(module: str, identifier: str, payloads: list[dict[str, Any]], output_root: Path) -> None:
    predictions = [row for payload in payloads for row in columnar_to_rows(payload.get("predictions", {}))]
    if predictions:
        path = output_root / "preds" / module / f"{int(identifier)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(predictions, columns=config.PREDICTION_COLUMNS).sort_values(
            ["variant", "fold", "row_index", "model"]
        ).to_parquet(path, index=False, compression="snappy")
    if module == "M6":
        if len(payloads) != 1:
            raise RuntimeError(f"M6 dataset {identifier} must have exactly one checkpoint")
        path = output_root / "marginal_train" / f"{identifier}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payloads[0]["marginal"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def aggregate_complete(
    checkpoint_root: Path,
    output_root: Path,
    plan: list[WorkUnit],
    identity_factory: Callable[[str], CheckpointIdentity],
    modules: set[str] | None = None,
    groups: set[str] | None = None,
    migrations: list[dict[str, Any]] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    requested_groups = groups or {"cpu", "gpu", "ftt"}
    plan_groups: set[str] = set()
    for group in requested_groups:
        plan_groups.update({"cpu", "none"} if group == "cpu" else {group})
    selected = [
        unit for unit in plan
        if (modules is None or unit.module in modules) and unit.group in plan_groups
    ]
    expected_all = {checkpoint_path(checkpoint_root, unit).resolve() for unit in selected}
    # The checkpoint tree is scanned exactly once. Other modules are allowed to
    # coexist because Phase 1 is delivered one module at a time.
    actual_all = {path.resolve() for path in checkpoint_root.rglob("*.json")}
    selected_roots = {(checkpoint_root / module).resolve() for module in (modules or {u.module for u in selected})}
    actual_selected = {
        path for path in actual_all
        if path.stem in plan_groups
        and any(root == path or root in path.parents for root in selected_roots)
    }
    extras = actual_selected - expected_all
    if extras:
        raise RuntimeError(f"unexpected checkpoint: {sorted(map(str, extras))[0]}")
    missing = expected_all - actual_all
    if missing:
        raise RuntimeError(f"missing checkpoint: {sorted(map(str, missing))[0]}")
    all_metrics: dict[str, list[dict[str, Any]]] = {}
    all_selection: dict[str, list[dict[str, Any]]] = {}
    fm_inner: list[dict[str, Any]] = []
    ftt_rows: list[dict[str, Any]] = []
    run_logs: list[dict[str, Any]] = []
    environments: dict[str, list[dict[str, Any]]] = {}
    for (module, identifier), group_units in iter_plan_groups(selected):
        payloads = load_complete_checkpoints(
            checkpoint_root, group_units, identity_factory, reject_extras=False,
            migrations=migrations,
        )
        _write_group(module, identifier, payloads, output_root)
        all_metrics.setdefault(module, []).extend(row for payload in payloads for row in payload.get("metrics", []))
        for payload in payloads:
            if payload["unit"]["group"] == "cpu":
                all_selection.setdefault(module, []).extend(payload.get("selection", []))
            if module == "M1" and payload["unit"]["group"] == "gpu":
                fm_inner.extend(payload.get("selection", []))
            ftt_rows.extend(payload.get("ftt_log", []))
            run_logs.append(payload["log"])
            environment = payload.get("environment", {})
            if environment:
                host = str(environment["host"])
                records = environments.setdefault(host, [])
                if environment not in records:
                    records.append(environment)
    for module, rows in all_metrics.items():
        if rows:
            path = output_root / "metrics" / f"{module}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=config.METRIC_COLUMNS).sort_values(
                ["id", "variant", "fold", "model"]
            ).to_csv(path, index=False)
    for module, rows in all_selection.items():
        if rows:
            path = output_root / "selection" / f"{module}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=config.SELECTION_COLUMNS).sort_values(
                ["id", "variant", "fold", "model"]
            ).to_csv(path, index=False)
    if fm_inner:
        path = output_root / "selection" / "M1_fm_inner.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fm_inner, columns=config.SELECTION_COLUMNS).sort_values(
            ["id", "variant", "fold", "model"]
        ).to_csv(path, index=False)
    if ftt_rows:
        pd.DataFrame(ftt_rows, columns=["openml_id", "fold", "best_epoch", "n_params"]).sort_values(
            ["openml_id", "fold"]
        ).to_csv(output_root / "ftt_log.csv", index=False)
    with (output_root / "run_log.txt").open("w", encoding="utf-8", newline="\n") as stream:
        for record in sorted(run_logs, key=lambda item: item["task_key"]):
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (output_root / "environment_run.json").write_text(
        json.dumps({"hosts": environments}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--host-assignment", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--modules", nargs="+", choices=("M1", "M2", "M4", "M5", "M6"), required=True)
    parser.add_argument("--groups", nargs="+", choices=("cpu", "gpu", "ftt"), required=True)
    parser.add_argument("--migration-file", type=Path)
    args = parser.parse_args()
    registered_root = args.registered_root.resolve()
    assignment = args.host_assignment.resolve()
    plan = build_plan(registered_root, assignment)
    verify_registered_inputs(registered_root)
    driver_dir = Path(__file__).resolve().parent
    driver_hash = verify_driver_manifest(driver_dir)
    registration_hash = sha256_file(registered_root / "MANIFEST.sha256")
    assignment_hash = sha256_file(assignment)
    def identity(task_key: str) -> CheckpointIdentity:
        return CheckpointIdentity(
            registration_hash, driver_hash, assignment_hash, task_key
        )
    migrations = load_migrations(args.migration_file.resolve() if args.migration_file else None)
    aggregate_complete(
        args.checkpoint_root.resolve(), args.output_root.resolve(), plan, identity,
        modules=set(args.modules), groups=set(args.groups), migrations=migrations,
    )


if __name__ == "__main__":
    main()
