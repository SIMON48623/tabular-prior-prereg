from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import config
from .execution import WorkUnit, _generator


def execution_stage(unit: WorkUnit, priority_variants: list[str] | None = None) -> str:
    if unit.module != "M2":
        return unit.module
    priority = priority_variants if priority_variants is not None else config.m2_priority_variants()
    if unit.variant in priority:
        return f"M2_PRIORITY_R{unit.variant.rsplit('r', 1)[1]}"
    return "M2_REMAINDER"


def build_plan(root: Path, assignment_path: Path) -> list[WorkUnit]:
    pool = pd.read_csv(root / "final_pool" / "datasets_stratifiers.csv")
    pool = pool.loc[pool["excluded_reason"].fillna("").eq("")].copy()
    assignment = pd.read_csv(assignment_path)
    if list(assignment.columns) != ["openml_id", "host"] or len(assignment) != 478:
        raise RuntimeError("host assignment must have exact schema and 478 rows")
    hosts = dict(zip(assignment.openml_id.astype(int), assignment.host.astype(str), strict=True))
    if set(hosts) != set(pool.openml_id.astype(int)):
        raise RuntimeError("host assignment IDs differ from final pool")
    generator = _generator(root)
    if int(generator.INJECT_REPS) != config.INJECT_REPS:
        raise RuntimeError(
            f"registered INJECT_REPS={generator.INJECT_REPS} differs from driver {config.INJECT_REPS}"
        )
    units: list[WorkUnit] = []
    stratifier = pd.read_csv(root / "frozen" / "stratifier_v2.csv")
    if "target_region" not in stratifier or "openml_id" not in stratifier:
        raise RuntimeError("frozen/stratifier_v2.csv lacks target_region or openml_id")
    target_flag = stratifier["target_region"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    target_ids = set(stratifier.loc[target_flag, "openml_id"].astype(int))
    if len(target_ids) != 238:
        raise RuntimeError(f"target_region must contain exactly 238 datasets, got {len(target_ids)}")
    if not target_ids.issubset(set(pool.openml_id.astype(int))):
        raise RuntimeError("target_region contains IDs outside the final pool")
    for openml_id in sorted(pool.openml_id.astype(int)):
        for fold in range(5):
            units.append(WorkUnit("M1", str(openml_id), "raw", fold, "cpu", hosts[openml_id]))
            units.append(WorkUnit("M1", str(openml_id), "raw", fold, "gpu", "4090"))
            units.append(WorkUnit("M1", str(openml_id), "raw", fold, "ftt", "4090"))
    for openml_id in sorted(target_ids):
        units.append(WorkUnit("M6", str(openml_id), "raw", -1, "none", hosts[openml_id]))
    priority = config.m2_priority_variants()
    remainder = config.m2_remainder_variants()
    for openml_id in sorted(target_ids):
        for variant in priority + remainder:
            for fold in range(5):
                units.append(WorkUnit("M2", str(openml_id), variant, fold, "cpu", hosts[openml_id]))
                units.append(WorkUnit("M2", str(openml_id), variant, fold, "gpu", "4090"))
        frame = pd.read_parquet(root / "cleaned_data" / f"{openml_id}.parquet", columns=["y", "fold"])
        smallest_train = len(frame) - int(frame["fold"].value_counts().max())
        minority = float(frame["y"].mean())
        for size in generator.SUBSAMPLE_SIZES:
            if generator.subsample_eligible(size, smallest_train, minority):
                for rep in range(generator.SUBSAMPLE_REPS):
                    for fold in range(5):
                        variant = f"n{size}_r{rep}"
                        units.append(WorkUnit("M4", str(openml_id), variant, fold, "cpu", hosts[openml_id]))
                        units.append(WorkUnit("M4", str(openml_id), variant, fold, "gpu", "4090"))
    for gen in generator.SYN_GEN:
        for k in generator.SYN_K:
            for rho in generator.SYN_RHO:
                for auc in generator.SYN_AUC:
                    identifier = f"{gen}|{k}|{rho}|{auc}"
                    for size in generator.SYN_N_TRAIN:
                        for rep in range(generator.SYN_REPS):
                            variant = f"n{size}_r{rep}"
                            units.append(WorkUnit("M5", identifier, variant, 0, "cpu", "686"))
                            units.append(WorkUnit("M5", identifier, variant, 0, "gpu", "4090"))
    stage_rank = {name: number for number, name in enumerate(config.EXECUTION_STAGES)}
    return sorted(
        units,
        key=lambda unit: (
            stage_rank[execution_stage(unit, priority)], unit.identifier,
            unit.variant, unit.fold, unit.group,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-root", required=True, type=Path)
    parser.add_argument("--host-assignment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build_plan(args.registered_root.resolve(), args.host_assignment.resolve())
    with args.output.open("w", encoding="utf-8") as stream:
        for unit in plan:
            stream.write(json.dumps(asdict(unit), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
