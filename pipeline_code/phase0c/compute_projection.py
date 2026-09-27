from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


CPU_MODELS = ["lr", "ebm", "catboost", "lgbm", "xgb", "rf"]
FM_MODELS = ["tabpfn35", "tabicl2", "tabpfn2"]
GPU_MODELS = FM_MODELS + ["ftt"]
SIZES = [100, 250, 500, 1000]


def fit_log_models(timing: pd.DataFrame):
    timing = timing.copy()
    timing["total_seconds"] = timing.fit_seconds + timing.predict_seconds
    models: dict[tuple[str, str], np.ndarray] = {}
    for (model, stage), g in timing.groupby(["model", "stage"]):
        x = np.column_stack(
            [
                np.ones(len(g)),
                np.log(np.maximum(g.n_train.to_numpy(float), 2)),
                np.log(np.maximum(g.p_train.to_numpy(float), 1)),
            ]
        )
        y = np.log(np.maximum(g.total_seconds.to_numpy(float), 1e-6))
        models[(str(model), str(stage))] = np.linalg.lstsq(x, y, rcond=None)[0]
    return models


def predict(models, model: str, stage: str, n: float, p: float) -> float:
    beta = models[(model, stage)]
    row = np.array([1.0, math.log(max(float(n), 2.0)), math.log(max(float(p), 1.0))])
    return float(math.exp(float(row @ beta)))


def add_standard(acc, module: str, models, n: float, p: float, variants: float = 1.0):
    outer_n, inner_n = 0.8 * n, 0.64 * n
    for model in CPU_MODELS:
        seconds = variants * (
            5 * predict(models, model, "outer", outer_n, p)
            + 25 * predict(models, model, "inner", inner_n, p)
        )
        acc.append((module, model, "CPU", seconds))
    for model in FM_MODELS:
        seconds = variants * 5 * predict(models, model, "outer", outer_n, p)
        acc.append((module, model, "GPU", seconds))


def load_pool(root: Path) -> pd.DataFrame:
    old = pd.read_csv(root / "project_resume/tabular_prior_v2/phase0b_outputs/datasets_stratifiers.csv")
    new = pd.read_csv(root / "phase0c_outputs/datasets_stratifiers_ext.csv")
    d = pd.concat([old, new], ignore_index=True)
    d = d[d.excluded_reason.fillna("").eq("")].copy()
    meta = pd.read_csv(root / "phase0c_outputs/metadata.csv")
    d = d.merge(meta[["openml_id", "n1", "n0"]], on="openml_id", how="left", validate="one_to_one")
    if len(d) != 478:
        raise RuntimeError(f"pool rows={len(d)}")
    d["target_region"] = d.auc_lr.between(0.60, 0.85, inclusive="both")
    return d


def m4_variant_count(root: Path, row) -> int:
    folds = pd.read_parquet(root / "cleaned_data" / f"{int(row.openml_id)}.parquet", columns=["fold"])["fold"]
    n_train_min = int(row.n) - int(folds.value_counts().max())
    eligible = [size for size in SIZES if size < n_train_min and float(row.minority_rate) * size >= 10]
    return 5 * len(eligible)


def main() -> None:
    root = Path(__file__).resolve().parent
    timing = pd.read_csv(root / "phase0c_outputs/pilot_timing.csv")
    models = fit_log_models(timing)
    pool = load_pool(root)
    target = pool[pool.target_region]
    acc: list[tuple[str, str, str, float]] = []

    for row in pool.itertuples(index=False):
        add_standard(acc, "M1", models, float(row.n), float(row.p_used))
        for model in FM_MODELS:
            seconds = 25 * predict(models, model, "inner", 0.64 * float(row.n), float(row.p_used))
            acc.append(("M1", model, "GPU", seconds))
        for model in GPU_MODELS[-1:]:
            seconds = 5 * predict(models, model, "outer", 0.8 * float(row.n), float(row.p_used))
            acc.append(("M1", model, "GPU", seconds))

    for row in target.itertuples(index=False):
        # Per replicate: 9 variants add 8 columns and 4 dispersed-noise variants add 15.
        add_standard(acc, "M2", models, float(row.n), float(row.p_used) + 8, variants=27)
        add_standard(acc, "M2", models, float(row.n), float(row.p_used) + 15, variants=12)
        # All included datasets have p>=5, so k=1 and k=3 each contribute six variants.
        add_standard(acc, "M3", models, float(row.n), max(1.0, float(row.p_used) - 1), variants=6)
        add_standard(acc, "M3", models, float(row.n), max(1.0, float(row.p_used) - 3), variants=6)
        variants = m4_variant_count(root, row)
        for size in SIZES:
            folds = pd.read_parquet(root / "cleaned_data" / f"{int(row.openml_id)}.parquet", columns=["fold"])["fold"]
            n_train_min = int(row.n) - int(folds.value_counts().max())
            if size >= n_train_min or float(row.minority_rate) * size < 10:
                continue
            for model in CPU_MODELS:
                seconds = 5 * (
                    5 * predict(models, model, "outer", size, float(row.p_used))
                    + 25 * predict(models, model, "inner", 0.8 * size, float(row.p_used))
                )
                acc.append(("M4", model, "CPU", seconds))
            for model in FM_MODELS:
                seconds = 5 * 5 * predict(models, model, "outer", size, float(row.p_used))
                acc.append(("M4", model, "GPU", seconds))

    # M5 has 48 mechanism cells, four training sizes and 20 repetitions = 3,840 cases.
    for size in (100, 300, 1000, 3000):
        cases = 48 * 20
        for model in CPU_MODELS:
            seconds = cases * (
                predict(models, model, "outer", size, 32)
                + 5 * predict(models, model, "inner", 0.8 * size, 32)
            )
            acc.append(("M5", model, "CPU", seconds))
        for model in FM_MODELS:
            seconds = cases * predict(models, model, "outer", size, 32)
            acc.append(("M5", model, "GPU", seconds))

    resource = pd.DataFrame(acc, columns=["module", "model", "resource", "seconds"])
    by_module = resource.groupby(["module", "resource"], as_index=False).seconds.sum()
    by_model = resource.groupby(["model", "resource"], as_index=False).seconds.sum()
    modules = ["M1", "M2", "M3", "M4", "M5", "M6"]
    module_rows = []
    for module in modules:
        c = by_module[(by_module.module == module) & (by_module.resource == "CPU")].seconds.sum() / 3600
        g = by_module[(by_module.module == module) & (by_module.resource == "GPU")].seconds.sum() / 3600
        module_rows.append({"module": module, "cpu_core_hours": c, "gpu_hours": g})
    module_df = pd.DataFrame(module_rows)

    env = json.loads((root / "phase0c_outputs/environment_phase1.json").read_text(encoding="utf-8"))
    cpu_model = env.get("cpu", {}).get("model") or env.get("cpu_model") or "Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz"
    gpu_model = env.get("gpu", {}).get("model") or env.get("gpu_model") or "NVIDIA GeForce RTX 4090"
    median_fit = timing.groupby("model").fit_seconds.median().sort_index()
    ebm_hours = by_model.loc[by_model.model.eq("ebm"), "seconds"].sum() / 3600
    cpu_total = module_df.cpu_core_hours.sum()
    gpu_total = module_df.gpu_hours.sum()
    ebm_share = 100 * ebm_hours / cpu_total
    wall_rows = []
    for extra in [0, 1, 2, 4, 8]:
        cores = 16 + 32 * extra
        wall = sum(max(r.cpu_core_hours / cores, r.gpu_hours) for r in module_df.itertuples(index=False))
        wall_rows.append((extra, cores, wall, wall / 24))

    lines = [
        "# Phase 1 compute projection",
        "",
        "This projection uses the complete Phase 0c pilot timing table. For each model and stage (outer/inner), "
        "a log-linear least-squares model of total fit-plus-predict seconds against log training rows and log feature count "
        "was fitted, then evaluated for every final-pool dataset and registered module workload.",
        "",
        f"- Final pool: {len(pool)} datasets; target region (`0.60 <= auc_lr <= 0.85`): {len(target)} datasets.",
        f"- CPU: `{cpu_model}`; current-server traditional-model capacity: 16 concurrent single-thread workers.",
        f"- GPU: `{gpu_model}`; one sequential GPU worker.",
        "- M2 assumes the default three injection repetitions (39 variants per target dataset).",
        "- M3 assumes all 12 registered removal variants; every target dataset has at least five retained features.",
        "- M4 eligibility is evaluated per dataset from its frozen folds, sample size, and minority rate.",
        "- M5 includes 192 registered grid cells × 20 repetitions = 3,840 cases.",
        "- M6 fits no model; the tables therefore report zero traditional-model CPU core-hours and zero GPU hours for M6. "
        "Its marginal-AUROC bookkeeping cost is outside the model-fit projection.",
        "",
        "## Projected resource time",
        "",
        "| Module | Single-thread CPU core-hours | GPU hours |",
        "|---|---:|---:|",
    ]
    for r in module_df.itertuples(index=False):
        lines.append(f"| {r.module} | {r.cpu_core_hours:.2f} | {r.gpu_hours:.2f} |")
    lines += [
        f"| **Total** | **{cpu_total:.2f}** | **{gpu_total:.2f}** |",
        "",
        "## Projected resource time by model",
        "",
        "| Model | Resource | Hours | Median pilot fit seconds |",
        "|---|---|---:|---:|",
    ]
    for r in by_model.sort_values(["resource", "model"]).itertuples(index=False):
        lines.append(f"| {r.model} | {r.resource} | {r.seconds / 3600:.2f} | {median_fit[r.model]:.6f} |")
    lines += [
        "",
        "## Wall-clock scenarios",
        "",
        "Within each module, CPU and GPU work are assumed to run concurrently; modules are summed in registered sequence. "
        "Additional 32-core instances are assumed to have equivalent per-core throughput and perfect task distribution.",
        "",
        "| Additional 32-core CPU instances | Available traditional-model workers | Wall hours | Wall days |",
        "|---:|---:|---:|---:|",
    ]
    for extra, cores, hours, days in wall_rows:
        lines.append(f"| {extra} | {cores} | {hours:.2f} | {days:.2f} |")
    lines += [
        "",
        "## EBM share",
        "",
        f"Projected EBM time is {ebm_hours:.2f} single-thread core-hours, {ebm_share:.2f}% of all projected traditional-model CPU core-hours.",
        "",
        "These are workload projections from the fixed pilot and final-pool size distribution, not performance comparisons.",
    ]
    (root / "phase0c_outputs/compute_projection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
