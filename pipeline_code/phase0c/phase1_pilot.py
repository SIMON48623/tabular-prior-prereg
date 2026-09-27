from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONHASHSEED", "13")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits


SEED = 13
OUTER_SEED = 42
CPU_MODELS = ["lr", "ebm", "catboost", "lgbm", "xgb", "rf"]
FM_MODELS = ["tabpfn35", "tabicl2", "tabpfn2"]
LEVELS = (0.65, 0.75, 0.85, 0.95)
CHECKPOINTS = {"tabicl2": "tabicl-classifier-v2-20260212.ckpt"}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def pilot_specs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    i = 0
    for n in (500, 1000, 2000):
        for p in (10, 30):
            for frac in (0.3, 0.8):
                rows.append({"i": i, "dataset": f"pilot{i}", "n": n, "p": p, "frac": frac})
                i += 1
    return rows


def variants(module: str) -> list[str]:
    if module == "M1":
        return ["raw"]
    out = ["noise_r0"]
    for lev in LEVELS:
        out.extend([f"conc_{lev}_r0", f"disp_{lev}_r0", f"dispn_{lev}_r0"])
    return out


def make_base(spec: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_classification(
        n_samples=int(spec["n"]),
        n_features=int(spec["p"]),
        n_informative=max(2, round(float(spec["frac"]) * int(spec["p"]))),
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.65],
        flip_y=0.05,
        class_sep=0.5,
        random_state=1000 + int(spec["i"]),
    )
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])]), np.asarray(y, dtype=int)


def make_variant(root: Path, spec: dict[str, Any], variant: str) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_base(spec)
    if variant == "raw":
        return X, y
    code_dir = root / "project" / "tabular_prior_v2" / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from generators import injection_block

    if variant == "noise_r0":
        block, names = injection_block(y, str(spec["dataset"]), "noise", None, rep=0)
    else:
        arm, level, _rep = variant.split("_")
        arm_name = {"conc": "concentrated", "disp": "dispersed", "dispn": "dispersed_noise"}[arm]
        block, names = injection_block(y, str(spec["dataset"]), arm_name, float(level), rep=0)
    block = pd.DataFrame(np.asarray(block), columns=names)
    return pd.concat([X.reset_index(drop=True), block], axis=1), y


def split_outer(X: pd.DataFrame, y: np.ndarray, fold: int):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    return list(splitter.split(X, y))[fold]


def preprocessors(columns: list[str], categorical: list[str]):
    numeric = [c for c in columns if c not in categorical]
    lr_parts: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        lr_parts.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        lr_parts.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    lr_prep = ColumnTransformer(lr_parts, remainder="drop")
    ord_parts: list[tuple[str, Pipeline, list[str]]] = []
    if categorical:
        ord_parts.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical))
    if numeric:
        ord_parts.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    ord_prep = ColumnTransformer(ord_parts, remainder="drop", verbose_feature_names_out=False)
    return lr_prep, ord_prep, len(categorical)


def cpu_fit_predict(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, categorical: list[str]):
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    lr_prep, ord_prep, n_cat = preprocessors(list(X_train.columns), categorical)
    fit_start = time.perf_counter()
    if model_name == "lr":
        with threadpool_limits(limits=1):
            Xtr = lr_prep.fit_transform(X_train)
            model = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=2000, random_state=42)
            model.fit(Xtr, y_train)
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        with threadpool_limits(limits=1):
            pred = model.predict_proba(lr_prep.transform(X_test))[:, 1]
    else:
        Xtr = np.asarray(ord_prep.fit_transform(X_train), dtype=float)
        cat_idx = list(range(n_cat))
        if model_name == "ebm":
            model = ExplainableBoostingClassifier(random_state=SEED, n_jobs=1, feature_types=["nominal"] * n_cat + ["continuous"] * (Xtr.shape[1] - n_cat))
            model.fit(Xtr, y_train)
        elif model_name == "catboost":
            Xtr_cb = pd.DataFrame(Xtr)
            for idx in cat_idx:
                Xtr_cb[idx] = Xtr_cb[idx].astype(int).astype(str)
            model = CatBoostClassifier(verbose=0, random_seed=SEED, thread_count=1, cat_features=cat_idx)
            model.fit(Xtr_cb, y_train)
        elif model_name == "lgbm":
            model = LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1)
            model.fit(Xtr, y_train, categorical_feature=cat_idx)
        elif model_name == "xgb":
            model = XGBClassifier(random_state=SEED, n_jobs=1, enable_categorical=True)
            if n_cat:
                Xtr_xgb = pd.DataFrame(Xtr)
                for idx in cat_idx:
                    Xtr_xgb[idx] = Xtr_xgb[idx].astype(int).astype("category")
                model.fit(Xtr_xgb, y_train)
            else:
                model.fit(Xtr, y_train)
        elif model_name == "rf":
            model = RandomForestClassifier(random_state=SEED, n_jobs=1)
            model.fit(Xtr, y_train)
        else:
            raise ValueError(model_name)
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        Xte = np.asarray(ord_prep.transform(X_test), dtype=float)
        if model_name == "catboost":
            Xte_cb = pd.DataFrame(Xte)
            for idx in cat_idx:
                Xte_cb[idx] = Xte_cb[idx].astype(int).astype(str)
            pred = model.predict_proba(Xte_cb)[:, 1]
        elif model_name == "xgb" and n_cat:
            Xte_xgb = pd.DataFrame(Xte)
            for idx in cat_idx:
                Xte_xgb[idx] = Xte_xgb[idx].astype(int).astype("category")
            pred = model.predict_proba(Xte_xgb)[:, 1]
        else:
            pred = model.predict_proba(Xte)[:, 1]
    predict_seconds = time.perf_counter() - pred_start
    return np.asarray(pred, dtype=float), fit_seconds, predict_seconds


def timing_row(dataset: str, variant: str, fold: Any, model: str, stage: str, n_train: int, p_train: int, fit_seconds: float, predict_seconds: float, device: str):
    return {"dataset": dataset, "variant": variant, "fold": fold, "model": model, "stage": stage, "n_train": n_train, "p_train": p_train, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "device": device}


def cpu_task(args: tuple[str, str, int, str]) -> dict[str, Any]:
    root_s, module, spec_i, variant = args
    root = Path(root_s)
    spec = pilot_specs()[spec_i]
    dataset = str(spec["dataset"])
    fold = int(os.environ["PHASE0C_PILOT_FOLD"])
    out_path = root / "run_state" / "pilot" / "cpu" / module / f"{dataset}__{variant}__f{fold}.json"
    if out_path.exists():
        return {"path": str(out_path), "cached": True}
    X, y = make_variant(root, spec, variant)
    train_idx, test_idx = split_outer(X, y, fold)
    Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    timings: list[dict[str, Any]] = []
    outer_auc: dict[str, float] = {}
    failures: dict[str, str] = {}
    for model_name in CPU_MODELS:
        try:
            pred, fs, ps = cpu_fit_predict(model_name, Xtr, ytr, Xte, [])
            outer_auc[model_name] = float(roc_auc_score(yte, pred))
            timings.append(timing_row(dataset, variant, fold, model_name, "outer", len(ytr), X.shape[1], fs, ps, "CPU"))
        except Exception as exc:
            failures[f"outer:{model_name}"] = f"{type(exc).__name__}: {exc}"

    inner_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_auc: dict[str, list[float]] = {}
    for model_name in CPU_MODELS:
        scores: list[float] = []
        try:
            for inner_fold, (itr, iva) in enumerate(inner_splitter.split(Xtr, ytr)):
                pred, fs, ps = cpu_fit_predict(model_name, Xtr.iloc[itr], ytr[itr], Xtr.iloc[iva], [])
                scores.append(float(roc_auc_score(ytr[iva], pred)))
                timings.append(timing_row(dataset, variant, f"o{fold}_i{inner_fold}", model_name, "inner", len(itr), X.shape[1], fs, ps, "CPU"))
            inner_auc[model_name] = scores
        except Exception as exc:
            failures[f"inner:{model_name}"] = f"{type(exc).__name__}: {exc}"

    eligible = [m for m in CPU_MODELS if m in inner_auc and m in outer_auc and len(inner_auc[m]) == 5]
    selected_s = max(eligible, key=lambda m: (float(np.mean(inner_auc[m])), -CPU_MODELS.index(m))) if eligible else None
    add_eligible = [m for m in ("lr", "ebm") if m in eligible]
    selected_add = max(add_eligible, key=lambda m: (float(np.mean(inner_auc[m])), -("lr", "ebm").index(m))) if add_eligible else None
    payload = {"module": module, "dataset": dataset, "variant": variant, "fold": fold, "outer_auc": outer_auc, "inner_auc": inner_auc, "selected_s": selected_s, "selected_add": selected_add, "timings": timings, "failures": failures}
    atomic_json(out_path, payload)
    return {"path": str(out_path), "cached": False, "failures": failures}


def run_cpu(root: Path, workers: int) -> None:
    tasks: list[tuple[str, str, int, str, int]] = []
    for module in ("M1", "M2"):
        for spec in pilot_specs():
            for variant in variants(module):
                for fold in range(5):
                    tasks.append((str(root), module, int(spec["i"]), variant, fold))
    pending = []
    for root_s, module, spec_i, variant, fold in tasks:
        path = root / "run_state" / "pilot" / "cpu" / module / f"pilot{spec_i}__{variant}__f{fold}.json"
        if not path.exists():
            pending.append((root_s, module, spec_i, variant, fold))
    print(f"CPU tasks total={len(tasks)} pending={len(pending)} workers={workers}", flush=True)

    def submit_arg(item):
        root_s, module, spec_i, variant, fold = item
        return root_s, module, spec_i, variant, fold

    # A worker receives its outer-fold number through a per-call wrapper because
    # the checkpoint unit is dataset x variant x outer fold.
    def one(item):
        root_s, module, spec_i, variant, fold = item
        os.environ["PHASE0C_PILOT_FOLD"] = str(fold)
        return cpu_task((root_s, module, spec_i, variant))

    # ProcessPoolExecutor cannot pickle a nested function with spawn, while the
    # target server uses fork.  Keep a serial fallback for portability.
    if workers == 1:
        results = [one(t) for t in pending]
        for i, result in enumerate(results, 1):
            print(f"CPU {i}/{len(pending)} {result['path']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(cpu_task_with_fold, t): t for t in pending}
            for i, fut in enumerate(as_completed(future_map), 1):
                result = fut.result()
                print(f"CPU {i}/{len(pending)} {result['path']} failures={len(result.get('failures', {}))}", flush=True)


def cpu_task_with_fold(item: tuple[str, str, int, str, int]):
    root_s, module, spec_i, variant, fold = item
    os.environ["PHASE0C_PILOT_FOLD"] = str(fold)
    return cpu_task((root_s, module, spec_i, variant))


def seed_torch() -> None:
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)


def gpu_fit_predict(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame):
    import torch
    from tabicl import TabICLClassifier
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    seed_torch()
    fit_start = time.perf_counter()
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    if model_name == "tabpfn35":
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="cuda", random_state=SEED)
    elif model_name == "tabpfn2":
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda", random_state=SEED)
    elif model_name == "tabicl2":
        model = TabICLClassifier(checkpoint_version=CHECKPOINTS["tabicl2"], device="cuda", random_state=SEED, allow_auto_download=False)
    else:
        raise ValueError(model_name)
    model.fit(Xtr, y_train)
    fit_seconds = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    pred = np.asarray(model.predict_proba(Xte))[:, 1]
    predict_seconds = time.perf_counter() - pred_start
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pred.astype(float), fit_seconds, predict_seconds


def ftt_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame):
    import torch
    from rtdl_revisiting_models import FTTransformer

    seed_torch()
    fit_start = time.perf_counter()
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    Xtr = np.asarray(prep.fit_transform(X_train), dtype=np.float32)
    train_pos, val_pos = train_test_split(np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=SEED)
    kwargs = FTTransformer.get_default_kwargs()
    model = FTTransformer(n_cont_features=Xtr.shape[1], cat_cardinalities=[], d_out=1, **kwargs).cuda()
    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
    n_pos = int(y_train[train_pos].sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_pos) - n_pos) / n_pos, device="cuda"))
    tx = torch.tensor(Xtr, dtype=torch.float32, device="cuda")
    ty = torch.tensor(y_train, dtype=torch.float32, device="cuda")
    best, best_state, stale, best_epoch = math.inf, None, 0, -1
    for epoch in range(100):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(tx[train_pos], None).squeeze(1), ty[train_pos])
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(tx[val_pos], None).squeeze(1), ty[val_pos]).cpu())
        if val < best:
            best, stale, best_epoch = val, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("FTT produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    fit_seconds = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    Xte = np.asarray(prep.transform(X_test), dtype=np.float32)
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.tensor(Xte, device="cuda"), None).squeeze(1)).cpu().numpy()
    predict_seconds = time.perf_counter() - pred_start
    n_params = int(sum(p.numel() for p in model.parameters()))
    del model
    gc.collect(); torch.cuda.empty_cache()
    return pred.astype(float), fit_seconds, predict_seconds, best_epoch, n_params


def gpu_task(root: Path, module: str, spec_i: int, variant: str, fold: int, gpu_name: str):
    spec = pilot_specs()[spec_i]
    dataset = str(spec["dataset"])
    out_path = root / "run_state" / "pilot" / "gpu" / module / f"{dataset}__{variant}__f{fold}.json"
    if out_path.exists():
        return out_path, True
    X, y = make_variant(root, spec, variant)
    train_idx, test_idx = split_outer(X, y, fold)
    Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    timings: list[dict[str, Any]] = []
    outer_auc: dict[str, float] = {}
    failures: dict[str, str] = {}
    for model_name in FM_MODELS:
        try:
            pred, fs, ps = gpu_fit_predict(model_name, Xtr, ytr, Xte)
            outer_auc[model_name] = float(roc_auc_score(yte, pred))
            timings.append(timing_row(dataset, variant, fold, model_name, "outer", len(ytr), X.shape[1], fs, ps, gpu_name))
        except Exception as exc:
            failures[f"outer:{model_name}"] = f"{type(exc).__name__}: {exc}"
    ftt_meta = None
    if module == "M1":
        inner_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for model_name in FM_MODELS:
            for inner_fold, (itr, iva) in enumerate(inner_splitter.split(Xtr, ytr)):
                try:
                    pred, fs, ps = gpu_fit_predict(model_name, Xtr.iloc[itr], ytr[itr], Xtr.iloc[iva])
                    _ = float(roc_auc_score(ytr[iva], pred))
                    timings.append(timing_row(dataset, variant, f"o{fold}_i{inner_fold}", model_name, "inner", len(itr), X.shape[1], fs, ps, gpu_name))
                except Exception as exc:
                    failures[f"inner:{model_name}:{inner_fold}"] = f"{type(exc).__name__}: {exc}"
        try:
            pred, fs, ps, best_epoch, n_params = ftt_fit_predict(Xtr, ytr, Xte)
            outer_auc["ftt"] = float(roc_auc_score(yte, pred))
            timings.append(timing_row(dataset, variant, fold, "ftt", "outer", len(ytr), X.shape[1], fs, ps, gpu_name))
            ftt_meta = {"best_epoch": best_epoch, "n_params": n_params}
        except Exception as exc:
            failures["outer:ftt"] = f"{type(exc).__name__}: {exc}"
    payload = {"module": module, "dataset": dataset, "variant": variant, "fold": fold, "outer_auc": outer_auc, "timings": timings, "failures": failures, "ftt_meta": ftt_meta}
    atomic_json(out_path, payload)
    return out_path, False


def run_gpu(root: Path) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    gpu_name = torch.cuda.get_device_name(0)
    tasks = [(m, int(s["i"]), v, f) for m in ("M1", "M2") for s in pilot_specs() for v in variants(m) for f in range(5)]
    done = 0
    for module, spec_i, variant, fold in tasks:
        path, cached = gpu_task(root, module, spec_i, variant, fold, gpu_name)
        done += 1
        print(f"GPU {done}/{len(tasks)} cached={cached} {path}", flush=True)


def load_jsons(path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.rglob("*.json"))]


def aggregate(root: Path) -> None:
    cpu = load_jsons(root / "run_state" / "pilot" / "cpu")
    gpu = load_jsons(root / "run_state" / "pilot" / "gpu")
    expected_each = 12 * (1 + 13) * 5
    if len(cpu) != expected_each or len(gpu) != expected_each:
        raise RuntimeError(f"incomplete checkpoints cpu={len(cpu)} gpu={len(gpu)} expected_each={expected_each}")
    failures = [(r["dataset"], r["variant"], r["fold"], r["failures"]) for r in cpu + gpu if r.get("failures")]
    if failures:
        atomic_json(root / "run_state" / "pilot_failures.json", failures)
        raise RuntimeError(f"pilot failures={len(failures)}")
    timings = [x for r in cpu + gpu for x in r["timings"]]
    timing_df = pd.DataFrame(timings)
    outer_m2 = timing_df[(timing_df["stage"] == "outer") & (timing_df["variant"] != "raw")]
    outer_m1 = timing_df[(timing_df["stage"] == "outer") & (timing_df["variant"] == "raw")]
    if len(outer_m2) != 12 * 13 * 5 * 9 or len(outer_m1) != 12 * 5 * 10:
        raise RuntimeError(f"outer timing coverage m2={len(outer_m2)} m1={len(outer_m1)}")
    out = root / "phase0c_outputs"; out.mkdir(parents=True, exist_ok=True)
    timing_df.sort_values(["dataset", "variant", "fold", "stage", "model"]).to_csv(out / "pilot_timing.csv", index=False)

    cpu_key = {(r["dataset"], r["variant"], int(r["fold"])): r for r in cpu if r["module"] == "M2"}
    gpu_key = {(r["dataset"], r["variant"], int(r["fold"])): r for r in gpu if r["module"] == "M2"}
    result_rows: list[dict[str, Any]] = []
    for spec in pilot_specs():
        dataset = str(spec["dataset"])
        for variant in variants("M2"):
            crows = [cpu_key[(dataset, variant, f)] for f in range(5)]
            grows = [gpu_key[(dataset, variant, f)] for f in range(5)]
            s = [r["selected_s"] for r in crows]
            sadd = [r["selected_add"] for r in crows]
            if any(v is None for v in s + sadd):
                raise RuntimeError(f"undefined selection {dataset} {variant}")
            auc_s = float(np.mean([r["outer_auc"][m] for r, m in zip(crows, s)]))
            auc_add = float(np.mean([r["outer_auc"][m] for r, m in zip(crows, sadd)]))
            for fm in FM_MODELS:
                auc_fm = float(np.mean([r["outer_auc"][fm] for r in grows]))
                result_rows.append({"dataset": dataset, "variant": variant, "foundation_model": fm, "delta_sel": auc_fm - auc_s, "delta_add": auc_fm - auc_add, "S": json.dumps(s, separators=(",", ":")), "S_add": json.dumps(sadd, separators=(",", ":"))})
    results = pd.DataFrame(result_rows)
    results.to_csv(out / "pilot_results.csv", index=False)
    c_rows = []
    for dataset in [str(s["dataset"]) for s in pilot_specs()]:
        for fm in FM_MODELS:
            sel_diffs, add_diffs = [], []
            for lev in LEVELS:
                a = results[(results.dataset == dataset) & (results.foundation_model == fm) & (results.variant == f"conc_{lev}_r0")].iloc[0]
                b = results[(results.dataset == dataset) & (results.foundation_model == fm) & (results.variant == f"dispn_{lev}_r0")].iloc[0]
                sel_diffs.append(float(a.delta_sel - b.delta_sel))
                add_diffs.append(float(a.delta_add - b.delta_add))
            c_rows.append({"dataset": dataset, "foundation_model": fm, "C_i": float(np.mean(sel_diffs)), "C_i_add": float(np.mean(add_diffs))})
    pd.DataFrame(c_rows).to_csv(out / "pilot_C.csv", index=False)
    print(f"aggregate complete timing_rows={len(timing_df)} result_rows={len(results)} C_rows={len(c_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["cpu", "gpu", "aggregate"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "cpu":
        run_cpu(root, args.workers)
    elif args.action == "gpu":
        run_gpu(root)
    else:
        aggregate(root)


if __name__ == "__main__":
    main()
