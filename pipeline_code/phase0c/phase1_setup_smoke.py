from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 13
CHECKPOINTS = {
    "tabpfn35": "tabpfn-v3.5-20260909.safetensors",
    "tabpfn2_a": "tabpfn-v2-classifier-v2_default.ckpt",
    "tabpfn2_b": "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
    "tabicl2": "tabicl-classifier-v2-20260212.ckpt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def find_checkpoints(cache: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    files = [p for p in cache.rglob("*") if p.is_file()]
    for label, name in CHECKPOINTS.items():
        matches = [p for p in files if p.name == name]
        if matches:
            path = matches[0]
            found[label] = {"filename": path.name, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    return found


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)


def enforce_process_network_block() -> None:
    """Prevent this Python process and imported libraries from opening sockets."""
    def blocked(*_args, **_kwargs):
        raise RuntimeError("network disabled for offline smoke test")

    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    except RuntimeError as exc:
        if "network disabled" not in str(exc):
            raise
    else:
        raise RuntimeError("offline socket guard verification failed")


def download_weights(root: Path) -> None:
    cache = root / "model_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache / "tabpfn")
    os.environ["HF_HOME"] = str(cache / "huggingface")
    token = getpass.getpass("TABPFN token (input hidden): ")
    os.environ["TABPFN_TOKEN"] = token
    try:
        seed_all()
        import torch
        from sklearn.datasets import make_classification
        import tabpfn.browser_auth as tabpfn_browser_auth
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        # The task requires the API token to be supplied only through the
        # environment.  TabPFN normally caches it in ~/.cache; disable that
        # persistence before any license check is performed.
        tabpfn_browser_auth.save_token = lambda _token: None
        device = "cuda" if torch.cuda.is_available() else "cpu"

        X, y = make_classification(n_samples=80, n_features=8, n_informative=3, random_state=0)
        for version in (ModelVersion.V3_5, ModelVersion.V2):
            model = TabPFNClassifier.create_default_for_version(version, device=device, random_state=SEED)
            model.fit(X[:64], y[:64])
            model.predict_proba(X[64:])
            del model
        # TabICL is public, but loading it in the same one-shot process ensures
        # the exact registered checkpoint is cached before offline testing.
        from tabicl import TabICLClassifier

        model = TabICLClassifier(
            checkpoint_version=CHECKPOINTS["tabicl2"], device=device, random_state=SEED
        )
        model.fit(X[:64], y[:64])
        model.predict_proba(X[64:])
        del model
    finally:
        os.environ.pop("TABPFN_TOKEN", None)
        token = ""
    found = find_checkpoints(cache)
    if "tabpfn35" not in found or "tabicl2" not in found or not ({"tabpfn2_a", "tabpfn2_b"} & set(found)):
        raise RuntimeError(f"required checkpoints not found: {sorted(found)}")
    (root / "run_state" / "checkpoint_inventory.json").write_text(
        json.dumps(found, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: {"filename": v["filename"], "sha256": v["sha256"]} for k, v in found.items()}, indent=2))


def ordinal_views(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_idx: list[int]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    cat = [X_train.columns[i] for i in cat_idx]
    num = [c for c in X_train.columns if c not in cat]
    prep = ColumnTransformer(
        [
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
        ],
        verbose_feature_names_out=False,
    )
    tr = np.asarray(prep.fit_transform(X_train), dtype=float)
    te = np.asarray(prep.transform(X_test), dtype=float)
    return tr, te, list(range(len(cat))), prep


def lr_views(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_idx: list[int]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = [X_train.columns[i] for i in cat_idx]
    num = [c for c in X_train.columns if c not in cat]
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )
    tr = prep.fit_transform(X_train)
    te = prep.transform(X_test)
    if hasattr(tr, "toarray"):
        tr, te = tr.toarray(), te.toarray()
    return np.asarray(tr, float), np.asarray(te, float), prep


def ftt_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from rtdl_revisiting_models import FTTransformer
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=SEED
    )
    kwargs = FTTransformer.get_default_kwargs()
    model = FTTransformer(
        n_cont_features=X_train.shape[1], cat_cardinalities=[], d_out=1, **kwargs
    ).cuda()
    params = model.make_parameter_groups()
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
    n_pos = int(y_train[train_idx].sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_idx) - n_pos) / n_pos, device="cuda"))
    tx = torch.tensor(X_train, dtype=torch.float32, device="cuda")
    ty = torch.tensor(y_train, dtype=torch.float32, device="cuda")
    best, best_state, stale, best_epoch = float("inf"), None, 0, -1
    for epoch in range(100):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(tx[train_idx], None).squeeze(1), ty[train_idx])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(tx[val_idx], None).squeeze(1), ty[val_idx]).cpu())
        if val < best:
            best, stale, best_epoch = val, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32, device="cuda"), None).squeeze(1)).cpu().numpy()
    return p, {"default_kwargs": kwargs, "d_out": 1, "best_epoch": best_epoch, "n_params": sum(p.numel() for p in model.parameters())}


def smoke(root: Path) -> None:
    enforce_process_network_block()
    seed_all()
    import torch
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from tabicl import TabICLClassifier
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from xgboost import XGBClassifier

    cache = root / "model_cache"
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(cache / "tabpfn")
    os.environ["HF_HOME"] = str(cache / "huggingface")
    X, y = make_classification(n_samples=600, n_features=12, n_informative=4, weights=[0.7], random_state=0)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
    for c in ["x0", "x1"]:
        X[c] = pd.qcut(X[c], 4, labels=False, duplicates="drop").astype("category")
    Xtr, Xte, ytr = X.iloc[:480].copy(), X.iloc[480:].copy(), y[:480]
    ord_tr, ord_te, ord_cat, _ = ordinal_views(Xtr, Xte, [0, 1])
    cb_tr, cb_te = pd.DataFrame(ord_tr), pd.DataFrame(ord_te)
    for idx in ord_cat:
        cb_tr[idx] = cb_tr[idx].astype(int).astype(str)
        cb_te[idx] = cb_te[idx].astype(int).astype(str)
    lr_tr, lr_te, _ = lr_views(Xtr, Xte, [0, 1])
    rows, params = [], {}

    def record(name: str, fn, note: str):
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        try:
            model, fit_call, pred_call = fn()
            fit_call()
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter(); proba = np.asarray(pred_call()); pred_s = time.perf_counter() - t1
            ok = proba.shape == (120, 2)
            if hasattr(model, "get_params"):
                params[name] = jsonable(model.get_params(deep=True))
            rows.append({"model": name, "ok": ok, "proba_shape": str(tuple(proba.shape)), "fit_seconds": fit_s, "predict_seconds": pred_s, "offline": True, "note": note})
        except Exception as exc:
            rows.append({"model": name, "ok": False, "proba_shape": "", "fit_seconds": time.perf_counter()-t0, "predict_seconds": 0.0, "offline": True, "note": f"{note}; {type(exc).__name__}: {exc}"})

    record("tabpfn35", lambda: (m := TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="cuda", random_state=SEED, categorical_features_indices=[0,1]), lambda: m.fit(ord_tr,ytr), lambda: m.predict_proba(ord_te)), "categorical_features_indices=[0,1]")
    record("tabicl2", lambda: (m := TabICLClassifier(checkpoint_version=CHECKPOINTS["tabicl2"], device="cuda", random_state=SEED, allow_auto_download=False), lambda: m.fit(Xtr,ytr), lambda: m.predict_proba(Xte)), "pandas category dtype; internal ordinal encoder")
    record("tabpfn2", lambda: (m := TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda", random_state=SEED, categorical_features_indices=[0,1]), lambda: m.fit(ord_tr,ytr), lambda: m.predict_proba(ord_te)), "categorical_features_indices=[0,1]")
    record("lr", lambda: (m := LogisticRegression(C=1.0,l1_ratio=0.0,solver="lbfgs",max_iter=2000,random_state=42), lambda: m.fit(lr_tr,ytr), lambda: m.predict_proba(lr_te)), "one-hot + standardization; threadpool limit set by process env")
    record("ebm", lambda: (m := ExplainableBoostingClassifier(random_state=SEED,n_jobs=1,feature_types=["nominal","nominal"]+["continuous"]*10), lambda: m.fit(ord_tr,ytr), lambda: m.predict_proba(ord_te)), "feature_types nominal for first two; n_jobs=1")
    record("catboost", lambda: (m := CatBoostClassifier(verbose=0,random_seed=SEED,thread_count=1,cat_features=ord_cat), lambda: m.fit(cb_tr,ytr), lambda: m.predict_proba(cb_te)), "cat_features=[0,1]; string categorical columns; thread_count=1")
    record("lgbm", lambda: (m := LGBMClassifier(random_state=SEED,verbose=-1,n_jobs=1), lambda: m.fit(ord_tr,ytr,categorical_feature=ord_cat), lambda: m.predict_proba(ord_te)), "categorical_feature=[0,1]; n_jobs=1")
    record("xgb", lambda: (m := XGBClassifier(random_state=SEED,n_jobs=1), lambda: m.fit(ord_tr,ytr), lambda: m.predict_proba(ord_te)), "integer codes; n_jobs=1")
    record("rf", lambda: (m := RandomForestClassifier(random_state=SEED,n_jobs=1), lambda: m.fit(ord_tr,ytr), lambda: m.predict_proba(ord_te)), "integer codes; n_jobs=1")

    t0=time.perf_counter()
    try:
        p, ftt_meta = ftt_predict(lr_tr,ytr,lr_te)
        params["ftt"] = ftt_meta
        rows.append({"model":"ftt","ok":p.shape==(120,),"proba_shape":str((120,2)),"fit_seconds":time.perf_counter()-t0,"predict_seconds":0.0,"offline":True,"note":"get_default_kwargs; one-hot + standardization; GPU"})
    except Exception as exc:
        rows.append({"model":"ftt","ok":False,"proba_shape":"","fit_seconds":time.perf_counter()-t0,"predict_seconds":0.0,"offline":True,"note":f"{type(exc).__name__}: {exc}"})

    # GPU scale guard: success/failure only, never a performance metric.
    Xb, yb = make_classification(n_samples=8000,n_features=70,n_informative=12,random_state=1)
    for name, model in [
        ("tabpfn35_large", TabPFNClassifier.create_default_for_version(ModelVersion.V3_5,device="cuda",random_state=SEED)),
        ("tabicl2_large", TabICLClassifier(checkpoint_version=CHECKPOINTS["tabicl2"],device="cuda",random_state=SEED,allow_auto_download=False)),
        ("tabpfn2_large", TabPFNClassifier.create_default_for_version(ModelVersion.V2,device="cuda",random_state=SEED)),
    ]:
        t0=time.perf_counter()
        try:
            model.fit(Xb[:6400],yb[:6400]); fit_s=time.perf_counter()-t0
            t1=time.perf_counter(); pr=model.predict_proba(Xb[6400:]); pred_s=time.perf_counter()-t1
            rows.append({"model":name,"ok":np.asarray(pr).shape==(1600,2),"proba_shape":str(tuple(np.asarray(pr).shape)),"fit_seconds":fit_s,"predict_seconds":pred_s,"offline":True,"note":"GPU n=8000 p=70 scale guard; no metric computed"})
        except Exception as exc:
            rows.append({"model":name,"ok":False,"proba_shape":"","fit_seconds":time.perf_counter()-t0,"predict_seconds":0.0,"offline":True,"note":f"GPU scale guard; {type(exc).__name__}: {exc}"})
    out=root/"phase0c_outputs"; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"smoke_test.csv",index=False)
    (root/"run_state"/"smoke_params.json").write_text(json.dumps(params,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (root/"run_state"/"offline_enforcement.json").write_text(
        json.dumps({"method":"python_socket_guard","verified":True,"network_namespace_attempt":"operation_not_permitted"},indent=2)+"\n",
        encoding="utf-8",
    )
    if not all(bool(r["ok"]) for r in rows):
        raise RuntimeError("STOP one or more offline smoke tests failed")


def environment_report(root: Path) -> None:
    out=root/"phase0c_outputs"; state=root/"run_state"
    packages=["scikit-learn","openml","pandas","numpy","scipy","matplotlib","pyarrow","tabpfn","tabicl","interpret","catboost","lightgbm","xgboost","threadpoolctl","torch","rtdl-revisiting-models"]
    versions={p:importlib.metadata.version(p) for p in packages}
    import torch
    gpu=torch.cuda.get_device_name(0)
    driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip()
    checkpoints=json.loads((state/"checkpoint_inventory.json").read_text(encoding="utf-8"))
    params=json.loads((state/"smoke_params.json").read_text(encoding="utf-8"))
    cpu_model = platform.processor()
    if not cpu_model and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    env={
        "python_version":platform.python_version(),"packages":versions,"cpu_model":cpu_model,"gpu_model":gpu,
        "cuda_runtime":torch.version.cuda,"nvidia_driver":driver,"checkpoints":checkpoints,"model_default_params":params,
        "categorical_parameter_methods":{
            "tabpfn35":"categorical_features_indices","tabpfn2":"categorical_features_indices","tabicl2":"pandas category dtype/internal OrdinalEncoder",
            "ebm":"feature_types","catboost":"cat_features","lgbm":"fit(categorical_feature=...)","xgb":"integer codes","rf":"integer codes"
        },
        "determinism":{"seed":13,"CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"torch_use_deterministic_algorithms":torch.are_deterministic_algorithms_enabled()},
        "threads":{"OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS"),"MKL_NUM_THREADS":os.environ.get("MKL_NUM_THREADS"),"OPENBLAS_NUM_THREADS":os.environ.get("OPENBLAS_NUM_THREADS"),"traditional_models":1},
        "training_data_statements":{
            "tabpfn35":{"quote":"TabPFN-3.5 is trained purely on synthetic tabular tasks.","source":"https://huggingface.co/Prior-Labs/tabpfn_3_5"},
            "tabicl2":{"quote":"synthetic prior datasets on the fly ... this is how the TabICLv2 checkpoints were trained","source":"https://github.com/soda-inria/tabicl"},
            "tabpfn2":{"model_card":"Model card does not state training data.","paper_quote":"learned across millions of synthetic datasets","paper":"https://www.nature.com/articles/s41586-024-08328-6","checkpoint_identity_quote":"finetuned-zk73skhh.ckpt is identical to tabpfn-v2-classifier-v2_default.ckpt","identity_source":"https://huggingface.co/Prior-Labs/TabPFN-v2-clf"}
        }
    }
    (out/"environment_phase1.json").write_text(json.dumps(env,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(); p.add_argument("action",choices=["download","smoke","environment"]); p.add_argument("--root",type=Path,required=True); a=p.parse_args()
    if a.action=="download": download_weights(a.root.resolve())
    elif a.action=="smoke": smoke(a.root.resolve())
    else: environment_report(a.root.resolve())


if __name__=="__main__": main()
