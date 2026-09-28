from __future__ import annotations

from . import config

import gc
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .runtime import RegisteredCheckpoint, offline_socket_guard, set_offline_environment


class WeightAttestationFailure(RuntimeError):
    """A local foundation-weight load or loaded-parameter proof failed."""


def foundation_constructor_kwargs(
    model_name: str,
    checkpoints: dict[str, RegisteredCheckpoint],
    categorical_positions: list[int],
) -> dict[str, Any]:
    checkpoint = checkpoints[model_name]
    common: dict[str, Any] = {
        "device": "cuda",
        "random_state": config.SEED,
        "model_path": str(checkpoint.path),
    }
    if model_name == "tabicl2":
        return {
            **common,
            "checkpoint_version": checkpoint.path.name,
            "allow_auto_download": False,
        }
    return {**common, "categorical_features_indices": categorical_positions}


@contextmanager
def gpu_cleanup_guard(torch_module: Any):
    try:
        yield
    finally:
        gc.collect()
        torch_module.cuda.empty_cache()


def _loaded_state_objects(root: Any) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    seen: set[int] = set()

    def visit(value: Any, path: str, depth: int) -> None:
        if value is None or id(value) in seen or depth > 6:
            return
        seen.add(id(value))
        state_method = getattr(value, "state_dict", None)
        if callable(state_method):
            try:
                state = state_method()
            except Exception:
                state = None
            if isinstance(state, dict) and state and all(
                hasattr(tensor, "detach") and hasattr(tensor, "cpu")
                for tensor in state.values()
            ):
                found.append((path, state))
                return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], f"{path}[{key!s}]", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
        elif hasattr(value, "__dict__"):
            for name in sorted(vars(value)):
                if name.startswith("__"):
                    continue
                visit(vars(value)[name], f"{path}.{name}", depth + 1)

    visit(root, "estimator", 0)
    return found


def loaded_parameter_sha256(estimator: Any) -> str:
    """Hash the tensor state actually resident in a fitted estimator."""
    states = _loaded_state_objects(estimator)
    if not states:
        raise RuntimeError("estimator exposes no loaded torch parameters for attestation")
    state_digests: list[bytes] = []
    for _object_path, state in states:
        state_digest = hashlib.sha256()
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            array = tensor.numpy()
            state_digest.update(name.encode("utf-8"))
            state_digest.update(str(array.dtype).encode("ascii"))
            state_digest.update(json.dumps(list(array.shape)).encode("ascii"))
            state_digest.update(array.tobytes(order="C"))
        state_digests.append(state_digest.digest())
    digest = hashlib.sha256()
    for state_digest in sorted(state_digests):
        digest.update(state_digest)
    return digest.hexdigest()


_ATTESTED_PARAMETER_HASHES: dict[tuple[str, str], str] = {}


def attest_loaded_parameters(
    model_name: str, estimator: Any, checkpoint: RegisteredCheckpoint,
) -> str:
    try:
        states = _loaded_state_objects(estimator)
        if not states:
            raise RuntimeError("estimator exposes no loaded torch parameters for attestation")
        key = (model_name, checkpoint.sha256)
        expected_path = Path(__file__).with_name("loaded_parameter_sha256.json")
        records = json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.is_file() else {}
        record = records.get(model_name)
        observed = _ATTESTED_PARAMETER_HASHES.get(key)
        if observed is None:
            observed = loaded_parameter_sha256(estimator)
        if not isinstance(record, dict):
            raise RuntimeError(
                f"missing registered loaded parameter attestation for {model_name}; observed={observed}"
            )
        if record.get("weight_sha256") != checkpoint.sha256:
            raise RuntimeError(f"loaded parameter attestation weight mismatch for {model_name}")
        if record.get("parameter_sha256") != observed:
            raise RuntimeError(
                f"loaded parameter hash mismatch for {model_name}: observed={observed}"
            )
        _ATTESTED_PARAMETER_HASHES[key] = observed
        return observed
    except WeightAttestationFailure:
        raise
    except Exception as exc:
        raise WeightAttestationFailure(
            f"{model_name} loaded-parameter attestation failed: {type(exc).__name__}: {exc}"
        ) from exc


def _construct_foundation_model(
    model_name: str,
    checkpoints: dict[str, RegisteredCheckpoint],
    categorical_positions: list[int],
) -> Any:
    try:
        from tabicl import TabICLClassifier
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if model_name == "tabicl2":
            model = TabICLClassifier(**foundation_constructor_kwargs(
                model_name, checkpoints, categorical_positions
            ))
            model._load_model()
        else:
            version = ModelVersion.V3_5 if model_name == "tabpfn35" else ModelVersion.V2
            model = TabPFNClassifier.create_default_for_version(
                version,
                **foundation_constructor_kwargs(
                    model_name, checkpoints, categorical_positions
                ),
            )
            model._initialize_model_variables()
        return model
    except WeightAttestationFailure:
        raise
    except Exception as exc:
        raise WeightAttestationFailure(
            f"{model_name} local weight loading failed: {type(exc).__name__}: {exc}"
        ) from exc


def load_only_parameter_hashes(
    checkpoints: dict[str, RegisteredCheckpoint], input_frame: Any,
) -> tuple[dict[str, str], dict[str, int]]:
    """Load and attest local weights without fitting; data supplies metadata only."""
    try:
        shape = getattr(input_frame, "shape")
        signature = {"rows": int(shape[0]), "columns": int(shape[1])}
        hashes: dict[str, str] = {}
        for model_name in config.FOUNDATION_MODELS:
            model = _construct_foundation_model(model_name, checkpoints, [])
            hashes[model_name] = attest_loaded_parameters(
                model_name, model, checkpoints[model_name]
            )
            del model
        return hashes, signature
    except WeightAttestationFailure:
        raise
    except Exception as exc:
        raise WeightAttestationFailure(
            f"load-only parameter hash check failed: {type(exc).__name__}: {exc}"
        ) from exc


def epoch_batches(n_rows: int, epoch: int, batch_size: int = 256) -> list[np.ndarray]:
    if n_rows <= 0 or batch_size <= 0:
        raise ValueError("n_rows and batch_size must be positive")
    permutation = np.random.default_rng(config.SEED + int(epoch)).permutation(n_rows)
    return [permutation[start:start + batch_size] for start in range(0, n_rows, batch_size)]


def forward_slices(n_rows: int, batch_size: int = 1024) -> list[tuple[int, int]]:
    if n_rows < 0 or batch_size <= 0:
        raise ValueError("n_rows must be non-negative and batch_size positive")
    return [
        (start, min(start + batch_size, n_rows))
        for start in range(0, n_rows, batch_size)
    ]


def sigmoid_float64(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _forward_tensor_batched(model: Any, tensor: Any, positions: np.ndarray | None = None,
                            batch_size: int = 1024) -> Any:
    import torch
    selected = np.arange(len(tensor)) if positions is None else np.asarray(positions, dtype=int)
    chunks = []
    for start, stop in forward_slices(len(selected), batch_size):
        rows = selected[start:stop]
        chunks.append(model(tensor[rows], None).squeeze(1))
    return torch.cat(chunks) if chunks else torch.empty(0, device=tensor.device)


def _foundation_fit_predict_impl(model_name: str, train: pd.DataFrame, labels: np.ndarray,
                                 test: pd.DataFrame, categorical: list[str],
                                 checkpoints: dict[str, RegisteredCheckpoint],
                                 input_audit: Callable[[Any], None] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    set_offline_environment()
    with offline_socket_guard():
        import torch
        from .data import prepare_ordinal_views, prepare_tabicl_views

        determinism = config.gpu_determinism_record()
        if model_name == "tabicl2":
            prepared = prepare_tabicl_views(train, test, categorical)
            train_view, test_view = prepared.train, prepared.test
            categorical_positions = prepared.categorical_positions
            model = _construct_foundation_model(
                model_name, checkpoints, categorical_positions
            )
            categorical_parameter = "pandas category dtype"
        else:
            prepared = prepare_ordinal_views(train, test, categorical)
            train_view, test_view = prepared.train, prepared.test
            model = _construct_foundation_model(
                model_name, checkpoints, prepared.categorical_positions
            )
            categorical_positions = prepared.categorical_positions
            categorical_parameter = f"categorical_features_indices={categorical_positions}"
        if input_audit is not None:
            input_audit(train_view)
            input_audit(test_view)
        parameter_sha256 = attest_loaded_parameters(
            model_name, model, checkpoints[model_name]
        )
        try:
            model.fit(train_view, labels)
        except (FileNotFoundError, EOFError) as exc:
            raise WeightAttestationFailure(
                f"{model_name} local weight loading failed during fit: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}".lower()
            if any(marker in text for marker in (
                "checkpoint", "state_dict", "weight file", "model_path", "safetensor",
            )):
                raise WeightAttestationFailure(
                    f"{model_name} local weight loading failed during fit: {type(exc).__name__}: {exc}"
                ) from exc
            raise
        output = np.asarray(model.predict_proba(test_view), dtype=float)[:, 1]
    return output, {
        "categorical_positions": categorical_positions,
        "categorical_parameter": categorical_parameter,
        "determinism": determinism,
        "loaded_parameter_sha256": parameter_sha256,
        "unseen_category_handling": (
            "OrdinalEncoder unknown_value=-1; pandas category level -1"
            if model_name == "tabicl2"
            else "OrdinalEncoder unknown_value=-1; categorical_features_indices"
        ),
    }


def foundation_fit_predict(model_name: str, train: pd.DataFrame, labels: np.ndarray,
                           test: pd.DataFrame, categorical: list[str],
                           checkpoints: dict[str, RegisteredCheckpoint],
                           input_audit: Callable[[Any], None] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    with gpu_cleanup_guard(torch):
        return _foundation_fit_predict_impl(
            model_name, train, labels, test, categorical, checkpoints, input_audit
        )


def _ftt_fit_predict_impl(train: pd.DataFrame, labels: np.ndarray, test: pd.DataFrame,
                          categorical: list[str],
                          input_audit: Callable[[Any], None] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from rtdl_revisiting_models import FTTransformer

    from .data import prepare_ftt_views

    determinism = config.gpu_determinism_record()
    prepared = prepare_ftt_views(train, test, categorical)
    if input_audit is not None:
        input_audit(prepared.train)
        input_audit(prepared.test)
    train_position, validation_position = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=config.SEED,
    )
    kwargs = FTTransformer.get_default_kwargs()
    model = FTTransformer(
        n_cont_features=prepared.train.shape[1], cat_cardinalities=[], d_out=1, **kwargs,
    ).cuda()
    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
    train_labels = labels[train_position]
    positive = int(train_labels.sum())
    if positive == 0 or positive == len(train_labels):
        raise RuntimeError("FT-Transformer inner training split has one class")
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(train_labels) - positive) / positive, device="cuda")
    )
    x_tensor = torch.tensor(prepared.train, dtype=torch.float32, device="cuda")
    y_tensor = torch.tensor(labels, dtype=torch.float32, device="cuda")
    best_loss, best_state, best_epoch, stale = math.inf, None, -1, 0
    for epoch in range(100):
        model.train()
        for relative_batch in epoch_batches(len(train_position), epoch):
            rows = train_position[relative_batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_tensor[rows], None).squeeze(1), y_tensor[rows])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(
                _forward_tensor_batched(model, x_tensor, validation_position, 1024),
                y_tensor[validation_position],
            ).cpu())
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("FT-Transformer produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(prepared.test, dtype=torch.float32, device="cuda")
        logits = _forward_tensor_batched(model, test_tensor, None, 1024).cpu().numpy()
        output = sigmoid_float64(logits)
    n_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    return np.asarray(output, dtype=float), {
        "categorical_positions": [],
        "categorical_parameter": "cat_cardinalities=[]; LR-style one-hot+standardization",
        "determinism": determinism,
        "best_epoch": best_epoch,
        "n_params": n_parameters,
        "unseen_category_handling": "OneHotEncoder(handle_unknown=ignore)",
    }


def ftt_fit_predict(train: pd.DataFrame, labels: np.ndarray, test: pd.DataFrame,
                    categorical: list[str],
                    input_audit: Callable[[Any], None] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    with gpu_cleanup_guard(torch):
        return _ftt_fit_predict_impl(train, labels, test, categorical, input_audit)
