"""Training loop, checkpointing and experiment tracking for the PyTorch models.

The loop is deliberately small and explicit — weighted cross-entropy, Adam,
``ReduceLROnPlateau``, early stopping on validation **macro-F1** — because the
interesting decisions in this project are about the imbalance, not about training
tricks.

Selecting on macro-F1 rather than loss or accuracy matters more than it looks. On
MIT-BIH the validation loss keeps improving while the model gets better at the
90k normal beats and no better at the 97 fusion beats; macro-F1 stops that.

Experiment tracking goes to MLflow when it is installed (a local file store under
``<data_dir>/mlruns`` by default, or the workspace tracking server on Databricks)
and is silently skipped when it is not, so the module has no hard dependency on it.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg import metrics
from ecg.config import Config

try:  # MLflow is optional.
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


@dataclass
class TrainingSettings:
    """Hyper-parameters of one training run.

    Attributes:
        max_epochs: Upper bound on epochs; early stopping usually ends sooner.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        patience: Epochs without validation macro-F1 improvement before stopping.
        lr_patience: Epochs without improvement before the learning rate is halved.
        min_delta: Improvement below this counts as no improvement.
        grad_clip: Gradient-norm clip; ``0`` disables it.
        seed: Seed for python, numpy and torch.
        device: ``"cpu"`` or ``"cuda"``.
    """

    max_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 6
    lr_patience: int = 3
    min_delta: float = 1e-4
    grad_clip: float = 0.0
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_config(cls, cfg: Config, **overrides: Any) -> "TrainingSettings":
        """Build settings from the project configuration, with optional overrides."""
        base = {
            "max_epochs": cfg.max_epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "patience": cfg.patience,
            "lr_patience": cfg.lr_patience,
            "seed": cfg.seed,
        }
        base.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**base)


@dataclass
class TrainingResult:
    """Everything one training run produces.

    Attributes:
        history: Per-epoch metrics.
        best_epoch: Epoch with the highest validation macro-F1 (1-indexed).
        best_score: That macro-F1.
        seconds: Wall-clock training time.
        stopped_early: Whether early stopping triggered.
        settings: The hyper-parameters used.
    """

    history: pd.DataFrame
    best_epoch: int
    best_score: float
    seconds: float
    stopped_early: bool
    settings: TrainingSettings = field(default_factory=TrainingSettings)


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_threads(cfg: Config) -> None:
    """Pin PyTorch's thread count when the configuration asks for it."""
    if cfg.torch_threads and cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)


# ------------------------------------------------------------------- inference


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a model over a loader and return ``(y_true, y_pred, y_score)``.

    ``y_score`` holds softmax probabilities, so it feeds the ranking metrics in
    :mod:`ecg.metrics` directly.
    """
    model.eval().to(device)
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    for inputs, labels in loader:
        logits = model(inputs.to(device))
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())

    logits_array = np.concatenate(logits_all)
    y_true = np.concatenate(labels_all)
    y_score = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    return y_true, y_score.argmax(axis=1), y_score


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    label_names: Sequence[str],
    model_name: str,
    split: str,
    device: str = "cpu",
) -> Dict[str, object]:
    """Score a model on a loader with :func:`ecg.metrics.evaluate`."""
    y_true, y_pred, y_score = predict(model, loader, device)
    return metrics.evaluate(
        y_true, y_pred, label_names, y_score=y_score, model_name=model_name, split=split
    )


# ------------------------------------------------------------------- tracking


@contextlib.contextmanager
def mlflow_run(
    run_name: str,
    cfg: Optional[Config] = None,
    experiment: str = "ecg-heartbeat",
    enabled: bool = True,
    tags: Optional[Mapping[str, str]] = None,
):
    """Start an MLflow run, or do nothing when MLflow is absent or disabled.

    On Databricks the ambient tracking URI is left untouched, so runs land in the
    workspace experiment. Locally the backend is a SQLite database under
    ``<data_dir>/mlruns`` — MLflow 3 put the plain-file store into maintenance
    mode and refuses it by default — with artifacts beside it. Either way nothing
    is written inside the git working tree.
    """
    if not (enabled and MLFLOW_AVAILABLE):
        yield None
        return

    from ecg.session import is_databricks

    if cfg is not None and not is_databricks():
        store = Path(cfg.mlflow_dir).resolve()
        (store / "artifacts").mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(
                experiment, artifact_location=(store / "artifacts").as_uri()
            )
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(dict(tags))
        yield run


def _log_params(active_run, params: Mapping[str, Any]) -> None:
    if active_run is not None and MLFLOW_AVAILABLE:
        mlflow.log_params({key: str(value) for key, value in params.items()})


def _log_metrics(active_run, values: Mapping[str, float], step: Optional[int] = None) -> None:
    if active_run is not None and MLFLOW_AVAILABLE:
        mlflow.log_metrics(
            {key: float(value) for key, value in values.items() if value is not None}, step=step
        )


# -------------------------------------------------------------------- training


def train(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    label_names: Sequence[str],
    settings: Optional[TrainingSettings] = None,
    class_weights: Optional[torch.Tensor] = None,
    checkpoint_path: Optional[Path | str] = None,
    active_run: Any = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train a model, keeping the checkpoint with the best validation macro-F1.

    Args:
        model: Module to train, modified in place; on return it holds the best
            weights seen, not the last ones.
        loaders: ``{"train": ..., "val": ...}``; a ``"test"`` loader is ignored here.
        label_names: Class names, for the per-epoch metrics.
        settings: Hyper-parameters; defaults are used when omitted.
        class_weights: Per-class loss weights — the correction for the imbalance.
        checkpoint_path: Where to write the best weights, if anywhere.
        active_run: An open MLflow run to log into, or ``None``.
        verbose: Print one line per epoch.

    Returns:
        A :class:`TrainingResult`.

    Raises:
        KeyError: If ``loaders`` lacks a ``train`` or ``val`` entry.
    """
    if "train" not in loaders or "val" not in loaders:
        raise KeyError("loaders must contain a 'train' and a 'val' DataLoader")

    settings = settings or TrainingSettings()
    set_seed(settings.seed)
    device = settings.device
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=None if class_weights is None else class_weights.to(device)
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=settings.lr_patience
    )

    _log_params(active_run, asdict(settings))

    history: List[Dict[str, float]] = []
    best_score = -np.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    stopped_early = False
    started = time.time()

    for epoch in range(1, settings.max_epochs + 1):
        epoch_started = time.time()
        model.train()
        running_loss, n_seen = 0.0, 0

        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            if settings.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            n_seen += labels.size(0)

        train_loss = running_loss / max(n_seen, 1)
        y_true, y_pred, y_score = predict(model, loaders["val"], device)
        with torch.no_grad():
            val_loss = float(
                criterion(
                    torch.log(torch.from_numpy(np.clip(y_score, 1e-12, 1.0))),
                    torch.from_numpy(y_true),
                ).item()
            )
        scores = metrics.summary_metrics(y_true, y_pred, y_score, n_classes=len(label_names))

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_macro_f1": scores["macro_f1"],
            "val_balanced_accuracy": scores["balanced_accuracy"],
            "val_accuracy": scores["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_started, 1),
        }
        history.append(record)
        _log_metrics(active_run, {k: v for k, v in record.items() if k != "epoch"}, step=epoch)

        improved = scores["macro_f1"] > best_score + settings.min_delta
        if improved:
            best_score = scores["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"  epoch {epoch:>3}/{settings.max_epochs}  "
                f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                f"val_macro_F1 {scores['macro_f1']:.4f}"
                f"{'  *' if improved else ''}  ({record['seconds']:.0f}s)",
                flush=True,
            )

        scheduler.step(scores["macro_f1"])

        if epochs_without_improvement >= settings.patience:
            stopped_early = True
            if verbose:
                print(
                    f"  early stop: no macro-F1 gain in {settings.patience} epochs "
                    f"(best {best_score:.4f} at epoch {best_epoch})",
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_path is not None and best_state is not None:
        save_checkpoint(model, checkpoint_path, metadata={"best_epoch": best_epoch,
                                                          "best_val_macro_f1": best_score})

    result = TrainingResult(
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_score=float(best_score),
        seconds=round(time.time() - started, 1),
        stopped_early=stopped_early,
        settings=settings,
    )
    _log_metrics(
        active_run,
        {
            "best_epoch": result.best_epoch,
            "best_val_macro_f1": result.best_score,
            "training_seconds": result.seconds,
        },
    )
    return result


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(
    model: nn.Module,
    path: Path | str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Save weights plus a small JSON sidecar describing the run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    if metadata:
        destination.with_suffix(".json").write_text(
            json.dumps(dict(metadata), indent=2, default=str), encoding="utf-8"
        )
    return str(destination)


def load_checkpoint(model: nn.Module, path: Path | str, device: str = "cpu") -> nn.Module:
    """Load weights into ``model`` in place and return it."""
    state = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


# ------------------------------------------------------------ Databricks path


def run_distributed(
    train_fn,
    num_processes: int = 1,
    local_mode: bool = True,
    use_gpu: bool = False,
    **kwargs,
):
    """Run a training function through Spark's :class:`TorchDistributor`.

    This is the entry point for a Databricks GPU cluster: the same ``train_fn``
    that runs here single-process is handed to Spark, which launches it across
    workers with ``torch.distributed`` already wired up.

    It is *not* the fast path on a small machine — spawning a subprocess to use
    the two cores the parent already had costs more than it saves — so the scripts
    in this project call :func:`train` directly by default and expose this behind
    a flag.

    Args:
        train_fn: Callable executed on each worker.
        num_processes: Processes to launch.
        local_mode: ``True`` to run on the driver, ``False`` to use the cluster.
        use_gpu: Whether workers should claim a GPU.
        **kwargs: Forwarded to ``train_fn``.

    Returns:
        Whatever ``train_fn`` returns on rank 0.
    """
    from pyspark.ml.torch.distributor import TorchDistributor

    distributor = TorchDistributor(
        num_processes=num_processes, local_mode=local_mode, use_gpu=use_gpu
    )
    return distributor.run(train_fn, **kwargs)


def history_to_long(history: pd.DataFrame, keep: Iterable[str] = ()) -> pd.DataFrame:
    """Reshape a training history to long form for plotting."""
    columns = list(keep) or ["train_loss", "val_loss", "val_macro_f1"]
    return history.melt(
        id_vars="epoch", value_vars=columns, var_name="metric", value_name="value"
    )
