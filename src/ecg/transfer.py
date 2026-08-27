"""MIT-BIH → PTB transfer learning.

The claim in Kachuee et al. is that a representation learned on the 109k
arrhythmia beats transfers to myocardial-infarction detection on the 14.5k PTB
beats. This module tests that claim instead of assuming it, by running three arms
on the identical PTB splits and comparing them:

===========  =============================================================
``scratch``  the same architecture, randomly initialised, trained on PTB only
``frozen``   MIT-BIH weights, convolutional stack frozen, new head trained
``finetune`` MIT-BIH weights, everything trainable at a lower learning rate
===========  =============================================================

``scratch`` is the arm that makes the experiment honest: without it, a good
``finetune`` score says nothing about transfer, only that the architecture suits
the task.

The EDA gives the mechanistic reason to expect transfer to work at all: both
collections were segmented, resampled to 125 Hz and min-max normalised the same
way, and their aggregate statistics land within 6 samples of mean beat length and
0.004 of mean amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecg import metrics, training
from ecg.config import Config
from ecg.models import ResidualCNN, count_parameters, describe
from ecg.training import TrainingSettings, TrainingResult

#: The three arms, in report order.
ARMS: tuple = ("scratch", "frozen", "finetune")

#: How each arm is described in the report.
ARM_DESCRIPTIONS: Dict[str, str] = {
    "scratch": "Random initialisation, trained on PTB only",
    "frozen": "MIT-BIH backbone frozen, new classifier head trained",
    "finetune": "MIT-BIH weights, all layers trainable at a lower learning rate",
}


@dataclass
class ArmResult:
    """One transfer arm: its model, training history and scores."""

    arm: str
    model: ResidualCNN
    training: TrainingResult
    evaluations: Dict[str, Dict[str, object]]
    trainable_parameters: int

    @property
    def test_summary(self) -> Dict[str, object]:
        """The test-split summary dictionary, tagged with the arm name."""
        summary = dict(self.evaluations["test"]["summary"])  # type: ignore[index]
        summary["model"] = f"cnn-{self.arm}"
        summary["arm"] = self.arm
        summary["trainable_parameters"] = self.trainable_parameters
        summary["epochs"] = int(self.training.history["epoch"].max())
        summary["training_seconds"] = self.training.seconds
        return summary


def build_arm(
    arm: str,
    n_classes: int = 2,
    source_checkpoint: Optional[Path | str] = None,
    hidden_dim: int = 32,
    device: str = "cpu",
) -> ResidualCNN:
    """Construct the model for one arm.

    Args:
        arm: ``"scratch"``, ``"frozen"`` or ``"finetune"``.
        n_classes: Classes of the target task.
        source_checkpoint: MIT-BIH weights; required for the transfer arms.
        hidden_dim: Width of the new classifier's first layer.
        device: Device to map the checkpoint onto.

    Returns:
        A model ready to train.

    Raises:
        ValueError: For an unknown arm, or a transfer arm without a checkpoint.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")

    if arm == "scratch":
        return ResidualCNN(n_classes=n_classes, hidden_dim=hidden_dim)

    if source_checkpoint is None:
        raise ValueError(f"arm {arm!r} needs the MIT-BIH checkpoint to transfer from")

    # Rebuild the source model, load its weights, then swap the 5-class head for
    # a fresh n_classes one. Only the representation crosses over.
    model = ResidualCNN(n_classes=5, hidden_dim=hidden_dim)
    training.load_checkpoint(model, source_checkpoint, device=device)
    model.replace_head(n_classes=n_classes, hidden_dim=hidden_dim)
    model.freeze_backbone(arm == "frozen")
    return model


def run_arm(
    arm: str,
    loaders: Mapping[str, DataLoader],
    label_names: Sequence[str],
    cfg: Config,
    source_checkpoint: Optional[Path | str] = None,
    class_weights: Optional[torch.Tensor] = None,
    settings: Optional[TrainingSettings] = None,
    finetune_lr_factor: float = 0.1,
    mlflow_enabled: bool = True,
    verbose: bool = True,
    persist: bool = True,
    checkpoint_path: Optional[Path | str] = None,
) -> ArmResult:
    """Train and score a single transfer arm.

    Args:
        arm: Which arm to run.
        loaders: ``{"train", "val", "test"}`` DataLoaders over PTB.
        label_names: PTB class names.
        cfg: Project configuration.
        source_checkpoint: MIT-BIH weights, for the transfer arms.
        class_weights: Per-class loss weights.
        settings: Training hyper-parameters; defaults come from ``cfg``.
        finetune_lr_factor: Multiplier applied to the learning rate of the
            ``finetune`` arm. Fine-tuning a pretrained network at the pretraining
            learning rate erases the representation in the first few steps, which
            is the whole thing transfer is supposed to preserve.
        mlflow_enabled: Log to MLflow when it is available.
        verbose: Print progress.
        persist: Write the best weights to disk. The low-data sweep sets this to
            ``False``: it trains the same three arms dozens of times, and letting
            those runs share a file name would overwrite the checkpoints the
            full-size experiment produced.
        checkpoint_path: Explicit destination; defaults to
            ``<models_dir>/cnn_ptbdb_<arm>.pt``.

    Returns:
        An :class:`ArmResult`.
    """
    settings = settings or TrainingSettings.from_config(cfg)
    if arm == "finetune":
        settings = TrainingSettings(
            **{**settings.__dict__, "learning_rate": settings.learning_rate * finetune_lr_factor}
        )

    # Seed before construction: `train` seeds the loop, but the initial weights are
    # drawn here, so without this the "seed" would not control initialisation.
    training.set_seed(settings.seed)
    model = build_arm(arm, n_classes=len(label_names), source_checkpoint=source_checkpoint)
    trainable = count_parameters(model, trainable_only=True)

    if verbose:
        print(f"\n[{arm}] {ARM_DESCRIPTIONS[arm]}")
        print(
            f"  {trainable:,} trainable of {count_parameters(model):,} parameters"
            f"   lr {settings.learning_rate:g}"
        )

    checkpoint = None
    if persist:
        checkpoint = Path(checkpoint_path or Path(cfg.models_dir) / f"cnn_ptbdb_{arm}.pt")
    with training.mlflow_run(
        f"ptbdb-{arm}",
        cfg,
        enabled=mlflow_enabled,
        tags={"stage": "transfer", "arm": arm, "source": "ptbdb"},
    ) as run:
        if run is not None and training.MLFLOW_AVAILABLE:
            import mlflow

            mlflow.log_params({"arm": arm, **{f"model_{k}": v for k, v in describe(model).items()}})

        result = training.train(
            model,
            loaders,
            label_names,
            settings=settings,
            class_weights=class_weights,
            checkpoint_path=checkpoint,
            active_run=run,
            verbose=verbose,
        )

        evaluations = {
            split: training.evaluate_model(
                model, loaders[split], label_names, f"cnn-{arm}", split, settings.device
            )
            for split in ("val", "test")
            if split in loaders
        }
        training._log_metrics(
            run,
            {f"test_{k}": v for k, v in evaluations["test"]["summary"].items()  # type: ignore[index]
             if isinstance(v, (int, float))},
        )

    return ArmResult(
        arm=arm,
        model=model,
        training=result,
        evaluations=evaluations,
        trainable_parameters=trainable,
    )


def run_all_arms(
    loaders: Mapping[str, DataLoader],
    label_names: Sequence[str],
    cfg: Config,
    source_checkpoint: Path | str,
    class_weights: Optional[torch.Tensor] = None,
    arms: Sequence[str] = ARMS,
    settings: Optional[TrainingSettings] = None,
    mlflow_enabled: bool = True,
    verbose: bool = True,
) -> Dict[str, ArmResult]:
    """Run every arm and return them keyed by name."""
    return {
        arm: run_arm(
            arm,
            loaders,
            label_names,
            cfg,
            source_checkpoint=source_checkpoint,
            class_weights=class_weights,
            settings=settings,
            mlflow_enabled=mlflow_enabled,
            verbose=verbose,
        )
        for arm in arms
    }


def comparison(results: Mapping[str, ArmResult]) -> pd.DataFrame:
    """Stack the arms into one table, best macro-F1 first."""
    rows: List[Dict[str, object]] = []
    for arm, result in results.items():
        summary = result.test_summary
        summary["description"] = ARM_DESCRIPTIONS[arm]
        rows.append(summary)
    return metrics.comparison_table(rows)


def transfer_gain(results: Mapping[str, ArmResult], metric: str = "macro_f1") -> pd.DataFrame:
    """Quantify what transfer bought, relative to the from-scratch arm.

    Returns:
        Columns ``arm``, ``metric``, ``value``, ``delta_vs_scratch``,
        ``relative_error_reduction`` — the last one being the share of the
        remaining headroom that transfer closed.
    """
    if "scratch" not in results:
        raise KeyError("the 'scratch' arm is required as the reference")

    reference = float(results["scratch"].test_summary[metric])  # type: ignore[arg-type]
    rows = []
    for arm, result in results.items():
        value = float(result.test_summary[metric])  # type: ignore[arg-type]
        headroom = 1.0 - reference
        rows.append(
            {
                "arm": arm,
                "metric": metric,
                "value": round(value, 4),
                "delta_vs_scratch": round(value - reference, 4),
                "relative_error_reduction": (
                    round((value - reference) / headroom, 4) if headroom > 1e-9 else None
                ),
                "epochs": result.test_summary["epochs"],
                "trainable_parameters": result.trainable_parameters,
            }
        )
    return pd.DataFrame(rows)


def low_data_curve(
    arrays: Mapping[str, tuple],
    label_names: Sequence[str],
    cfg: Config,
    source_checkpoint: Path | str,
    fractions: Sequence[float] = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00),
    arms: Sequence[str] = ARMS,
    settings: Optional[TrainingSettings] = None,
    batch_size: int = 64,
    mlflow_enabled: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Measure every arm as the target training set shrinks.

    Transfer learning is a claim about the *low-data* regime: a pretrained
    representation should matter most when there is not enough target data to
    learn one. Comparing the arms only at full size answers a different, easier
    question. This sweeps the PTB training split down to a few hundred beats and
    scores each arm on the untouched test split at every size.

    The validation and test splits are held fixed at full size throughout, so the
    only thing changing is how much the model gets to learn from.

    Args:
        arrays: ``{split: (X, y)}`` from :func:`ecg.torch_data.load_arrays`.
        label_names: Target class names.
        cfg: Project configuration.
        source_checkpoint: MIT-BIH weights.
        fractions: Shares of the training split to try.
        arms: Arms to run at each size.
        settings: Training hyper-parameters.
        batch_size: Batch size; smaller than the default because the smallest
            subsets hold only a few hundred beats.
        mlflow_enabled: Log each cell to MLflow.
        verbose: Print progress.

    Returns:
        Columns ``fraction``, ``n_train``, ``arm``, ``macro_f1``,
        ``balanced_accuracy``, ``roc_auc``, ``epochs``.
    """
    from ecg.torch_data import class_weights as _class_weights
    from ecg.torch_data import make_loaders, stratified_subsample

    settings = settings or TrainingSettings.from_config(cfg)
    X_train, y_train = arrays["train"]
    rows: List[Dict[str, object]] = []

    for fraction in fractions:
        X_sub, y_sub = stratified_subsample(X_train, y_train, fraction, seed=cfg.seed)
        loaders = make_loaders(
            {"train": (X_sub, y_sub), "val": arrays["val"], "test": arrays["test"]},
            batch_size=batch_size,
            seed=cfg.seed,
        )
        weights = _class_weights(y_sub, len(label_names))
        if verbose:
            print(f"\n--- {fraction:.0%} of the PTB training split = {len(y_sub):,} beats ---",
                  flush=True)

        for arm in arms:
            result = run_arm(
                arm,
                loaders,
                label_names,
                cfg,
                source_checkpoint=source_checkpoint,
                class_weights=weights,
                settings=settings,
                mlflow_enabled=mlflow_enabled,
                verbose=False,
                persist=False,
            )
            summary = result.test_summary
            rows.append(
                {
                    "fraction": fraction,
                    "n_train": len(y_sub),
                    "arm": arm,
                    "macro_f1": summary["macro_f1"],
                    "balanced_accuracy": summary["balanced_accuracy"],
                    "roc_auc": summary.get("roc_auc"),
                    "epochs": summary["epochs"],
                }
            )
            if verbose:
                print(f"    {arm:<9} macro-F1 {summary['macro_f1']:.4f}", flush=True)

    return pd.DataFrame(rows)


def repeat_arms(
    loaders: Mapping[str, DataLoader],
    label_names: Sequence[str],
    cfg: Config,
    source_checkpoint: Path | str,
    seeds: Sequence[int],
    class_weights: Optional[torch.Tensor] = None,
    arms: Sequence[str] = ARMS,
    metric: str = "macro_f1",
    mlflow_enabled: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every arm once per seed, so the comparison has an error bar.

    A single run of this experiment is not enough to rank the arms. Convolution
    backward passes on CPU are not bit-deterministic across threads, and over
    dozens of epochs that compounds into different early-stopping points — two
    "identical" runs of the same arm can land a point or more apart. Repeating the
    arms across seeds turns "scratch beat fine-tune" into a claim with a spread
    attached, which is the difference between an observation and a result.

    No arm persists a checkpoint here: the single-seed run owns those artefacts,
    and this sweep is measurement.

    Args:
        loaders: PTB DataLoaders.
        label_names: Target class names.
        cfg: Project configuration.
        source_checkpoint: MIT-BIH weights.
        seeds: Seeds to repeat over.
        class_weights: Per-class loss weights.
        arms: Arms to run.
        metric: Metric recorded per run.
        mlflow_enabled: Log each run to MLflow.
        verbose: Print progress.

    Returns:
        Long-format columns ``seed``, ``arm``, ``macro_f1``, ``balanced_accuracy``,
        ``roc_auc``, ``epochs``.
    """
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        if verbose:
            print(f"\n=== seed {seed} ===", flush=True)
        settings = TrainingSettings.from_config(cfg)
        settings.seed = seed
        for arm in arms:
            result = run_arm(
                arm,
                loaders,
                label_names,
                cfg,
                source_checkpoint=source_checkpoint,
                class_weights=class_weights,
                settings=settings,
                mlflow_enabled=mlflow_enabled,
                verbose=False,
                persist=False,
            )
            summary = result.test_summary
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "macro_f1": summary["macro_f1"],
                    "balanced_accuracy": summary["balanced_accuracy"],
                    "roc_auc": summary.get("roc_auc"),
                    "epochs": summary["epochs"],
                }
            )
            if verbose:
                print(f"  {arm:<9} {metric} {summary[metric]:.4f}", flush=True)
    return pd.DataFrame(rows)


def seed_summary(runs: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Aggregate :func:`repeat_arms` into mean, spread and a delta vs ``scratch``."""
    agg = (
        runs.groupby("arm")[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "n_seeds"})
        .round(4)
        .reindex([arm for arm in ARMS if arm in set(runs["arm"])])
        .reset_index()
    )
    if "scratch" in set(agg["arm"]):
        reference = float(agg.loc[agg["arm"] == "scratch", "mean"].iloc[0])
        agg["delta_vs_scratch"] = (agg["mean"] - reference).round(4)
    agg["metric"] = metric
    return agg
