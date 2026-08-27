#!/usr/bin/env python
"""Train the residual 1-D CNN on one collection and write its report artefacts.

Usage::

    python scripts/train_cnn.py                       # MIT-BIH, defaults from conf/
    python scripts/train_cnn.py --source ptbdb --max-epochs 30
    python scripts/train_cnn.py --no-class-weights    # ablation
    python scripts/train_cnn.py --distributed         # via Spark TorchDistributor
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

from ecg import eda, reporting, viz
from ecg.config import load_config
from ecg.models import ResidualCNN, describe
from ecg.session import get_spark, stop_spark
from ecg.torch_data import class_weights, describe_arrays, load_arrays, make_loaders
from ecg.training import (
    MLFLOW_AVAILABLE,
    TrainingSettings,
    configure_threads,
    evaluate_model,
    load_checkpoint,
    mlflow_run,
    run_distributed,
    save_checkpoint,
    train,
)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["mitbih", "ptbdb"], default="mitbih")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Train with an unweighted loss, to measure what the weighting buys",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Launch through Spark's TorchDistributor instead of training in-process",
    )
    parser.add_argument("--tag", default=None, help="Suffix for the artefact names")
    return parser.parse_args(argv)


def distributed_entry(
    settings_dict: dict,
    config_dict: dict,
    source: str,
    checkpoint: str,
    weighted: bool,
    dropout: float,
) -> str:
    """Training body executed inside a ``TorchDistributor`` worker.

    Must be importable and picklable, so it takes plain dictionaries and rebuilds
    everything from the on-disk array cache rather than closing over live objects.

    Returns:
        The checkpoint path, which the parent process reloads to evaluate.
    """
    from ecg.config import Config as _Config
    from ecg.models import ResidualCNN as _CNN
    from ecg.torch_data import class_weights as _weights
    from ecg.torch_data import load_arrays as _load
    from ecg.torch_data import make_loaders as _loaders
    from ecg.training import TrainingSettings as _Settings
    from ecg.training import train as _train
    from ecg import eda as _eda

    cfg = _Config(**config_dict)
    label_names = _eda.ordered_labels(source)
    arrays = _load(None, cfg, source=source, use_cache=True)
    loaders = _loaders(arrays, batch_size=cfg.batch_size, seed=cfg.seed)
    model = _CNN(n_classes=len(label_names), dropout=dropout)
    _train(
        model,
        loaders,
        label_names,
        settings=_Settings(**settings_dict),
        class_weights=_weights(arrays["train"][1], len(label_names)) if weighted else None,
        checkpoint_path=checkpoint,
        verbose=True,
    )
    return checkpoint


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        torch_threads=args.threads,
    ).ensure_dirs()
    configure_threads(cfg)

    source = args.source
    label_names = eda.ordered_labels(source)
    slug = f"{source}_cnn" + (f"_{args.tag}" if args.tag else "")
    checkpoint = cfg.model_path(f"{slug}.pt")

    # Spark is needed only to materialise (and cache) the arrays; the training
    # loop itself is pure PyTorch over in-memory tensors.
    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("ERROR")
    arrays = load_arrays(spark, cfg, source=source)
    stop_spark(spark)

    print(describe_arrays(arrays, label_names).to_string(index=False), flush=True)
    loaders = make_loaders(arrays, batch_size=cfg.batch_size, seed=cfg.seed)

    weights = None if args.no_class_weights else class_weights(arrays["train"][1], len(label_names))
    if weights is not None:
        print("class weights:", {n: round(float(w), 3) for n, w in zip(label_names, weights)},
              flush=True)

    model = ResidualCNN(n_classes=len(label_names), dropout=args.dropout)
    print("model:", describe(model), flush=True)

    settings = TrainingSettings.from_config(cfg)
    started = time.time()

    with mlflow_run(
        slug,
        cfg,
        enabled=not args.no_mlflow,
        tags={"stage": "cnn", "source": source, "weighted": str(weights is not None)},
    ) as run:
        if run is not None and MLFLOW_AVAILABLE:
            import mlflow

            mlflow.log_params({f"model_{k}": v for k, v in describe(model).items()})
            mlflow.log_params(
                {"batch_size": cfg.batch_size, "class_weights": weights is not None,
                 "dropout": args.dropout}
            )

        if args.distributed:
            run_distributed(
                distributed_entry,
                settings_dict=settings.__dict__,
                config_dict=cfg.to_dict(),
                source=source,
                checkpoint=str(checkpoint),
                weighted=weights is not None,
                dropout=args.dropout,
            )
            load_checkpoint(model, checkpoint)
            result = None
        else:
            result = train(
                model,
                loaders,
                label_names,
                settings=settings,
                class_weights=weights,
                checkpoint_path=checkpoint,
                active_run=run,
                verbose=True,
            )

        evaluations = {
            split: evaluate_model(model, loaders[split], label_names, slug, split)
            for split in ("val", "test")
        }

    if result is not None:
        print(
            f"\ntrained in {result.seconds / 60:.1f} min "
            f"(best epoch {result.best_epoch}, val macro-F1 {result.best_score:.4f})",
            flush=True,
        )
    print("\ntest:", evaluations["test"]["summary"], flush=True)
    print()
    print(evaluations["test"]["per_class"].to_string(index=False), flush=True)

    viz.apply_style()
    if result is not None:
        reporting.save_history(
            result.history, cfg, slug, result.best_epoch,
            title=f"Residual CNN training — {source}",
        )
    for split, evaluation in evaluations.items():
        reporting.save_evaluation(
            evaluation,
            cfg,
            f"{slug}_{split}",
            label_order=label_names,
            title=f"residual CNN ({source}, {split})",
            # Figures only for the split that gets reported; the validation
            # tables are kept for the record without cluttering reports/figures.
            make_figures=split == "test",
        )

    save_checkpoint(
        model,
        checkpoint,
        metadata={
            "source": source,
            "label_names": list(label_names),
            "class_weights": weights is not None,
            **describe(model),
            **({"best_epoch": result.best_epoch,
                "best_val_macro_f1": result.best_score} if result else {}),
            "test_macro_f1": evaluations["test"]["summary"]["macro_f1"],
        },
    )
    print(f"\ncheckpoint: {checkpoint}")
    print(f"total wall clock: {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
