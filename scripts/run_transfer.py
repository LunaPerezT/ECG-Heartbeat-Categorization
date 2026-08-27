#!/usr/bin/env python
"""Run the MIT-BIH → PTB transfer experiment and write its report artefacts.

Three arms on identical PTB splits: from scratch, frozen backbone, full fine-tune.

Usage::

    python scripts/run_transfer.py
    python scripts/run_transfer.py --arms scratch frozen --max-epochs 40
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

from ecg import eda, reporting, transfer, viz
from ecg.config import load_config
from ecg.session import get_spark, stop_spark
from ecg.torch_data import class_weights, describe_arrays, load_arrays, make_loaders
from ecg.training import TrainingSettings, configure_threads


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="MIT-BIH weights to transfer from")
    parser.add_argument("--arms", nargs="*", default=list(transfer.ARMS), choices=list(transfer.ARMS))
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Repeat every arm across these seeds and report mean ± spread",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


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

    checkpoint = args.checkpoint or cfg.model_path("mitbih_cnn.pt")
    label_names = eda.ordered_labels("ptbdb")

    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("ERROR")
    arrays = load_arrays(spark, cfg, source="ptbdb")
    stop_spark(spark)

    print(describe_arrays(arrays, label_names).to_string(index=False), flush=True)
    loaders = make_loaders(arrays, batch_size=cfg.batch_size, seed=cfg.seed)
    weights = class_weights(arrays["train"][1], len(label_names))
    print("class weights:", {n: round(float(w), 3) for n, w in zip(label_names, weights)})

    started = time.time()

    results = transfer.run_all_arms(
        loaders,
        label_names,
        cfg,
        source_checkpoint=checkpoint,
        class_weights=weights,
        arms=args.arms,
        settings=TrainingSettings.from_config(cfg),
        mlflow_enabled=not args.no_mlflow,
    )

    table = transfer.comparison(results)
    print("\n" + table.to_string(index=False))

    viz.apply_style()
    reporting.save_table(table, cfg, "ptbdb_transfer_comparison")

    if "scratch" in results:
        gain = transfer.transfer_gain(results)
        print("\n" + gain.to_string(index=False))
        reporting.save_table(gain, cfg, "ptbdb_transfer_gain")

    for arm, result in results.items():
        slug = f"ptbdb_cnn_{arm}"
        reporting.save_history(
            result.training.history,
            cfg,
            slug,
            result.training.best_epoch,
            title=f"PTB transfer — {arm}",
        )
        for split, evaluation in result.evaluations.items():
            reporting.save_evaluation(
                evaluation,
                cfg,
                f"{slug}_{split}",
                label_order=label_names,
                title=f"{arm} (PTB, {split})",
                make_figures=split == "test",
            )

    viz.save_figure(
        viz.plot_model_comparison(
            table,
            metric="macro_f1",
            title="MIT-BIH → PTB transfer",
            subtitle="Test split · identical splits, architecture and budget across arms",
        ),
        cfg.figures_dir / "ptbdb_transfer_comparison.png",
    )

    if args.seeds:
        print("\n=== repeating every arm across seeds ===")
        runs = transfer.repeat_arms(
            loaders,
            label_names,
            cfg,
            source_checkpoint=checkpoint,
            seeds=args.seeds,
            class_weights=weights,
            arms=args.arms,
            mlflow_enabled=False,
        )
        seeds_summary = transfer.seed_summary(runs)
        print("\n" + runs.to_string(index=False))
        print("\n" + seeds_summary.to_string(index=False))
        reporting.save_table(runs, cfg, "ptbdb_transfer_seed_runs")
        reporting.save_table(seeds_summary, cfg, "ptbdb_transfer_seed_summary")

    print(f"\ncompleted in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
