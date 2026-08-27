#!/usr/bin/env python
"""Fit the classical Spark ML baselines and write their report artefacts.

Usage::

    python scripts/train_baselines.py
    python scripts/train_baselines.py --source ptbdb
    python scripts/train_baselines.py --models logistic-regression random-forest gradient-boosted-trees
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

from ecg import baselines as bl
from ecg import eda, metrics, reporting, viz
from ecg.config import load_config
from ecg.features import MODEL_FEATURE_COLUMNS
from ecg.preprocessing import load_dataset
from ecg.session import get_spark, session_summary, stop_spark
from ecg.training import MLFLOW_AVAILABLE, mlflow_run


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["mitbih", "ptbdb"], default="mitbih")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--shuffle-partitions", type=int, default=None)
    parser.add_argument("--driver-memory", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=sorted(bl.BASELINES),
        help=f"Baselines to fit (default: {' '.join(bl.DEFAULT_BASELINES)})",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        shuffle_partitions=args.shuffle_partitions,
        driver_memory=args.driver_memory,
    ).ensure_dirs()

    source = args.source
    label_names = eda.ordered_labels(source)

    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session:", session_summary(spark))

    dataset = load_dataset(spark, cfg, source)
    print(f"{dataset.count():,} beats · {len(label_names)} classes\n")

    started = time.time()
    results = bl.run_baselines(
        dataset, label_names, names=args.models, seed=cfg.seed, splits=("val", "test")
    )

    table = bl.results_table(results, split="test")
    print("\n" + table.to_string(index=False))

    viz.apply_style()
    reporting.save_table(table, cfg, f"{source}_baselines_test")

    for name, entry in results.items():
        slug = f"{source}_{name}"
        bl.save_model(entry["model"], cfg, source, name)
        for split in ("val", "test"):
            reporting.save_evaluation(
                entry[split],
                cfg,
                f"{slug}_{split}",
                label_order=label_names,
                title=f"{name} ({source}, {split})",
                make_figures=split == "test",
            )

        with mlflow_run(
            slug,
            cfg,
            enabled=not args.no_mlflow,
            tags={"stage": "baseline", "source": source, "family": "spark-ml"},
        ) as run:
            if run is not None and MLFLOW_AVAILABLE:
                import mlflow

                mlflow.log_params({"model": name, **entry["spec"].params})
                mlflow.log_metrics(
                    {
                        f"test_{k}": float(v)
                        for k, v in entry["test"]["summary"].items()
                        if isinstance(v, (int, float))
                    }
                )
                mlflow.log_metric("fit_seconds", float(entry["fit_seconds"]))

        if hasattr(entry["model"], "featureImportances"):
            names = bl.assembled_feature_names(MODEL_FEATURE_COLUMNS)
            reporting.save_table(
                bl.feature_importances(entry["model"], names, top_n=25),
                cfg,
                f"{slug}_feature_importances",
            )

    # A trivial reference, so the comparison chart shows what accuracy hides.
    y_true = (
        bl.split_frames(dataset)["test"].select("label").toPandas()["label"].to_numpy()
    )
    trivial = metrics.majority_class_baseline(y_true, label_names)
    reporting.save_json(trivial["summary"], cfg, f"{source}_majority_baseline_summary")

    viz.save_figure(
        viz.plot_model_comparison(
            table,
            metric="macro_f1",
            title=f"Spark ML baselines — {source}",
            subtitle="Test split · class-weighted training · higher is better",
            reference=float(trivial["summary"]["macro_f1"]),
            reference_label="always predict the majority class",
        ),
        cfg.figures_dir / f"{source}_baselines_comparison.png",
    )

    print(f"\ncompleted in {(time.time() - started) / 60:.1f} min")
    stop_spark(spark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
