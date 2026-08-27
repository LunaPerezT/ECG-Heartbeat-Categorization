#!/usr/bin/env python
"""Sweep the PTB training-set size to find where transfer actually pays off.

The full-size comparison in ``run_transfer.py`` answers "does pretraining help
when PTB already has 10k training beats?". This answers the question transfer
learning is actually about: "does it help when it does not?".

Usage::

    python scripts/run_low_data.py
    python scripts/run_low_data.py --fractions 0.02 0.1 1.0 --max-epochs 30
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
from ecg.torch_data import load_arrays
from ecg.training import TrainingSettings, configure_threads


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--fractions", nargs="*", type=float, default=[0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    )
    parser.add_argument("--arms", nargs="*", default=list(transfer.ARMS), choices=list(transfer.ARMS))
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        max_epochs=args.max_epochs,
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

    started = time.time()
    curve = transfer.low_data_curve(
        arrays,
        label_names,
        cfg,
        source_checkpoint=checkpoint,
        fractions=args.fractions,
        arms=args.arms,
        settings=TrainingSettings.from_config(cfg),
        batch_size=args.batch_size,
        mlflow_enabled=False,
    )

    print("\n" + curve.to_string(index=False))
    reporting.save_table(curve, cfg, "ptbdb_low_data_curve")

    wide = curve.pivot(index="n_train", columns="arm", values="macro_f1")
    if {"scratch", "frozen", "finetune"}.issubset(wide.columns):
        wide["frozen - scratch"] = (wide["frozen"] - wide["scratch"]).round(4)
        wide["finetune - scratch"] = (wide["finetune"] - wide["scratch"]).round(4)
    print("\n" + wide.to_string())
    reporting.save_table(wide.reset_index(), cfg, "ptbdb_low_data_deltas")

    viz.apply_style()
    viz.save_figure(
        viz.plot_low_data_curve(curve, arms=args.arms),
        cfg.figures_dir / "ptbdb_low_data_curve.png",
    )

    print(f"\ncompleted in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
