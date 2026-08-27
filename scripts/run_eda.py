#!/usr/bin/env python
"""Run the full exploratory analysis and write every table and figure to reports/.

Usage::

    python scripts/run_eda.py
    python scripts/run_eda.py --source mitbih --shuffle-partitions 32
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

from ecg import eda, viz
from ecg.config import load_config
from ecg.features import add_beat_features
from ecg.ingest import load_beats
from ecg.session import get_spark, session_summary, stop_spark

#: Descriptors shown in the per-class box panels.
BOX_FEATURES = [
    "signal_length",
    "amp_mean",
    "amp_std",
    "energy",
    "peak_index",
    "mean_abs_diff",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--shuffle-partitions", type=int, default=None)
    parser.add_argument("--driver-memory", default=None)
    parser.add_argument(
        "--source",
        choices=["mitbih", "ptbdb", "both"],
        default="both",
        help="Collection to analyse",
    )
    return parser.parse_args(argv)


def save_table(frame: pd.DataFrame, cfg, name: str) -> None:
    """Write a summary table to reports/tables and echo its shape."""
    destination = cfg.tables_dir / f"{name}.csv"
    frame.to_csv(destination, index=False)
    print(f"  table  {destination.name:<34} {frame.shape[0]:>5} rows")


def analyse(spark, cfg, source: str) -> None:
    """Run every EDA step for one collection."""
    print(f"\n=== {source} ===")
    beats = add_beat_features(load_beats(spark, cfg, source=source)).cache()
    n_beats = beats.count()
    print(f"  {n_beats:,} beats")

    order = eda.ordered_labels(source)

    distribution = eda.class_distribution(beats)
    save_table(distribution, cfg, f"{source}_class_distribution")
    save_table(eda.imbalance_summary(distribution), cfg, f"{source}_imbalance")
    save_table(eda.quality_report(beats), cfg, f"{source}_quality")
    save_table(eda.duplicate_report(beats), cfg, f"{source}_duplicates")
    save_table(eda.padding_report(beats), cfg, f"{source}_padding")

    summary = eda.feature_summary(beats)
    save_table(summary, cfg, f"{source}_feature_summary")

    correlation = eda.feature_correlation(beats)
    save_table(correlation.reset_index(names="feature"), cfg, f"{source}_correlation")

    profile = eda.waveform_profile(beats)
    histogram = eda.length_histogram(beats)
    save_table(histogram, cfg, f"{source}_length_histogram")
    save_table(profile, cfg, f"{source}_waveform_profile")
    samples = eda.sample_beats(beats, n_per_group=cfg.plot_sample_per_class, seed=cfg.seed)

    n_cols = 3 if len(order) > 2 else 2
    figures = {
        f"{source}_class_distribution": viz.plot_class_distribution(distribution, source, order),
        f"{source}_sample_beats": viz.plot_sample_beats(samples, source, order, n_per_class=4),
        f"{source}_waveform_profiles": viz.plot_waveform_profiles(
            profile, source, order, n_cols=n_cols
        ),
        f"{source}_length_distribution": viz.plot_length_distribution(
            histogram, source, order, n_cols=n_cols
        ),
        f"{source}_feature_boxes": viz.plot_feature_boxes(summary, source, BOX_FEATURES, order),
        f"{source}_correlation": viz.plot_correlation_heatmap(
            correlation, title=f"Descriptor correlation — {source}"
        ),
    }
    if source == "mitbih":
        figures[f"{source}_split_comparison"] = viz.plot_split_comparison(
            distribution, source, order
        )

    for name, figure in figures.items():
        path = viz.save_figure(figure, cfg.figures_dir / f"{name}.png")
        print(f"  figure {name + '.png':<34} {path}")

    beats.unpersist()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        shuffle_partitions=args.shuffle_partitions,
        driver_memory=args.driver_memory,
    ).ensure_dirs()
    cfg.validate_raw()

    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session:", session_summary(spark))
    viz.apply_style()

    started = time.time()
    sources = ["mitbih", "ptbdb"] if args.source == "both" else [args.source]
    for source in sources:
        analyse(spark, cfg, source)

    print(f"\nCompleted in {time.time() - started:.1f}s")
    print(f"Figures: {cfg.figures_dir}")
    print(f"Tables:  {cfg.tables_dir}")
    stop_spark(spark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
