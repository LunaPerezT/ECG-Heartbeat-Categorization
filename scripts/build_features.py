#!/usr/bin/env python
"""Build the model-ready feature datasets and persist the fitted pipeline.

Usage::

    python scripts/build_features.py
    python scripts/build_features.py --source mitbih --scaler minmax
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from ecg.config import load_config
from ecg.ingest import load_beats
from ecg.preprocessing import build_dataset, split_summary
from ecg.session import get_spark, session_summary, stop_spark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--shuffle-partitions", type=int, default=None)
    parser.add_argument("--driver-memory", default=None)
    parser.add_argument("--source", choices=["mitbih", "ptbdb", "both"], default="both")
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Scaler applied to the assembled feature vector",
    )
    parser.add_argument(
        "--waveform-only",
        action="store_true",
        help="Build a feature vector from the 187 samples alone, without descriptors",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        shuffle_partitions=args.shuffle_partitions,
        driver_memory=args.driver_memory,
    ).ensure_dirs()
    cfg.validate_raw()

    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session:", session_summary(spark))

    scaler = None if args.scaler == "none" else args.scaler
    descriptors = [] if args.waveform_only else None
    sources = ["mitbih", "ptbdb"] if args.source == "both" else [args.source]

    beats = load_beats(spark, cfg)
    for source in sources:
        print(f"\n=== {source} ===")
        started = time.time()
        dataset, model = build_dataset(
            beats,
            cfg,
            source=source,
            scaler=scaler,
            descriptor_columns=descriptors,
            write=True,
        )
        vector_size = dataset.select("features").head()[0].size
        print(f"  feature vector: {vector_size} dimensions")
        print(f"  parquet:  {cfg.parquet_path(f'features_{source}')}")
        print(f"  pipeline: {cfg.parquet_path(f'preprocessing_pipeline_{source}')}")
        print(f"  stages:   {[type(stage).__name__ for stage in model.stages]}")
        print(split_summary(dataset).to_string(index=False))
        print(f"  built in {time.time() - started:.1f}s")

    stop_spark(spark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
