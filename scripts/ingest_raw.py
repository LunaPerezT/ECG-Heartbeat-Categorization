#!/usr/bin/env python
"""Ingest the four raw CSVs into partitioned Parquet.

Usage::

    python scripts/ingest_raw.py
    python scripts/ingest_raw.py --data-dir /path/to/heartbeat --shuffle-partitions 32
"""

from __future__ import annotations

import argparse
import sys
import time

from ecg.config import load_config
from ecg.ingest import BEATS_DATASET, ingest_all
from ecg.session import get_spark, session_summary, stop_spark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="Folder holding the four raw CSVs")
    parser.add_argument("--processed-dir", default=None, help="Destination for Parquet output")
    parser.add_argument("--shuffle-partitions", type=int, default=None)
    parser.add_argument("--driver-memory", default=None, help="e.g. 4g")
    parser.add_argument("--dataset-name", default=BEATS_DATASET)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        shuffle_partitions=args.shuffle_partitions,
        driver_memory=args.driver_memory,
    )
    cfg.validate_raw()

    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session:", session_summary(spark))
    print("Reading from:", cfg.raw_dir)

    started = time.time()
    beats = ingest_all(spark, cfg, write=True, dataset_name=args.dataset_name)
    total = beats.count()
    elapsed = time.time() - started

    print(f"Wrote {total:,} beats to {cfg.parquet_path(args.dataset_name)} in {elapsed:.1f}s")
    beats.groupBy("source", "split").count().orderBy("source", "split").show()

    stop_spark(spark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
