"""Raw CSV ingestion into the canonical Spark representation.

The four published CSVs are head-less matrices of 188 float columns. This module
turns them into a single tidy DataFrame with one row per heartbeat, an
``array<double>`` signal column, a typed label and provenance columns, and
persists it as partitioned Parquet so every later stage reads columnar data
instead of re-parsing 583 MB of text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ecg.config import Config
from ecg.schema import (
    LABEL_MAPS,
    RAW_LABEL_COLUMN,
    SIGNAL_COLUMNS,
    canonical_schema,
    raw_schema,
)

#: Which raw file belongs to which collection and split.
RAW_FILE_SPECS: Dict[str, Dict[str, str]] = {
    "mitbih_train": {"source": "mitbih", "split": "train"},
    "mitbih_test": {"source": "mitbih", "split": "test"},
    "ptbdb_normal": {"source": "ptbdb", "split": "full"},
    "ptbdb_abnormal": {"source": "ptbdb", "split": "full"},
}

#: Name of the Parquet dataset written by :func:`ingest_all`.
BEATS_DATASET = "beats"


def read_raw_csv(spark: SparkSession, path: Path | str) -> DataFrame:
    """Read one raw CSV with the fixed 188-column schema.

    Args:
        spark: Active Spark session.
        path: Path to the CSV file.

    Returns:
        A DataFrame with columns ``s000`` .. ``s186`` and ``label_raw``.
    """
    return spark.read.csv(
        str(path),
        schema=raw_schema(),
        header=False,
        mode="FAILFAST",
    )


def to_canonical(df: DataFrame, source: str, split: str) -> DataFrame:
    """Reshape a raw DataFrame into the canonical one-row-per-beat form.

    The 187 wide sample columns are collapsed into a single ``array<double>``.
    Keeping the beat as an array rather than 187 columns is what makes the rest of
    the pipeline expressible with Spark's higher-order array functions, and it
    keeps the Parquet footprint small.

    Args:
        df: Output of :func:`read_raw_csv`.
        source: ``"mitbih"`` or ``"ptbdb"``.
        split: ``"train"``, ``"test"`` or ``"full"``.

    Returns:
        A DataFrame matching :func:`ecg.schema.canonical_schema`.

    Raises:
        KeyError: If ``source`` is not a known collection.
    """
    label_map = LABEL_MAPS[source]
    label_expr = F.create_map(
        *[item for key, value in label_map.items() for item in (F.lit(key), F.lit(value))]
    )

    # monotonically_increasing_id is unique and partition-ordered, which is all a
    # beat identifier needs; it avoids the full shuffle a row_number() window costs.
    beat_index = F.monotonically_increasing_id()

    return (
        df.select(
            F.concat_ws(
                "_", F.lit(source), F.lit(split), F.format_string("%012d", beat_index)
            ).alias("beat_id"),
            F.lit(source).alias("source"),
            F.lit(split).alias("split"),
            F.col(RAW_LABEL_COLUMN).cast("int").alias("label"),
            F.array(*[F.col(name) for name in SIGNAL_COLUMNS]).alias("signal"),
        )
        .withColumn("label_name", label_expr[F.col("label")])
        .select(*[field.name for field in canonical_schema().fields])
    )


def load_raw_file(spark: SparkSession, cfg: Config, key: str) -> DataFrame:
    """Read and canonicalise a single raw file by its config key.

    Args:
        spark: Active Spark session.
        cfg: Project configuration.
        key: One of the keys in :data:`RAW_FILE_SPECS`.

    Returns:
        A canonical DataFrame for that file.
    """
    spec = RAW_FILE_SPECS[key]
    raw = read_raw_csv(spark, cfg.raw_path(key))
    return to_canonical(raw, source=spec["source"], split=spec["split"])


def load_mitbih(spark: SparkSession, cfg: Config) -> DataFrame:
    """Load the MIT-BIH Arrhythmia collection (train + test) as one DataFrame."""
    train = load_raw_file(spark, cfg, "mitbih_train")
    test = load_raw_file(spark, cfg, "mitbih_test")
    return train.unionByName(test)


def load_ptbdb(spark: SparkSession, cfg: Config) -> DataFrame:
    """Load the PTB Diagnostic collection (normal + abnormal) as one DataFrame.

    The published PTB files carry no train/test split, so both are tagged
    ``split = "full"`` and the split is created later, reproducibly, by
    :func:`ecg.preprocessing.stratified_split`.
    """
    normal = load_raw_file(spark, cfg, "ptbdb_normal")
    abnormal = load_raw_file(spark, cfg, "ptbdb_abnormal")
    return normal.unionByName(abnormal)


def load_all(spark: SparkSession, cfg: Config) -> DataFrame:
    """Load both collections into a single canonical DataFrame."""
    return load_mitbih(spark, cfg).unionByName(load_ptbdb(spark, cfg))


def write_parquet(
    df: DataFrame,
    path: Path | str,
    partition_by: Iterable[str] = ("source", "split"),
    mode: str = "overwrite",
) -> str:
    """Write a DataFrame as partitioned Parquet.

    Args:
        df: DataFrame to persist.
        path: Destination directory.
        partition_by: Partition columns; ``("source", "split")`` yields four small
            partitions that let later stages read one collection without touching
            the other.
        mode: Spark save mode.

    Returns:
        The destination path as a string.
    """
    destination = str(path)
    writer = df.write.mode(mode)
    partition_cols = list(partition_by)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(destination)
    return destination


def read_parquet(spark: SparkSession, path: Path | str) -> DataFrame:
    """Read a Parquet dataset written by :func:`write_parquet`."""
    return spark.read.parquet(str(path))


def ingest_all(
    spark: SparkSession,
    cfg: Config,
    write: bool = True,
    dataset_name: str = BEATS_DATASET,
) -> DataFrame:
    """Run the full ingestion: validate inputs, canonicalise, optionally persist.

    Args:
        spark: Active Spark session.
        cfg: Project configuration.
        write: When ``True`` the result is written to
            ``cfg.parquet_path(dataset_name)`` and read back, so downstream code
            works against the columnar copy.
        dataset_name: Name of the Parquet dataset directory.

    Returns:
        The canonical DataFrame for both collections.

    Example:
        >>> cfg = load_config()                       # doctest: +SKIP
        >>> beats = ingest_all(get_spark(cfg), cfg)   # doctest: +SKIP
        >>> beats.count()                             # doctest: +SKIP
        123998
    """
    cfg.validate_raw()
    beats = load_all(spark, cfg)

    if not write:
        return beats

    cfg.ensure_dirs()
    destination = cfg.parquet_path(dataset_name)
    write_parquet(beats, destination)
    return read_parquet(spark, destination)


def load_beats(
    spark: SparkSession,
    cfg: Config,
    source: Optional[str] = None,
    split: Optional[str] = None,
    dataset_name: str = BEATS_DATASET,
) -> DataFrame:
    """Read the ingested Parquet dataset, optionally filtered by partition.

    Falls back to reading the raw CSVs when the Parquet dataset does not exist
    yet, so notebooks work on a fresh clone without a mandatory ingestion step.

    Args:
        spark: Active Spark session.
        cfg: Project configuration.
        source: Optional ``mitbih``/``ptbdb`` filter (partition pruned).
        split: Optional ``train``/``test``/``full`` filter (partition pruned).
        dataset_name: Name of the Parquet dataset directory.

    Returns:
        The requested canonical DataFrame.
    """
    destination = cfg.parquet_path(dataset_name)
    if Path(destination).exists():
        df = read_parquet(spark, destination)
    else:
        df = ingest_all(spark, cfg, write=False)

    if source is not None:
        df = df.where(F.col("source") == source)
    if split is not None:
        df = df.where(F.col("split") == split)
    return df
