"""Ingestion: raw CSV parsing, canonicalisation and Parquet round-trip."""

from __future__ import annotations

import pytest

from ecg.config import Config
from ecg.ingest import (
    RAW_FILE_SPECS,
    ingest_all,
    load_mitbih,
    read_parquet,
    read_raw_csv,
    to_canonical,
    write_parquet,
)
from ecg.schema import N_SAMPLES, canonical_schema


def write_csv(path, rows):
    """Write rows of ``(samples..., label)`` in the dataset's own CSV format."""
    with open(path, "w", encoding="utf-8") as handle:
        for samples, label in rows:
            padded = list(samples) + [0.0] * (N_SAMPLES - len(samples))
            handle.write(",".join(f"{value:.18e}" for value in padded + [float(label)]) + "\n")


@pytest.fixture()
def tiny_dataset(tmp_path):
    """A miniature copy of the dataset layout, with known class counts."""
    write_csv(tmp_path / "mitbih_train.csv", [([1.0, 0.5], 0), ([1.0, 0.2], 2), ([1.0, 0.9], 3)])
    write_csv(tmp_path / "mitbih_test.csv", [([1.0, 0.4], 0)])
    write_csv(tmp_path / "ptbdb_normal.csv", [([1.0, 0.3], 0)])
    write_csv(tmp_path / "ptbdb_abnormal.csv", [([1.0, 0.7], 1), ([1.0, 0.6], 1)])
    return tmp_path


def test_read_raw_csv_shape(spark, tiny_dataset):
    df = read_raw_csv(spark, tiny_dataset / "mitbih_train.csv")
    assert df.count() == 3
    assert len(df.columns) == N_SAMPLES + 1


def test_to_canonical_matches_the_declared_schema(spark, tiny_dataset):
    raw = read_raw_csv(spark, tiny_dataset / "mitbih_train.csv")
    canonical = to_canonical(raw, source="mitbih", split="train")
    assert canonical.schema.fieldNames() == canonical_schema().fieldNames()

    row = canonical.first()
    assert len(row["signal"]) == N_SAMPLES
    assert row["source"] == "mitbih"
    assert row["split"] == "train"
    assert row["beat_id"].startswith("mitbih_train_")


def test_labels_are_mapped_to_names(spark, tiny_dataset):
    cfg = Config(data_dir=tiny_dataset)
    mapping = {
        row["label"]: row["label_name"]
        for row in load_mitbih(spark, cfg).select("label", "label_name").distinct().collect()
    }
    assert mapping == {0: "N", 2: "V", 3: "F"}


def test_beat_ids_are_unique(spark, tiny_dataset):
    cfg = Config(data_dir=tiny_dataset)
    beats = load_mitbih(spark, cfg)
    assert beats.select("beat_id").distinct().count() == beats.count()


def test_ingest_all_round_trips_through_parquet(spark, tiny_dataset, tmp_path):
    cfg = Config(data_dir=tiny_dataset, processed_dir=tmp_path / "processed")
    stored = ingest_all(spark, cfg, write=True)

    counts = {
        (row["source"], row["split"]): row["count"]
        for row in stored.groupBy("source", "split").count().collect()
    }
    assert counts == {("mitbih", "train"): 3, ("mitbih", "test"): 1, ("ptbdb", "full"): 3}
    assert (cfg.parquet_path("beats") / "source=mitbih").exists()


def test_write_parquet_without_partitions(spark, tiny_dataset, tmp_path):
    cfg = Config(data_dir=tiny_dataset)
    destination = tmp_path / "flat"
    write_parquet(load_mitbih(spark, cfg), destination, partition_by=())
    assert read_parquet(spark, destination).count() == 4


def test_missing_files_raise_a_helpful_error(tmp_path):
    cfg = Config(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="kaggle.com"):
        cfg.validate_raw()


def test_every_raw_file_has_a_spec():
    from ecg.config import RAW_FILES

    assert set(RAW_FILES) == set(RAW_FILE_SPECS)
