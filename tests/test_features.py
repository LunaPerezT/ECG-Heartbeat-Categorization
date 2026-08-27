"""Per-beat descriptors, checked against hand-computed values."""

from __future__ import annotations

import pytest
from pyspark.sql import functions as F

from ecg.features import (
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    add_beat_features,
    add_fingerprint,
    effective_length,
)
from ecg.schema import N_SAMPLES, canonical_schema

from .conftest import KNOWN_EXPECTED, pad


@pytest.fixture(scope="module")
def known_row(known_beat):
    return add_beat_features(known_beat).first()


@pytest.mark.parametrize("name", sorted(KNOWN_EXPECTED))
def test_descriptor_matches_hand_computed_value(known_row, name):
    assert known_row[name] == pytest.approx(KNOWN_EXPECTED[name], rel=1e-9, abs=1e-12)


def test_all_declared_features_are_produced(known_beat):
    produced = add_beat_features(known_beat).columns
    for name in FEATURE_COLUMNS:
        assert name in produced


def test_model_features_are_a_subset_of_all_features():
    assert set(MODEL_FEATURE_COLUMNS).issubset(FEATURE_COLUMNS)


def test_effective_length_ignores_trailing_padding_only(spark):
    """An interior zero must not truncate the beat; a trailing one must."""
    rows = [
        ("interior", "mitbih", "train", 0, "N", pad([1.0, 0.0, 0.4])),
        ("all_zero", "mitbih", "train", 0, "N", pad([])),
        ("full", "mitbih", "train", 0, "N", [0.5] * N_SAMPLES),
    ]
    df = spark.createDataFrame(rows, schema=canonical_schema())
    lengths = {
        row["beat_id"]: row["length"]
        for row in df.select("beat_id", effective_length().alias("length")).collect()
    }
    assert lengths == {"interior": 3, "all_zero": 0, "full": N_SAMPLES}


def test_duration_and_padding_ratio_are_consistent(known_row):
    assert known_row["duration_s"] == pytest.approx(4 / 125)
    assert known_row["padding_ratio"] == pytest.approx((N_SAMPLES - 4) / N_SAMPLES)


def test_statistics_ignore_the_padding(spark):
    """Padding must not be averaged in: the mean is over the real prefix only."""
    df = spark.createDataFrame(
        [("b", "mitbih", "train", 0, "N", pad([0.8, 0.4]))], schema=canonical_schema()
    )
    row = add_beat_features(df).first()
    assert row["amp_mean"] == pytest.approx(0.6)  # not 1.2 / 187
    assert row["amp_min"] == pytest.approx(0.4)  # the padding zeros are excluded


def test_fingerprint_is_stable_and_discriminating(spark):
    rows = [
        ("a", "mitbih", "train", 0, "N", pad([1.0, 0.5])),
        ("b", "mitbih", "train", 0, "N", pad([1.0, 0.5])),
        ("c", "mitbih", "train", 0, "N", pad([1.0, 0.6])),
    ]
    df = add_fingerprint(spark.createDataFrame(rows, schema=canonical_schema()))
    hashes = {row["beat_id"]: row["signal_hash"] for row in df.collect()}
    assert hashes["a"] == hashes["b"]
    assert hashes["a"] != hashes["c"]


def test_features_add_no_rows(synthetic_beats):
    assert add_beat_features(synthetic_beats).count() == synthetic_beats.count()


def test_no_nulls_or_nans_in_descriptors(synthetic_beats):
    featured = add_beat_features(synthetic_beats)
    bad = featured.where(
        F.greatest(
            *[F.col(name).isNull() | F.isnan(F.col(name).cast("double")) for name in FEATURE_COLUMNS]
        )
    )
    assert bad.count() == 0
