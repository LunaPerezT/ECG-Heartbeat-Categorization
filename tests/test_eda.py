"""EDA aggregations, checked against a synthetic dataset with known counts."""

from __future__ import annotations

import pytest

from ecg import eda
from ecg.schema import canonical_schema

from .conftest import pad

EXPECTED_COUNTS = {"N": 60, "S": 20, "V": 12, "F": 4, "Q": 4}


def test_dataset_overview_totals(synthetic_beats):
    overview = eda.dataset_overview(synthetic_beats)
    assert overview["n_beats"].sum() == 100
    assert overview["pct_of_total"].sum() == pytest.approx(100.0, abs=0.01)


def test_class_distribution_counts_and_shares(synthetic_beats):
    distribution = eda.class_distribution(synthetic_beats, by=("source",))
    counts = dict(zip(distribution["label_name"], distribution["n_beats"]))
    assert counts == EXPECTED_COUNTS
    assert distribution["pct"].sum() == pytest.approx(100.0, abs=0.01)


def test_imbalance_summary_identifies_the_extremes(synthetic_beats):
    distribution = eda.class_distribution(synthetic_beats, by=("source",))
    summary = eda.imbalance_summary(distribution).iloc[0]
    assert summary["majority_class"] == "N"
    assert summary["minority_class"] in {"F", "Q"}
    assert summary["imbalance_ratio"] == pytest.approx(15.0)


def test_quality_report_is_clean_for_valid_data(synthetic_beats):
    report = eda.quality_report(synthetic_beats).set_index("check")["n_beats"]
    assert report["rows_total"] == 100
    for check in ("null_signal", "wrong_length", "contains_nan_sample", "below_zero", "above_one"):
        assert report[check] == 0


def test_quality_report_flags_bad_rows(spark):
    rows = [
        ("ok", "mitbih", "train", 0, "N", pad([1.0, 0.5])),
        ("high", "mitbih", "train", 0, "N", pad([1.4, 0.5])),
        ("low", "mitbih", "train", 0, "N", pad([1.0, -0.2])),
        ("empty", "mitbih", "train", 0, "N", pad([])),
    ]
    report = (
        eda.quality_report(spark.createDataFrame(rows, schema=canonical_schema()))
        .set_index("check")["n_beats"]
    )
    assert report["above_one"] == 1
    assert report["below_zero"] == 1
    assert report["all_zero_beat"] == 1


def test_duplicate_report_counts_repeats_and_conflicts(spark):
    rows = [
        ("a", "mitbih", "train", 0, "N", pad([1.0, 0.5])),
        ("b", "mitbih", "test", 0, "N", pad([1.0, 0.5])),  # same waveform, other split
        ("c", "mitbih", "train", 2, "V", pad([1.0, 0.5])),  # same waveform, other label
        ("d", "mitbih", "train", 0, "N", pad([1.0, 0.9])),
    ]
    report = eda.duplicate_report(
        spark.createDataFrame(rows, schema=canonical_schema()), by=("source",)
    ).iloc[0]
    assert report["n_beats"] == 4
    assert report["n_unique_waveforms"] == 2
    assert report["n_duplicated_beats"] == 2
    assert report["n_waveforms_with_conflicting_labels"] == 1
    assert report["n_waveforms_in_both_splits"] == 1


def test_padding_report_bounds(synthetic_beats):
    report = eda.padding_report(synthetic_beats)
    assert set(report["label_name"]) == set(EXPECTED_COUNTS)
    assert (report["min_length"] <= report["median_length"]).all()
    assert (report["median_length"] <= report["max_length"]).all()


def test_feature_summary_quantiles_are_ordered(synthetic_beats):
    summary = eda.feature_summary(synthetic_beats, features=["signal_length", "amp_mean"])
    assert set(summary["feature"]) == {"signal_length", "amp_mean"}
    assert (summary["min"] <= summary["q25"]).all()
    assert (summary["q25"] <= summary["q50"]).all()
    assert (summary["q50"] <= summary["q75"]).all()
    assert (summary["q75"] <= summary["max"]).all()


def test_feature_correlation_is_square_and_unit_diagonal(synthetic_beats):
    matrix = eda.feature_correlation(synthetic_beats, features=["signal_length", "amp_mean"])
    assert list(matrix.columns) == ["signal_length", "amp_mean", "label"]
    assert matrix.shape == (3, 3)
    for name in matrix.columns:
        assert matrix.loc[name, name] == pytest.approx(1.0)


def test_waveform_profile_shape_and_support(synthetic_beats):
    profile = eda.waveform_profile(synthetic_beats, min_support_ratio=0.0)
    assert {"t", "time_s", "mean", "std", "q25", "q50", "q75", "n"}.issubset(profile.columns)
    assert profile["t"].min() == 0
    assert (profile["n"] > 0).all()


def test_waveform_profile_support_filter_trims_the_tail(synthetic_beats):
    full = eda.waveform_profile(synthetic_beats, min_support_ratio=0.0)
    trimmed = eda.waveform_profile(synthetic_beats, min_support_ratio=0.5)
    assert len(trimmed) < len(full)


def test_sample_beats_returns_the_requested_count(synthetic_beats):
    sample = eda.sample_beats(synthetic_beats, n_per_group=3, seed=7)
    per_class = sample.groupby("label_name").size()
    assert (per_class <= 3).all()
    assert per_class["N"] == 3
    assert len(sample.iloc[0]["signal"]) == 187


def test_sample_beats_is_reproducible(synthetic_beats):
    first = eda.sample_beats(synthetic_beats, n_per_group=3, seed=7)["beat_id"].tolist()
    second = eda.sample_beats(synthetic_beats, n_per_group=3, seed=7)["beat_id"].tolist()
    assert first == second


def test_length_histogram_shares_sum_to_100(synthetic_beats):
    histogram = eda.length_histogram(synthetic_beats, bin_width=10)
    totals = histogram.groupby("label_name")["pct"].sum()
    assert totals.round(2).eq(100.0).all()
    assert (histogram["bin_end"] - histogram["bin_start"] == 10).all()


def test_ordered_labels_filters_to_present_classes():
    assert eda.ordered_labels("mitbih") == ["N", "S", "V", "F", "Q"]
    assert eda.ordered_labels("mitbih", present=["Q", "N"]) == ["N", "Q"]
