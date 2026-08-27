"""Splits, class weights, resampling and the Spark ML pipeline."""

from __future__ import annotations

import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from ecg.config import Config
from ecg.features import MODEL_FEATURE_COLUMNS
from ecg.preprocessing import (
    DEFAULT_SPLIT_FRACTIONS,
    add_class_weight,
    build_preprocessing_pipeline,
    class_counts,
    class_weights,
    holdout_validation,
    resample,
    split_summary,
    stratified_split,
)
from ecg.schema import N_SAMPLES


def test_stratified_split_preserves_class_shares(synthetic_beats):
    split = stratified_split(synthetic_beats, fractions=DEFAULT_SPLIT_FRACTIONS, seed=1)
    shares = (
        split.groupBy("split_assigned", "label_name")
        .count()
        .groupBy("label_name")
        .pivot("split_assigned")
        .sum("count")
        .toPandas()
        .set_index("label_name")
        .fillna(0)
    )
    for label_name, row in shares.iterrows():
        total = row.sum()
        assert row["train"] / total == pytest.approx(0.70, abs=0.15), label_name
        assert row["test"] / total == pytest.approx(0.15, abs=0.15), label_name


def test_stratified_split_is_deterministic(synthetic_beats):
    first = stratified_split(synthetic_beats, seed=3).select("beat_id", "split_assigned")
    second = stratified_split(synthetic_beats, seed=3).select("beat_id", "split_assigned")
    assert first.exceptAll(second).count() == 0


def test_stratified_split_assigns_every_row_exactly_once(synthetic_beats):
    split = stratified_split(synthetic_beats, seed=1)
    assert split.count() == synthetic_beats.count()
    assert split.where(F.col("split_assigned").isNull()).count() == 0
    assert set(row[0] for row in split.select("split_assigned").distinct().collect()) == {
        "train",
        "val",
        "test",
    }


def test_split_fractions_must_sum_to_one(synthetic_beats):
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_split(synthetic_beats, fractions={"train": 0.5, "test": 0.2})


def test_holdout_validation_leaves_the_published_test_split_untouched(synthetic_beats):
    original_test = synthetic_beats.where(F.col("split") == "test").count()
    split = holdout_validation(synthetic_beats, val_fraction=0.2, seed=5)
    counts = {row["split_final"]: row["n"] for row in
              split.groupBy("split_final").agg(F.count(F.lit(1)).alias("n")).collect()}
    assert counts["test"] == original_test
    assert counts["train"] + counts["val"] == synthetic_beats.count() - original_test


def test_class_weights_follow_the_balanced_formula(synthetic_beats):
    counts = class_counts(synthetic_beats)
    weights = class_weights(synthetic_beats)
    total, n_classes = sum(counts.values()), len(counts)
    for label, count in counts.items():
        assert weights[label] == pytest.approx(total / (n_classes * count))
    # The rarest class must be weighted above the most common one.
    assert weights[3] > weights[0]


def test_add_class_weight_attaches_one_weight_per_row(synthetic_beats):
    weighted = add_class_weight(synthetic_beats)
    assert weighted.where(F.col("class_weight").isNull()).count() == 0
    assert weighted.select("label", "class_weight").distinct().count() == 5


def test_undersampling_shrinks_towards_the_minority(synthetic_beats):
    counts = class_counts(resample(synthetic_beats, "undersample", seed=2))
    assert max(counts.values()) <= 12  # minority size is 4; Bernoulli sampling is approximate
    assert min(counts.values()) >= 1


def test_oversampling_grows_towards_the_majority(synthetic_beats):
    counts = class_counts(resample(synthetic_beats, "oversample", seed=2))
    assert min(counts.values()) >= 40
    assert max(counts.values()) <= 80


def test_unknown_resample_strategy_raises(synthetic_beats):
    with pytest.raises(ValueError, match="Unknown strategy"):
        resample(synthetic_beats, "smote")


def test_pipeline_produces_the_expected_vector_size(synthetic_beats):
    model = build_preprocessing_pipeline(scaler="standard").fit(synthetic_beats)
    transformed = model.transform(synthetic_beats)
    expected = N_SAMPLES + len(MODEL_FEATURE_COLUMNS)
    assert transformed.select("features").first()[0].size == expected


def test_pipeline_without_descriptors_is_waveform_only(synthetic_beats):
    model = build_preprocessing_pipeline(descriptor_columns=[], scaler=None).fit(synthetic_beats)
    transformed = model.transform(synthetic_beats)
    assert transformed.select("features").first()[0].size == N_SAMPLES


def test_unknown_scaler_raises():
    with pytest.raises(ValueError, match="Unknown scaler"):
        build_preprocessing_pipeline(scaler="robust")


def test_pipeline_survives_a_save_load_round_trip(synthetic_beats, tmp_path):
    model = build_preprocessing_pipeline(scaler="standard").fit(synthetic_beats)
    destination = str(tmp_path / "pipeline")
    model.write().overwrite().save(destination)

    reloaded = PipelineModel.load(destination)
    assert [type(stage).__name__ for stage in reloaded.stages] == [
        type(stage).__name__ for stage in model.stages
    ]

    before = model.transform(synthetic_beats).select("features").first()[0].toArray()
    after = reloaded.transform(synthetic_beats).select("features").first()[0].toArray()
    assert before == pytest.approx(after)


def test_split_summary_shares_sum_to_100(synthetic_beats):
    split = stratified_split(synthetic_beats, output_col="split_final", seed=1)
    summary = split_summary(split)
    totals = summary.groupby("split_final")["pct"].sum()
    assert totals.round(1).eq(100.0).all()


def test_config_rejects_unknown_keys(tmp_path):
    from ecg.config import load_config

    with pytest.raises(ValueError, match="Unknown configuration keys"):
        load_config(config_file=tmp_path / "missing.yaml", nonsense=1)


def test_config_finds_a_nested_raw_folder(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "mitbih_train.csv").write_text("")
    cfg = Config(data_dir=tmp_path)
    assert cfg.raw_dir == raw
