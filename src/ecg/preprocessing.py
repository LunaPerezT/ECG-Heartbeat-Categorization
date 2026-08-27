"""Spark ML preprocessing: descriptors, feature vectors, splits and rebalancing.

The pipeline is expressed with real :class:`pyspark.ml.Pipeline` stages — including
two small custom transformers — rather than a chain of ad-hoc DataFrame calls, so
the fitted :class:`~pyspark.ml.PipelineModel` can be saved, versioned and reloaded
next to the data it produced, on a laptop or on Databricks alike.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ecg.config import Config
from ecg.features import MODEL_FEATURE_COLUMNS, add_beat_features

#: Name of the Parquet dataset written by :func:`build_dataset`.
FEATURES_DATASET = "features"

#: Name of the saved pipeline directory.
PIPELINE_DIR = "preprocessing_pipeline"

#: Default proportions used when a collection ships without a published split.
DEFAULT_SPLIT_FRACTIONS: Dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


class BeatFeatureTransformer(
    Transformer, HasInputCol, DefaultParamsReadable, DefaultParamsWritable
):
    """Pipeline stage that appends the per-beat descriptors from :mod:`ecg.features`.

    Wrapping the descriptor expressions in a Transformer keeps them inside the
    saved :class:`~pyspark.ml.PipelineModel`, so scoring code cannot drift from the
    feature definitions used at fit time.

    Example:
        >>> BeatFeatureTransformer(inputCol="signal").transform(beats)  # doctest: +SKIP
    """

    def __init__(self, inputCol: str = "signal") -> None:
        super().__init__()
        self._setDefault(inputCol="signal")
        self._set(inputCol=inputCol)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return add_beat_features(dataset, signal_col=self.getInputCol())


class ArrayToVector(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
):
    """Pipeline stage converting an ``array<double>`` column into an MLlib vector.

    :func:`pyspark.ml.functions.array_to_vector` is not available as a SQL
    function, so it cannot be expressed with ``SQLTransformer``; this thin wrapper
    fills that gap.
    """

    def __init__(self, inputCol: str = "signal", outputCol: str = "signal_vec") -> None:
        super().__init__()
        self._setDefault(inputCol="signal", outputCol="signal_vec")
        self._set(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.withColumn(self.getOutputCol(), array_to_vector(F.col(self.getInputCol())))


def build_preprocessing_pipeline(
    descriptor_columns: Optional[Iterable[str]] = None,
    scaler: Optional[str] = "standard",
    signal_col: str = "signal",
    output_col: str = "features",
) -> Pipeline:
    """Assemble the preprocessing :class:`~pyspark.ml.Pipeline`.

    Stages, in order:

    1. :class:`BeatFeatureTransformer` — adds the per-beat descriptors.
    2. :class:`ArrayToVector` — 187 samples become a dense vector.
    3. :class:`~pyspark.ml.feature.VectorAssembler` — waveform ⊕ descriptors.
    4. A scaler, if requested.

    Args:
        descriptor_columns: Descriptors appended to the waveform; defaults to
            :data:`ecg.features.MODEL_FEATURE_COLUMNS`. Pass an empty list for a
            waveform-only feature vector.
        scaler: ``"standard"`` (zero mean, unit variance), ``"minmax"`` or ``None``.
            The raw waveform is already min-max normalised per beat, so scaling
            mainly matters for the descriptors, whose units differ by orders of
            magnitude (samples vs. energy).
        signal_col: Name of the ``array<double>`` column.
        output_col: Name of the final feature vector column.

    Returns:
        An unfitted pipeline.

    Raises:
        ValueError: If ``scaler`` is not a recognised option.
    """
    descriptors: List[str] = (
        list(MODEL_FEATURE_COLUMNS) if descriptor_columns is None else list(descriptor_columns)
    )
    assembled_col = f"{output_col}_raw" if scaler else output_col

    stages: List[Transformer] = [
        BeatFeatureTransformer(inputCol=signal_col),
        ArrayToVector(inputCol=signal_col, outputCol="signal_vec"),
        VectorAssembler(
            inputCols=["signal_vec"] + descriptors,
            outputCol=assembled_col,
            handleInvalid="skip",
        ),
    ]

    if scaler == "standard":
        stages.append(
            StandardScaler(
                inputCol=assembled_col, outputCol=output_col, withMean=True, withStd=True
            )
        )
    elif scaler == "minmax":
        stages.append(MinMaxScaler(inputCol=assembled_col, outputCol=output_col))
    elif scaler is not None:
        raise ValueError(f"Unknown scaler {scaler!r}; expected 'standard', 'minmax' or None")

    return Pipeline(stages=stages)


# --------------------------------------------------------------------- splits


def stratified_split(
    df: DataFrame,
    fractions: Mapping[str, float] = DEFAULT_SPLIT_FRACTIONS,
    label_col: str = "label",
    output_col: str = "split_assigned",
    seed: int = 42,
) -> DataFrame:
    """Assign a stratified split label to every row, deterministically.

    Uses ``percent_rank`` over a per-class random ordering rather than
    ``randomSplit``: the class proportions of each part then match the source
    exactly, which matters a lot here because the rarest MIT-BIH class holds well
    under 1% of the beats.

    Args:
        df: Any DataFrame with a label column.
        fractions: Split name → share, summing to 1.
        label_col: Column to stratify on.
        output_col: Name of the split column to add.
        seed: Random seed.

    Returns:
        ``df`` with the split column appended.

    Raises:
        ValueError: If the fractions do not sum to 1.
    """
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1, got {total}")

    window = Window.partitionBy(label_col).orderBy(F.rand(seed))
    ranked = df.withColumn("__percentile", F.percent_rank().over(window))

    assignment = None
    cumulative = 0.0
    names = list(fractions)
    for name in names[:-1]:
        cumulative += fractions[name]
        condition = F.col("__percentile") < F.lit(cumulative)
        assignment = F.when(condition, F.lit(name)) if assignment is None else assignment.when(
            condition, F.lit(name)
        )
    assignment = (
        F.lit(names[-1]) if assignment is None else assignment.otherwise(F.lit(names[-1]))
    )

    return ranked.withColumn(output_col, assignment).drop("__percentile")


def holdout_validation(
    df: DataFrame,
    val_fraction: float = 0.15,
    label_col: str = "label",
    split_col: str = "split",
    output_col: str = "split_final",
    seed: int = 42,
) -> DataFrame:
    """Carve a stratified validation set out of an existing ``train`` split.

    The published MIT-BIH test split is left untouched, so results stay comparable
    with the literature that uses it.

    Args:
        df: Canonical beats DataFrame with a ``split`` column.
        val_fraction: Share of the training beats moved to ``val``.
        label_col: Column to stratify on.
        split_col: Existing split column.
        output_col: Name of the resulting split column.
        seed: Random seed.

    Returns:
        ``df`` with ``output_col`` holding ``train``/``val``/``test``.
    """
    train = df.where(F.col(split_col) == "train")
    rest = df.where(F.col(split_col) != "train")

    split_train = stratified_split(
        train,
        fractions={"train": 1.0 - val_fraction, "val": val_fraction},
        label_col=label_col,
        output_col=output_col,
        seed=seed,
    )
    return split_train.unionByName(rest.withColumn(output_col, F.col(split_col)))


# ------------------------------------------------------------------ balancing


def class_counts(df: DataFrame, label_col: str = "label") -> Dict[int, int]:
    """Return ``{label: n_rows}`` as a plain python dict."""
    rows = df.groupBy(label_col).agg(F.count(F.lit(1)).alias("n")).collect()
    return {row[label_col]: int(row["n"]) for row in rows}


def class_weights(df: DataFrame, label_col: str = "label") -> Dict[int, float]:
    """Compute balanced class weights, ``n_total / (n_classes * n_class)``.

    This is the same formula as scikit-learn's ``class_weight="balanced"``, so the
    weights transfer directly to any downstream framework.

    Args:
        df: DataFrame to measure.
        label_col: Label column.

    Returns:
        ``{label: weight}``, where the majority class gets a weight below 1.
    """
    counts = class_counts(df, label_col)
    total = sum(counts.values())
    n_classes = len(counts)
    return {label: total / (n_classes * count) for label, count in counts.items()}


def add_class_weight(
    df: DataFrame,
    weights: Optional[Mapping[int, float]] = None,
    label_col: str = "label",
    output_col: str = "class_weight",
) -> DataFrame:
    """Append a per-row weight column usable as ``weightCol`` by Spark ML models.

    Weighting is preferred over resampling as the default remedy for the
    imbalance here: it changes no data, keeps the evaluation set honest, and is
    accepted by ``LogisticRegression``, ``RandomForestClassifier`` and
    ``GBTClassifier`` alike.
    """
    weights = weights or class_weights(df, label_col)
    mapping = F.create_map(
        *[item for label, weight in weights.items() for item in (F.lit(label), F.lit(float(weight)))]
    )
    return df.withColumn(output_col, mapping[F.col(label_col)])


def resample(
    df: DataFrame,
    strategy: str = "undersample",
    label_col: str = "label",
    seed: int = 42,
) -> DataFrame:
    """Rebalance the classes by sampling.

    Args:
        df: DataFrame to rebalance.
        strategy: ``"undersample"`` shrinks every class to the minority size;
            ``"oversample"`` replicates minority rows up to the majority size.
        label_col: Label column.
        seed: Random seed.

    Returns:
        The rebalanced DataFrame.

    Raises:
        ValueError: If ``strategy`` is unknown.

    Warning:
        Only ever apply this to the training split. Resampling a validation or
        test set silently changes the prior and makes every metric optimistic.
    """
    counts = class_counts(df, label_col)
    if not counts:
        return df

    if strategy == "undersample":
        target = min(counts.values())
        fractions = {label: min(1.0, target / count) for label, count in counts.items()}
        return df.sampleBy(label_col, fractions=fractions, seed=seed)

    if strategy == "oversample":
        target = max(counts.values())
        repeats = F.create_map(
            *[
                item
                for label, count in counts.items()
                for item in (F.lit(label), F.lit(int(math.ceil(target / count))))
            ]
        )
        expanded = df.withColumn(
            "__copy", F.explode(F.array_repeat(F.lit(1), repeats[F.col(label_col)]))
        ).drop("__copy")
        expanded_counts = {
            label: int(math.ceil(target / count)) * count for label, count in counts.items()
        }
        fractions = {label: min(1.0, target / count) for label, count in expanded_counts.items()}
        return expanded.sampleBy(label_col, fractions=fractions, seed=seed)

    raise ValueError(f"Unknown strategy {strategy!r}; expected 'undersample' or 'oversample'")


# ------------------------------------------------------------------- pipeline


def build_dataset(
    beats: DataFrame,
    cfg: Config,
    source: str = "mitbih",
    scaler: Optional[str] = "standard",
    descriptor_columns: Optional[Iterable[str]] = None,
    write: bool = True,
    dataset_name: str = FEATURES_DATASET,
) -> tuple[DataFrame, PipelineModel]:
    """Produce the model-ready dataset for one collection.

    Steps: restrict to the collection, create the ``train``/``val``/``test``
    assignment, fit the preprocessing pipeline **on the training split only**,
    transform everything, attach class weights, and persist.

    Fitting the scaler on the training split alone is the detail that keeps this
    honest: fitting it on the full frame would leak validation and test statistics
    into the features.

    Args:
        beats: Canonical beats DataFrame.
        cfg: Project configuration.
        source: Collection to build (``mitbih`` or ``ptbdb``).
        scaler: Scaler passed to :func:`build_preprocessing_pipeline`.
        descriptor_columns: Descriptors appended to the waveform vector.
        write: Persist the result to Parquet and the pipeline to disk.
        dataset_name: Parquet directory name; the collection is appended.

    Returns:
        A ``(dataset, fitted_pipeline)`` tuple.
    """
    subset = beats.where(F.col("source") == source)

    if source == "mitbih":
        assigned = holdout_validation(
            subset, val_fraction=cfg.val_fraction, output_col="split_final", seed=cfg.seed
        )
    else:
        assigned = stratified_split(
            subset, fractions=DEFAULT_SPLIT_FRACTIONS, output_col="split_final", seed=cfg.seed
        )

    assigned = assigned.cache()
    train = assigned.where(F.col("split_final") == "train")

    pipeline = build_preprocessing_pipeline(descriptor_columns=descriptor_columns, scaler=scaler)
    model = pipeline.fit(train)

    transformed = model.transform(assigned)
    weighted = add_class_weight(
        transformed.where(F.col("split_final") == "train")
    ).select("label", "class_weight").dropDuplicates(["label"])
    dataset = transformed.join(F.broadcast(weighted), on="label", how="left")

    if write:
        cfg.ensure_dirs()
        destination = cfg.parquet_path(f"{dataset_name}_{source}")
        dataset.write.mode("overwrite").partitionBy("split_final").parquet(str(destination))
        model.write().overwrite().save(str(cfg.parquet_path(f"{PIPELINE_DIR}_{source}")))
        dataset = dataset.sparkSession.read.parquet(str(destination))

    return dataset, model


def split_summary(df: DataFrame, split_col: str = "split_final") -> pd.DataFrame:
    """Return per-split, per-class counts and shares as a pandas table."""
    counts = df.groupBy(split_col, "label", "label_name").agg(F.count(F.lit(1)).alias("n_beats"))
    window = Window.partitionBy(split_col)
    counts = counts.withColumn("pct", F.round(F.col("n_beats") * 100.0 / F.sum("n_beats").over(window), 3))
    return counts.orderBy(split_col, "label").toPandas()


def load_dataset(
    spark, cfg: Config, source: str = "mitbih", dataset_name: str = FEATURES_DATASET
) -> DataFrame:
    """Read a dataset previously written by :func:`build_dataset`."""
    return spark.read.parquet(str(cfg.parquet_path(f"{dataset_name}_{source}")))


def load_pipeline(cfg: Config, source: str = "mitbih") -> PipelineModel:
    """Read the fitted pipeline previously saved by :func:`build_dataset`."""
    return PipelineModel.load(str(cfg.parquet_path(f"{PIPELINE_DIR}_{source}")))
