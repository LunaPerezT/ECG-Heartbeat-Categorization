"""Distributed exploratory analysis.

Every function follows the same contract: the aggregation runs in Spark over the
full dataset, and only the (small) result is brought back to the driver as a
pandas DataFrame ready to be displayed, saved to ``reports/tables/`` or handed to
:mod:`ecg.viz`. No function here collects raw beats except
:func:`sample_beats`, which is explicitly a sampler.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.ml.stat import Correlation, Summarizer
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ecg.features import FEATURE_COLUMNS, add_beat_features, add_fingerprint
from ecg.schema import LABEL_ORDER, N_SAMPLES, SAMPLING_RATE_HZ, describe_label

#: Default quantiles reported for every descriptor.
DEFAULT_QUANTILES: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Relative error used by ``percentile_approx``; 1e-3 is exact enough for plots
#: and roughly an order of magnitude cheaper than an exact percentile.
PERCENTILE_ACCURACY: int = 1000


# --------------------------------------------------------------------- shape


def dataset_overview(df: DataFrame) -> pd.DataFrame:
    """Return one row per ``(source, split)`` with size and class count.

    Args:
        df: Canonical beats DataFrame.

    Returns:
        Columns ``source``, ``split``, ``n_beats``, ``n_classes``, ``pct_of_total``.
    """
    agg = (
        df.groupBy("source", "split")
        .agg(F.count(F.lit(1)).alias("n_beats"), F.countDistinct("label").alias("n_classes"))
        .orderBy("source", "split")
        .toPandas()
    )
    total = agg["n_beats"].sum()
    agg["pct_of_total"] = (agg["n_beats"] / total * 100).round(2)
    return agg


def class_distribution(
    df: DataFrame,
    by: Sequence[str] = ("source", "split"),
    add_description: bool = True,
) -> pd.DataFrame:
    """Return the class histogram, with percentages within each group.

    Args:
        df: Canonical beats DataFrame.
        by: Grouping columns evaluated before the label.
        add_description: Append the clinical meaning of each class.

    Returns:
        Columns ``*by``, ``label``, ``label_name``, ``n_beats``, ``pct``
        (share within the group) and optionally ``description``.
    """
    group_cols: List[str] = list(by) + ["label", "label_name"]
    counts = df.groupBy(*group_cols).agg(F.count(F.lit(1)).alias("n_beats"))

    if by:
        window = Window.partitionBy(*by)
        counts = counts.withColumn("pct", F.col("n_beats") * 100.0 / F.sum("n_beats").over(window))
    else:
        counts = counts.withColumn(
            "pct", F.col("n_beats") * 100.0 / F.sum("n_beats").over(Window.partitionBy())
        )

    pdf = counts.orderBy(*by, "label").toPandas()
    pdf["pct"] = pdf["pct"].round(3)
    if add_description and "source" in pdf.columns:
        pdf["description"] = [
            describe_label(src, name) for src, name in zip(pdf["source"], pdf["label_name"])
        ]
    return pdf


def imbalance_summary(distribution: pd.DataFrame, by: Sequence[str] = ("source",)) -> pd.DataFrame:
    """Summarise class imbalance from the output of :func:`class_distribution`.

    Args:
        distribution: Output of :func:`class_distribution`.
        by: Grouping columns to summarise over.

    Returns:
        Columns ``*by``, ``majority_class``, ``minority_class``, ``majority_pct``,
        ``minority_pct``, ``imbalance_ratio``.
    """
    rows = []
    grouped = distribution.groupby(list(by), dropna=False) if by else [((), distribution)]
    for key, chunk in grouped:
        totals = chunk.groupby("label_name", as_index=False)["n_beats"].sum()
        total = totals["n_beats"].sum()
        majority = totals.loc[totals["n_beats"].idxmax()]
        minority = totals.loc[totals["n_beats"].idxmin()]
        record: Dict[str, object] = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        record.update(
            {
                "n_beats": int(total),
                "n_classes": int(totals.shape[0]),
                "majority_class": majority["label_name"],
                "majority_pct": round(float(majority["n_beats"]) / total * 100, 2),
                "minority_class": minority["label_name"],
                "minority_pct": round(float(minority["n_beats"]) / total * 100, 2),
                "imbalance_ratio": round(float(majority["n_beats"]) / float(minority["n_beats"]), 1),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- quality


def quality_report(df: DataFrame, signal_col: str = "signal") -> pd.DataFrame:
    """Run the integrity checks that decide whether cleaning is needed at all.

    The checks are deliberately expressed as a single aggregation so the whole
    report costs one pass over the data.

    Args:
        df: Canonical beats DataFrame.
        signal_col: Name of the signal column.

    Returns:
        Columns ``check``, ``n_beats`` (how many rows fail or match) and ``pct``.
    """
    sig = F.col(signal_col)

    checks = {
        "rows_total": F.count(F.lit(1)),
        "null_signal": F.sum(F.when(sig.isNull(), 1).otherwise(0)),
        "wrong_length": F.sum(F.when(F.size(sig) != N_SAMPLES, 1).otherwise(0)),
        "null_label": F.sum(F.when(F.col("label").isNull(), 1).otherwise(0)),
        "contains_null_sample": F.sum(
            F.when(F.size(F.filter(sig, lambda x: x.isNull())) > 0, 1).otherwise(0)
        ),
        "contains_nan_sample": F.sum(
            F.when(F.size(F.filter(sig, lambda x: F.isnan(x))) > 0, 1).otherwise(0)
        ),
        "below_zero": F.sum(
            F.when(F.size(F.filter(sig, lambda x: x < F.lit(0.0))) > 0, 1).otherwise(0)
        ),
        "above_one": F.sum(
            F.when(F.size(F.filter(sig, lambda x: x > F.lit(1.0))) > 0, 1).otherwise(0)
        ),
        "all_zero_beat": F.sum(F.when(F.array_max(sig) == F.lit(0.0), 1).otherwise(0)),
        "max_not_one": F.sum(F.when(F.abs(F.array_max(sig) - F.lit(1.0)) > 1e-9, 1).otherwise(0)),
        "min_not_zero": F.sum(F.when(F.abs(F.array_min(sig) - F.lit(0.0)) > 1e-9, 1).otherwise(0)),
    }

    row = df.agg(*[expr.alias(name) for name, expr in checks.items()]).toPandas().iloc[0]
    total = int(row["rows_total"])
    records = [
        {"check": name, "n_beats": int(row[name]), "pct": round(int(row[name]) / total * 100, 4)}
        for name in checks
    ]
    return pd.DataFrame(records)


def duplicate_report(df: DataFrame, by: Sequence[str] = ("source",)) -> pd.DataFrame:
    """Count exactly duplicated waveforms, and how many carry conflicting labels.

    Duplicated segments are expected in this dataset: consecutive beats from the
    same recording can be identical after cropping and quantisation. What would
    be a real problem is the same waveform appearing with two different labels, or
    a beat shared between the train and test splits, so both are measured here.

    Args:
        df: Canonical beats DataFrame.
        by: Grouping columns for the report.

    Returns:
        Columns ``*by``, ``n_beats``, ``n_unique_waveforms``, ``n_duplicated_beats``,
        ``pct_duplicated``, ``n_waveforms_with_conflicting_labels``,
        ``n_waveforms_in_both_splits``.
    """
    hashed = add_fingerprint(df)
    group_cols = list(by)

    per_hash = hashed.groupBy(*group_cols, "signal_hash").agg(
        F.count(F.lit(1)).alias("n_occurrences"),
        F.countDistinct("label").alias("n_labels"),
        F.countDistinct("split").alias("n_splits"),
    )

    summary = per_hash.groupBy(*group_cols).agg(
        F.sum("n_occurrences").alias("n_beats"),
        F.count(F.lit(1)).alias("n_unique_waveforms"),
        F.sum(F.when(F.col("n_occurrences") > 1, F.col("n_occurrences") - 1).otherwise(0)).alias(
            "n_duplicated_beats"
        ),
        F.sum(F.when(F.col("n_labels") > 1, 1).otherwise(0)).alias(
            "n_waveforms_with_conflicting_labels"
        ),
        F.sum(F.when(F.col("n_splits") > 1, 1).otherwise(0)).alias("n_waveforms_in_both_splits"),
    )

    pdf = summary.orderBy(*group_cols).toPandas() if group_cols else summary.toPandas()
    pdf["pct_duplicated"] = (pdf["n_duplicated_beats"] / pdf["n_beats"] * 100).round(3)
    return pdf


def padding_report(df: DataFrame, by: Sequence[str] = ("source", "label_name")) -> pd.DataFrame:
    """Describe the zero padding: how much of each beat is real signal.

    Args:
        df: Canonical beats DataFrame (features are added if missing).
        by: Grouping columns.

    Returns:
        Columns ``*by``, ``n_beats``, ``min_length``, ``median_length``,
        ``max_length``, ``mean_length``, ``pct_unpadded``, ``mean_duration_s``
        and ``pct_ambiguous_tail`` — beats whose last real sample is below 0.01,
        where the padding boundary is intrinsically fuzzy.
    """
    featured = df if "signal_length" in df.columns else add_beat_features(df)
    last_value = F.element_at(F.col("signal"), F.col("signal_length"))

    agg = (
        featured.groupBy(*by)
        .agg(
            F.count(F.lit(1)).alias("n_beats"),
            F.min("signal_length").alias("min_length"),
            F.expr(f"percentile_approx(signal_length, 0.5, {PERCENTILE_ACCURACY})").alias(
                "median_length"
            ),
            F.max("signal_length").alias("max_length"),
            F.round(F.avg("signal_length"), 1).alias("mean_length"),
            F.round(F.avg("duration_s"), 4).alias("mean_duration_s"),
            F.sum(F.when(F.col("signal_length") == F.lit(N_SAMPLES), 1).otherwise(0)).alias(
                "n_unpadded"
            ),
            F.sum(F.when(last_value < F.lit(0.01), 1).otherwise(0)).alias("n_ambiguous_tail"),
        )
        .orderBy(*by)
        .toPandas()
    )
    agg["pct_unpadded"] = (agg["n_unpadded"] / agg["n_beats"] * 100).round(2)
    agg["pct_ambiguous_tail"] = (agg["n_ambiguous_tail"] / agg["n_beats"] * 100).round(2)
    return agg.drop(columns=["n_unpadded", "n_ambiguous_tail"])


# ---------------------------------------------------------------- descriptors


def feature_summary(
    df: DataFrame,
    features: Optional[Iterable[str]] = None,
    by: Sequence[str] = ("source", "label_name"),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Return a tidy describe-style table for the per-beat descriptors.

    Args:
        df: Canonical beats DataFrame (features are added if missing).
        features: Descriptors to summarise; defaults to :data:`FEATURE_COLUMNS`.
        by: Grouping columns.
        quantiles: Quantiles reported per descriptor.

    Returns:
        Long-format table with columns ``*by``, ``feature``, ``n``, ``mean``,
        ``std``, ``min``, ``q05`` .. ``q95``, ``max``.
    """
    featured = df if "signal_length" in df.columns else add_beat_features(df)
    features = list(features or FEATURE_COLUMNS)

    aggregations = [F.count(F.lit(1)).alias("n")]
    for feature in features:
        aggregations += [
            F.avg(feature).alias(f"{feature}__mean"),
            F.stddev(feature).alias(f"{feature}__std"),
            F.min(feature).alias(f"{feature}__min"),
            F.max(feature).alias(f"{feature}__max"),
        ]
        quantile_list = ", ".join(str(q) for q in quantiles)
        aggregations.append(
            F.expr(
                f"percentile_approx({feature}, array({quantile_list}), {PERCENTILE_ACCURACY})"
            ).alias(f"{feature}__quantiles")
        )

    wide = featured.groupBy(*by).agg(*aggregations).orderBy(*by).toPandas()

    records = []
    for _, row in wide.iterrows():
        for feature in features:
            record: Dict[str, object] = {col: row[col] for col in by}
            record["feature"] = feature
            record["n"] = int(row["n"])
            record["mean"] = row[f"{feature}__mean"]
            record["std"] = row[f"{feature}__std"]
            record["min"] = row[f"{feature}__min"]
            for q, value in zip(quantiles, row[f"{feature}__quantiles"]):
                record[f"q{int(round(q * 100)):02d}"] = value
            record["max"] = row[f"{feature}__max"]
            records.append(record)

    numeric = ["mean", "std", "min", "max"] + [f"q{int(round(q * 100)):02d}" for q in quantiles]
    out = pd.DataFrame(records)
    out[numeric] = out[numeric].astype(float).round(4)
    return out


def feature_correlation(
    df: DataFrame,
    features: Optional[Iterable[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return the correlation matrix of the descriptors plus the label.

    Args:
        df: Canonical beats DataFrame (features are added if missing).
        features: Descriptors to include; defaults to :data:`FEATURE_COLUMNS`.
        method: ``"pearson"`` or ``"spearman"``.

    Returns:
        A square pandas DataFrame indexed and columned by feature name.
    """
    featured = df if "signal_length" in df.columns else add_beat_features(df)
    columns = list(features or FEATURE_COLUMNS) + ["label"]

    prepared = featured.select(*[F.col(c).cast("double").alias(c) for c in columns]).na.drop()
    assembled = VectorAssembler(inputCols=columns, outputCol="__features").transform(prepared)
    matrix = Correlation.corr(assembled, "__features", method).head()[0].toArray()
    return pd.DataFrame(matrix, index=columns, columns=columns).round(3)


# ----------------------------------------------------------------- waveforms


def waveform_profile(
    df: DataFrame,
    by: Sequence[str] = ("label_name",),
    exclude_padding: bool = True,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    min_support_ratio: float = 0.05,
) -> pd.DataFrame:
    """Return the average beat morphology per group, sample by sample.

    The signal array is exploded into ``(t, amplitude)`` pairs and aggregated per
    ``(group, t)``. With 109k beats that is ~20 M rows through one shuffle — a
    genuine distributed aggregation, and the reason this is worth doing in Spark
    rather than in pandas on a sample.

    Args:
        df: Canonical beats DataFrame.
        by: Grouping columns.
        exclude_padding: Drop samples beyond each beat's effective length so the
            trailing zeros do not bias the mean toward zero.
        quantiles: Quantiles computed per sample index.
        min_support_ratio: Drop sample indices supported by fewer than this share
            of the group's beats. Without it the tail of the profile is computed
            from the handful of unusually long beats that reach that far, which
            reads as noise rather than morphology.

    Returns:
        Columns ``*by``, ``t`` (sample index), ``time_s``, ``n``, ``mean``,
        ``std``, and one column per quantile (``q25``, ``q50``, ...).
    """
    featured = df if "signal_length" in df.columns else add_beat_features(df)

    exploded = featured.select(
        *by,
        "signal_length",
        F.posexplode("signal").alias("t", "amplitude"),
    )
    if exclude_padding:
        exploded = exploded.where(F.col("t") < F.col("signal_length"))

    quantile_list = ", ".join(str(q) for q in quantiles)
    agg = (
        exploded.groupBy(*by, "t")
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.avg("amplitude").alias("mean"),
            F.stddev("amplitude").alias("std"),
            F.expr(
                f"percentile_approx(amplitude, array({quantile_list}), {PERCENTILE_ACCURACY})"
            ).alias("__quantiles"),
        )
        .orderBy(*by, "t")
        .toPandas()
    )

    for index, q in enumerate(quantiles):
        agg[f"q{int(round(q * 100)):02d}"] = agg["__quantiles"].map(lambda v, i=index: v[i])
    agg = agg.drop(columns="__quantiles")

    if min_support_ratio > 0 and not agg.empty:
        group_max = agg.groupby(list(by))["n"].transform("max")
        agg = agg[agg["n"] >= group_max * min_support_ratio].reset_index(drop=True)

    agg["time_s"] = agg["t"] / SAMPLING_RATE_HZ
    return agg


def mean_waveform_fast(df: DataFrame, by: str = "label_name") -> pd.DataFrame:
    """Per-group mean and standard deviation of the padded beat, in one pass.

    Uses :class:`pyspark.ml.stat.Summarizer` over the whole 187-dimensional
    vector, which avoids the explode/shuffle of :func:`waveform_profile`. The
    trade-off is that the trailing zero padding is included in the average.

    Args:
        df: Canonical beats DataFrame.
        by: Single grouping column.

    Returns:
        Columns ``by``, ``t``, ``time_s``, ``mean``, ``std``.
    """
    vectorised = df.select(by, array_to_vector("signal").alias("vec"))
    summary = vectorised.groupBy(by).agg(
        Summarizer.metrics("mean", "std").summary(F.col("vec")).alias("summary"),
        F.count(F.lit(1)).alias("n"),
    )
    collected = summary.select(by, "n", "summary.mean", "summary.std").toPandas()

    records = []
    for _, row in collected.iterrows():
        for t, (mean, std) in enumerate(zip(row["mean"], row["std"])):
            records.append(
                {
                    by: row[by],
                    "t": t,
                    "time_s": t / SAMPLING_RATE_HZ,
                    "mean": float(mean),
                    "std": float(std),
                    "n": int(row["n"]),
                }
            )
    return pd.DataFrame(records)


def sample_beats(
    df: DataFrame,
    n_per_group: int = 6,
    by: str = "label_name",
    seed: int = 42,
) -> pd.DataFrame:
    """Collect a reproducible random sample of individual beats for plotting.

    Args:
        df: Canonical beats DataFrame.
        n_per_group: Beats drawn per group.
        by: Grouping column.
        seed: Random seed, so the same beats are drawn on every run.

    Returns:
        Columns ``beat_id``, ``by``, ``signal_length``, ``signal`` (python list).
    """
    featured = df if "signal_length" in df.columns else add_beat_features(df)
    window = Window.partitionBy(by).orderBy(F.rand(seed))
    picked = (
        featured.withColumn("__rank", F.row_number().over(window))
        .where(F.col("__rank") <= n_per_group)
        .select("beat_id", by, "signal_length", "signal", "__rank")
        .orderBy(by, "__rank")
        .drop("__rank")
    )
    return picked.toPandas()


def length_histogram(
    df: DataFrame,
    by: Sequence[str] = ("label_name",),
    bin_width: int = 5,
) -> pd.DataFrame:
    """Return a binned histogram of effective beat length.

    Args:
        df: Canonical beats DataFrame.
        by: Grouping columns.
        bin_width: Width of each bin, in samples.

    Returns:
        Columns ``*by``, ``bin_start``, ``bin_end``, ``n_beats``, ``pct``.
    """
    featured = df if "signal_length" in df.columns else add_beat_features(df)
    binned = featured.withColumn(
        "bin_start", (F.floor(F.col("signal_length") / bin_width) * bin_width).cast("int")
    )
    counts = binned.groupBy(*by, "bin_start").agg(F.count(F.lit(1)).alias("n_beats"))
    window = Window.partitionBy(*by)
    counts = counts.withColumn("pct", F.col("n_beats") * 100.0 / F.sum("n_beats").over(window))
    pdf = counts.orderBy(*by, "bin_start").toPandas()
    pdf["bin_end"] = pdf["bin_start"] + bin_width
    pdf["pct"] = pdf["pct"].round(3)
    return pdf


def ordered_labels(source: str, present: Optional[Iterable[str]] = None) -> List[str]:
    """Return the canonical plot order of a collection's class names."""
    order = LABEL_ORDER[source]
    if present is None:
        return list(order)
    present_set = set(present)
    return [name for name in order if name in present_set]
