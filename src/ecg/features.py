"""Per-beat descriptors computed entirely with Spark array expressions.

Every function here returns a :class:`~pyspark.sql.Column`, so the descriptors are
evaluated inside the JVM using higher-order array functions
(``transform``/``aggregate``/``filter``/``zip_with``). Nothing is collected to the
driver and no Python UDF is involved, which is what keeps the whole EDA
distributed and Databricks-friendly.

The dataset authors right-pad every beat with zeros up to 187 samples. All
amplitude statistics are therefore computed over the *effective* signal — the
prefix that precedes the trailing zero padding — otherwise the padding would drag
every mean toward zero and make short beats look flatter than they are.
"""

from __future__ import annotations

from typing import List

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from ecg.schema import N_SAMPLES, SAMPLING_RATE_HZ

#: Descriptor columns added by :func:`add_beat_features`, in report order.
FEATURE_COLUMNS: List[str] = [
    "signal_length",
    "duration_s",
    "padding_ratio",
    "n_zeros",
    "amp_min",
    "amp_max",
    "amp_range",
    "amp_mean",
    "amp_std",
    "energy",
    "rms",
    "peak_index",
    "peak_time_s",
    "mean_abs_diff",
    "max_abs_diff",
]

#: Descriptors kept as model inputs alongside the raw waveform.
#:
#: This is a subset of :data:`FEATURE_COLUMNS`, pruned by the correlation
#: analysis in ``notebooks/02_eda_mitbih.ipynb``:
#:
#: * ``duration_s``, ``padding_ratio`` and ``n_zeros`` are perfectly collinear
#:   with ``signal_length`` (|r| = 1.000) — one of the four is enough;
#: * ``rms`` correlates 0.990 with ``amp_mean`` and ``energy`` 0.893, so both are
#:   dropped in favour of ``amp_mean``;
#: * ``amp_max`` is constant at 1.0 for 99.8% of the beats and ``amp_range`` is
#:   simply ``1 - amp_min``, so neither carries independent information.
MODEL_FEATURE_COLUMNS: List[str] = [
    "signal_length",
    "amp_min",
    "amp_mean",
    "amp_std",
    "peak_index",
    "mean_abs_diff",
    "max_abs_diff",
]


def _signal(signal: Column | str = "signal") -> Column:
    """Coerce a column name or Column into a Column."""
    return F.col(signal) if isinstance(signal, str) else signal


def effective_length(signal: Column | str = "signal") -> Column:
    """Return the number of samples before the trailing zero padding.

    Implemented as ``max(i + 1 for i, x in enumerate(signal) if x != 0)``: the
    index of the last non-zero sample, plus one.

    Args:
        signal: The ``array<double>`` signal column.

    Returns:
        An integer Column in ``[0, 187]``.

    Note:
        Because each beat was min-max normalised, its true minimum is exactly
        ``0.0``. If that minimum happens to fall on the final sample of the real
        beat, this heuristic under-counts by one sample (8 ms). That is the only
        failure mode, and it is measured in
        :func:`ecg.eda.padding_report`.
    """
    sig = _signal(signal)
    indexed = F.transform(sig, lambda x, i: F.when(x != F.lit(0.0), i + 1).otherwise(F.lit(0)))
    return F.aggregate(indexed, F.lit(0), lambda acc, v: F.greatest(acc, v)).cast("int")


def valid_signal(signal: Column | str = "signal", length: Column | None = None) -> Column:
    """Return the beat with its trailing zero padding removed.

    Args:
        signal: The ``array<double>`` signal column.
        length: Pre-computed effective length; recomputed when omitted.

    Returns:
        An ``array<double>`` Column holding only the real samples.
    """
    sig = _signal(signal)
    length = effective_length(sig) if length is None else length
    return F.slice(sig, F.lit(1), length)


def _array_sum(arr: Column) -> Column:
    """Sum of an ``array<double>``."""
    return F.aggregate(arr, F.lit(0.0), lambda acc, x: acc + x)


def _array_sum_squares(arr: Column) -> Column:
    """Sum of squares of an ``array<double>``."""
    return F.aggregate(arr, F.lit(0.0), lambda acc, x: acc + x * x)


def _safe_divide(numerator: Column, denominator: Column) -> Column:
    """Divide, returning ``NULL`` instead of failing on an empty array."""
    return F.when(denominator > 0, numerator / denominator)


def array_mean(arr: Column) -> Column:
    """Arithmetic mean of an ``array<double>``, ``NULL`` when empty."""
    return _safe_divide(_array_sum(arr), F.size(arr))


def array_std(arr: Column) -> Column:
    """Population standard deviation of an ``array<double>``.

    Uses the ``E[x^2] - E[x]^2`` identity in a single pass and clamps the variance
    at zero so floating-point noise cannot produce ``NaN`` from ``sqrt``.
    """
    n = F.size(arr)
    mean = _safe_divide(_array_sum(arr), n)
    mean_sq = _safe_divide(_array_sum_squares(arr), n)
    return F.sqrt(F.greatest(mean_sq - mean * mean, F.lit(0.0)))


def absolute_differences(arr: Column) -> Column:
    """Return ``|x[i+1] - x[i]|`` for consecutive samples.

    A cheap proxy for slew rate: ventricular and fusion beats carry steeper,
    wider deflections than normal beats, which shows up here without any
    filtering or peak detection.
    """
    n = F.size(arr)
    head = F.slice(arr, F.lit(1), n - 1)
    tail = F.slice(arr, F.lit(2), n - 1)
    return F.zip_with(head, tail, lambda a, b: F.abs(b - a))


def signal_fingerprint(signal: Column | str = "signal") -> Column:
    """Return a SHA-256 fingerprint of the full waveform, for duplicate detection."""
    return F.sha2(_signal(signal).cast("string"), 256)


def add_beat_features(df: DataFrame, signal_col: str = "signal") -> DataFrame:
    """Append every per-beat descriptor in :data:`FEATURE_COLUMNS` to ``df``.

    Args:
        df: A canonical beats DataFrame (see :func:`ecg.schema.canonical_schema`).
        signal_col: Name of the ``array<double>`` column.

    Returns:
        The same DataFrame with the descriptor columns appended.

    Example:
        >>> beats = add_beat_features(load_beats(spark, cfg))     # doctest: +SKIP
        >>> beats.select("label_name", "signal_length").show(3)   # doctest: +SKIP
    """
    sig = F.col(signal_col)
    length = effective_length(sig)

    out = df.withColumn("signal_length", length)

    valid = valid_signal(sig, F.col("signal_length"))
    diffs = absolute_differences(valid)

    return (
        out.withColumn("duration_s", F.col("signal_length") / F.lit(float(SAMPLING_RATE_HZ)))
        .withColumn(
            "padding_ratio",
            (F.lit(N_SAMPLES) - F.col("signal_length")) / F.lit(float(N_SAMPLES)),
        )
        .withColumn("n_zeros", F.size(F.filter(sig, lambda x: x == F.lit(0.0))))
        .withColumn("amp_min", F.array_min(valid))
        .withColumn("amp_max", F.array_max(valid))
        .withColumn("amp_range", F.col("amp_max") - F.col("amp_min"))
        .withColumn("amp_mean", array_mean(valid))
        .withColumn("amp_std", array_std(valid))
        .withColumn("energy", _array_sum_squares(valid))
        .withColumn("rms", F.sqrt(_safe_divide(F.col("energy"), F.size(valid))))
        .withColumn(
            "peak_index",
            (F.array_position(sig, F.array_max(valid)) - F.lit(1)).cast("int"),
        )
        .withColumn("peak_time_s", F.col("peak_index") / F.lit(float(SAMPLING_RATE_HZ)))
        .withColumn("mean_abs_diff", array_mean(diffs))
        .withColumn("max_abs_diff", F.array_max(diffs))
    )


def add_fingerprint(df: DataFrame, signal_col: str = "signal") -> DataFrame:
    """Append a ``signal_hash`` column used by the duplicate-beat checks."""
    return df.withColumn("signal_hash", signal_fingerprint(signal_col))
