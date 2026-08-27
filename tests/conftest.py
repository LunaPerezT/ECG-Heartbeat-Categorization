"""Shared pytest fixtures: one local Spark session and small synthetic datasets."""

from __future__ import annotations

import math
from typing import List

import pytest
from pyspark.sql import SparkSession

from ecg.schema import N_SAMPLES, canonical_schema


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """A minimal local Spark session shared by the whole test session."""
    session = (
        SparkSession.builder.appName("ecg-tests")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def pad(prefix: List[float]) -> List[float]:
    """Right-pad a prefix with zeros up to the fixed 187-sample length."""
    if len(prefix) > N_SAMPLES:
        raise ValueError("prefix longer than a beat")
    return list(prefix) + [0.0] * (N_SAMPLES - len(prefix))


#: A beat whose descriptors are known exactly; see ``test_features``.
KNOWN_PREFIX: List[float] = [1.0, 0.5, 0.0, 0.25]

KNOWN_EXPECTED = {
    "signal_length": 4,
    "amp_min": 0.0,
    "amp_max": 1.0,
    "amp_mean": 0.4375,
    "amp_std": math.sqrt((1.0 + 0.25 + 0.0 + 0.0625) / 4 - 0.4375**2),
    "energy": 1.3125,
    "rms": math.sqrt(1.3125 / 4),
    "peak_index": 0,
    "mean_abs_diff": (0.5 + 0.5 + 0.25) / 3,
    "max_abs_diff": 0.5,
    "n_zeros": N_SAMPLES - 3,  # one interior zero plus 183 padding zeros
}


@pytest.fixture(scope="session")
def known_beat(spark: SparkSession):
    """A single-row canonical DataFrame holding :data:`KNOWN_PREFIX`."""
    return spark.createDataFrame(
        [("b0", "mitbih", "train", 0, "N", pad(KNOWN_PREFIX))], schema=canonical_schema()
    )


@pytest.fixture(scope="session")
def synthetic_beats(spark: SparkSession):
    """A small, deliberately imbalanced canonical DataFrame.

    Class ``N`` gets 60 beats, ``S`` 20, ``V`` 12, ``F`` 4 and ``Q`` 4 — enough to
    exercise the stratified split and the class-weight maths without needing the
    real 583 MB dataset.
    """
    plan = [(0, "N", 60), (1, "S", 20), (2, "V", 12), (3, "F", 4), (4, "Q", 4)]
    rows = []
    counter = 0
    for label, name, count in plan:
        for index in range(count):
            length = 20 + label * 10 + (index % 5)
            amplitude = 0.2 + 0.1 * label
            prefix = [1.0] + [amplitude + 0.001 * index] * (length - 2) + [0.5]
            split = "train" if index % 5 != 0 else "test"
            rows.append((f"b{counter:04d}", "mitbih", split, label, name, pad(prefix)))
            counter += 1
    return spark.createDataFrame(rows, schema=canonical_schema())
