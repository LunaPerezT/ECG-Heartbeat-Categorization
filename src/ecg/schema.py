"""Schema and label metadata for the ECG Heartbeat Categorization dataset.

Every raw CSV in the dataset is head-less and holds exactly ``N_COLUMNS`` (188)
comma-separated float columns per row: ``N_SAMPLES`` (187) amplitude samples of a
single, already segmented heartbeat, followed by the class label.

The signals were min-max normalised to ``[0, 1]`` by the dataset authors, cropped
to a single beat, downsampled to 125 Hz and right-padded with zeros so that every
beat has the same length. That padding is what ``ecg.features.effective_length``
undoes at analysis time.
"""

from __future__ import annotations

from typing import Dict, List

from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

#: Number of amplitude samples stored per heartbeat (columns 0..186).
N_SAMPLES: int = 187

#: Total number of columns in every raw CSV: 187 samples + 1 label.
N_COLUMNS: int = N_SAMPLES + 1

#: Sampling frequency of the published signals, in hertz.
SAMPLING_RATE_HZ: int = 125

#: Duration of a single sample, in seconds.
SAMPLE_PERIOD_S: float = 1.0 / SAMPLING_RATE_HZ

#: Name of the raw label column once the CSV has been parsed.
RAW_LABEL_COLUMN: str = "label_raw"

#: Ordered names of the 187 raw amplitude columns: ``s000`` .. ``s186``.
SIGNAL_COLUMNS: List[str] = [f"s{i:03d}" for i in range(N_SAMPLES)]

#: AAMI-style class codes used by the MIT-BIH collection.
MITBIH_LABELS: Dict[int, str] = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

#: Human-readable description of each MIT-BIH class.
MITBIH_LABEL_DESCRIPTIONS: Dict[str, str] = {
    "N": "Normal beat (normal, left/right bundle branch block, atrial escape, nodal escape)",
    "S": "Supraventricular ectopic beat (atrial premature, aberrant atrial, nodal, supraventricular)",
    "V": "Ventricular ectopic beat (premature ventricular contraction, ventricular escape)",
    "F": "Fusion beat (fusion of ventricular and normal)",
    "Q": "Unknown / unclassifiable beat (paced, fusion of paced and normal, unclassified)",
}

#: Binary classes used by the PTB Diagnostic collection.
PTBDB_LABELS: Dict[int, str] = {0: "normal", 1: "abnormal"}

#: Description of each PTB class.
PTBDB_LABEL_DESCRIPTIONS: Dict[str, str] = {
    "normal": "Healthy control recording",
    "abnormal": "Recording from a patient diagnosed with myocardial infarction",
}

#: Label maps keyed by the ``source`` value used across the pipeline.
LABEL_MAPS: Dict[str, Dict[int, str]] = {
    "mitbih": MITBIH_LABELS,
    "ptbdb": PTBDB_LABELS,
}

#: Plot/report ordering for each collection's classes.
LABEL_ORDER: Dict[str, List[str]] = {
    "mitbih": ["N", "S", "V", "F", "Q"],
    "ptbdb": ["normal", "abnormal"],
}


def raw_schema() -> StructType:
    """Return the explicit schema of a raw dataset CSV.

    An explicit schema is used instead of ``inferSchema`` on purpose: inference
    would trigger a full extra pass over ~583 MB of CSV, and the layout is fixed
    and documented by the dataset authors.

    Returns:
        A :class:`~pyspark.sql.types.StructType` with 187 ``DoubleType`` sample
        columns followed by a ``DoubleType`` label column. The label is read as a
        double because the CSV stores it in scientific notation
        (``0.000000000000000000e+00``) and is cast to an integer downstream.
    """
    fields = [StructField(name, DoubleType(), nullable=True) for name in SIGNAL_COLUMNS]
    fields.append(StructField(RAW_LABEL_COLUMN, DoubleType(), nullable=True))
    return StructType(fields)


def canonical_schema() -> StructType:
    """Return the schema of the canonical, analysis-ready DataFrame.

    This is the contract every downstream module reads and writes:

    ==============  ====================  ==========================================
    Column          Type                  Meaning
    ==============  ====================  ==========================================
    ``beat_id``     ``string``            Stable identifier ``<source>_<split>_<n>``
    ``source``      ``string``            ``mitbih`` or ``ptbdb``
    ``split``       ``string``            ``train``/``test`` (MIT-BIH) or ``full``
    ``label``       ``int``               Numeric class as published
    ``label_name``  ``string``            ``N``/``S``/``V``/``F``/``Q`` or
                                          ``normal``/``abnormal``
    ``signal``      ``array<double>``     187 amplitude samples in acquisition order
    ==============  ====================  ==========================================

    Returns:
        The canonical :class:`~pyspark.sql.types.StructType`.
    """
    return StructType(
        [
            StructField("beat_id", StringType(), nullable=False),
            StructField("source", StringType(), nullable=False),
            StructField("split", StringType(), nullable=False),
            StructField("label", IntegerType(), nullable=False),
            StructField("label_name", StringType(), nullable=False),
            StructField("signal", ArrayType(DoubleType(), containsNull=True), nullable=False),
        ]
    )


def label_name_map(source: str) -> Dict[int, str]:
    """Return the ``label -> label_name`` mapping for a collection.

    Args:
        source: Either ``"mitbih"`` or ``"ptbdb"``.

    Returns:
        The mapping of numeric labels to class names.

    Raises:
        KeyError: If ``source`` is not a known collection.
    """
    try:
        return LABEL_MAPS[source]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown source {source!r}; expected one of {sorted(LABEL_MAPS)}"
        ) from exc


def describe_label(source: str, label_name: str) -> str:
    """Return the human-readable description of a class name."""
    if source == "mitbih":
        return MITBIH_LABEL_DESCRIPTIONS.get(label_name, "")
    if source == "ptbdb":
        return PTBDB_LABEL_DESCRIPTIONS.get(label_name, "")
    return ""
