"""Schema and label-metadata contracts."""

from __future__ import annotations

import pytest

from ecg.schema import (
    LABEL_ORDER,
    MITBIH_LABELS,
    MITBIH_LABEL_DESCRIPTIONS,
    N_COLUMNS,
    N_SAMPLES,
    PTBDB_LABELS,
    SAMPLING_RATE_HZ,
    SIGNAL_COLUMNS,
    canonical_schema,
    describe_label,
    label_name_map,
    raw_schema,
)


def test_raw_schema_has_188_columns():
    fields = raw_schema().fields
    assert len(fields) == N_COLUMNS == 188
    assert [field.name for field in fields[:-1]] == SIGNAL_COLUMNS
    assert fields[-1].name == "label_raw"


def test_signal_columns_are_zero_padded_and_ordered():
    assert SIGNAL_COLUMNS[0] == "s000"
    assert SIGNAL_COLUMNS[-1] == f"s{N_SAMPLES - 1:03d}"
    assert SIGNAL_COLUMNS == sorted(SIGNAL_COLUMNS)


def test_canonical_schema_columns():
    names = [field.name for field in canonical_schema().fields]
    assert names == ["beat_id", "source", "split", "label", "label_name", "signal"]


def test_label_maps_are_complete():
    assert set(MITBIH_LABELS.values()) == set(LABEL_ORDER["mitbih"])
    assert set(MITBIH_LABEL_DESCRIPTIONS) == set(MITBIH_LABELS.values())
    assert set(PTBDB_LABELS.values()) == set(LABEL_ORDER["ptbdb"])
    assert label_name_map("mitbih")[3] == "F"


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        label_name_map("holter")


def test_describe_label_is_defined_for_every_class():
    for source, names in LABEL_ORDER.items():
        for name in names:
            assert describe_label(source, name), f"missing description for {source}/{name}"


def test_beat_duration_matches_sampling_rate():
    assert SAMPLING_RATE_HZ == 125
    assert N_SAMPLES / SAMPLING_RATE_HZ == pytest.approx(1.496, abs=1e-3)
