"""Evaluation metrics, checked against hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest

from ecg import metrics

LABELS = ["N", "S", "V"]


@pytest.fixture()
def perfect():
    y = np.array([0, 0, 1, 1, 2, 2])
    return y, y.copy()


def test_perfect_predictions_score_one(perfect):
    y_true, y_pred = perfect
    scores = metrics.summary_metrics(y_true, y_pred)
    assert scores["macro_f1"] == 1.0
    assert scores["balanced_accuracy"] == 1.0
    assert scores["accuracy"] == 1.0
    assert scores["mcc"] == 1.0


def test_confusion_matrix_is_labelled_and_square(perfect):
    y_true, y_pred = perfect
    matrix = metrics.confusion_matrix(y_true, y_pred, LABELS)
    assert list(matrix.index) == LABELS
    assert list(matrix.columns) == LABELS
    assert matrix.to_numpy().trace() == len(y_true)


def test_row_normalised_confusion_rows_sum_to_one():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    matrix = metrics.confusion_matrix(y_true, y_pred, LABELS, normalize="true")
    assert np.allclose(matrix.to_numpy().sum(axis=1), 1.0)


def test_per_class_report_matches_a_hand_computed_case():
    # Class N: 2 of 3 recalled, 2 of 2 predicted correctly.
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    report = metrics.per_class_report(y_true, y_pred, LABELS).set_index("class")

    assert report.loc["N", "recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert report.loc["N", "precision"] == pytest.approx(1.0)
    assert report.loc["N", "support"] == 3
    assert report.loc["S", "precision"] == pytest.approx(0.5)
    assert "macro avg" in report.index
    assert "weighted avg" in report.index


def test_macro_f1_weights_a_rare_class_as_heavily_as_a_common_one():
    """The reason macro-F1 is the project's primary metric."""
    y_true = np.array([0] * 98 + [1, 2])
    all_majority = np.zeros_like(y_true)

    scores = metrics.summary_metrics(y_true, all_majority, n_classes=3)
    assert scores["accuracy"] == pytest.approx(0.98)
    assert scores["macro_f1"] < 0.34  # one perfect class out of three


def test_majority_class_baseline_predicts_only_the_majority():
    y_true = np.array([0] * 8 + [1, 2])
    result = metrics.majority_class_baseline(y_true, LABELS)
    assert result["summary"]["model"] == "majority-class"
    assert result["summary"]["accuracy"] == pytest.approx(0.8)
    assert result["confusion"].loc["S", "N"] == 1


def test_binary_scores_include_roc_and_pr_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]])
    scores = metrics.summary_metrics(y_true, y_score.argmax(axis=1), y_score, n_classes=2)
    assert scores["roc_auc"] == pytest.approx(1.0)
    assert scores["pr_auc"] == pytest.approx(1.0)


def test_multiclass_scores_include_ovr_auc(perfect):
    y_true, y_pred = perfect
    # Rows must sum to 1: scikit-learn rejects unnormalised multiclass scores.
    y_score = np.eye(3)[y_pred] * 0.7 + 0.1
    assert np.allclose(y_score.sum(axis=1), 1.0)
    scores = metrics.summary_metrics(y_true, y_pred, y_score, n_classes=3)
    assert scores["roc_auc_ovr_macro"] == pytest.approx(1.0)


def test_evaluate_bundles_everything_the_report_needs(perfect):
    y_true, y_pred = perfect
    result = metrics.evaluate(y_true, y_pred, LABELS, model_name="demo", split="test")
    assert set(result) == {"summary", "per_class", "confusion", "confusion_normalised"}
    assert result["summary"]["model"] == "demo"
    assert result["summary"]["n"] == 6


def test_comparison_table_is_sorted_by_the_primary_metric():
    rows = [
        {"model": "weak", metrics.PRIMARY_METRIC: 0.2},
        {"model": "strong", metrics.PRIMARY_METRIC: 0.9},
        {"model": "middling", metrics.PRIMARY_METRIC: 0.5},
    ]
    table = metrics.comparison_table(rows)
    assert list(table["model"]) == ["strong", "middling", "weak"]


def test_spark_confusion_matrix_matches_the_numpy_one(spark):
    y_true = [0, 0, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 2, 0]
    # Spark ML models emit `prediction` as a double, so the fixture matches that.
    df = spark.createDataFrame(
        [(int(t), float(p)) for t, p in zip(y_true, y_pred)],
        schema="label int, prediction double",
    )
    from_spark = metrics.spark_confusion_matrix(df, LABELS)
    from_numpy = metrics.confusion_matrix(np.array(y_true), np.array(y_pred), LABELS)
    assert from_spark.equals(from_numpy)
