"""Evaluation shared by the Spark ML baselines and the PyTorch models.

Every model in this project — whether it produced its predictions in Spark or in
PyTorch — is scored through the same functions here, so the comparison table at
the end is apples to apples.

**Accuracy is not reported as a headline.** The EDA established a 113:1 imbalance
in MIT-BIH: a model that answers ``N`` to everything scores 82.8% accuracy while
being clinically worthless. The headline metric is **macro-averaged F1**, which
weights the 803 fusion beats as heavily as the 90,589 normal ones, backed by
per-class recall.

``scikit-learn`` is used for the scores themselves. The inputs are the predictions
of a test split — at most 21,892 rows — so there is nothing to distribute; the
confusion matrix can still be built inside Spark via :func:`spark_confusion_matrix`
when the prediction set is large.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

#: Metric used to select checkpoints, rank models and stop training early.
PRIMARY_METRIC = "macro_f1"


# ------------------------------------------------------------------ collection


def collect_predictions(
    predictions,
    label_col: str = "label",
    prediction_col: str = "prediction",
    probability_col: Optional[str] = "probability",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Bring a Spark prediction DataFrame to the driver as numpy arrays.

    Args:
        predictions: Output of a fitted Spark ML model's ``transform``.
        label_col: Ground-truth column.
        prediction_col: Predicted-class column.
        probability_col: Probability vector column; ``None`` for models such as
            ``LinearSVC`` that do not produce one.

    Returns:
        ``(y_true, y_pred, y_score)``, the last one ``None`` when unavailable.
    """
    columns = [label_col, prediction_col]
    if probability_col is not None and probability_col in predictions.columns:
        columns.append(probability_col)

    pdf = predictions.select(*columns).toPandas()
    y_true = pdf[label_col].to_numpy(dtype=int)
    y_pred = pdf[prediction_col].to_numpy(dtype=int)

    y_score = None
    if probability_col in pdf.columns:
        y_score = np.vstack([row.toArray() for row in pdf[probability_col]])
    return y_true, y_pred, y_score


def spark_confusion_matrix(
    predictions,
    label_names: Sequence[str],
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Build the confusion matrix with a Spark aggregation instead of collecting rows.

    Useful when the prediction set is too large to bring to the driver; the result
    is identical to :func:`confusion_matrix`.

    Args:
        predictions: Spark DataFrame with true and predicted labels.
        label_names: Class names, indexed by numeric label.
        label_col: Ground-truth column.
        prediction_col: Predicted-class column.

    Returns:
        A DataFrame indexed by true class, columned by predicted class.
    """
    from pyspark.sql import functions as F

    counts = (
        predictions.groupBy(label_col, prediction_col)
        .agg(F.count(F.lit(1)).alias("n"))
        .toPandas()
    )
    size = len(label_names)
    matrix = np.zeros((size, size), dtype=int)
    for _, row in counts.iterrows():
        matrix[int(row[label_col]), int(row[prediction_col])] = int(row["n"])
    return _as_frame(matrix, label_names)


# --------------------------------------------------------------------- scoring


def _as_frame(matrix: np.ndarray, label_names: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=list(label_names), columns=list(label_names))
    frame.index.name = "true"
    frame.columns.name = "predicted"
    return frame


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        label_names: Class names, indexed by numeric label.
        normalize: ``None`` for counts, ``"true"`` for row-wise recall shares,
            ``"pred"`` for column-wise precision shares.
    """
    labels = list(range(len(label_names)))
    matrix = sk_confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return _as_frame(matrix, label_names)


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> pd.DataFrame:
    """Return per-class precision, recall, F1 and support, plus macro/weighted rows.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        label_names: Class names, indexed by numeric label.

    Returns:
        Columns ``class``, ``precision``, ``recall``, ``f1``, ``support``.
    """
    labels = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rows = [
        {
            "class": name,
            "precision": precision[index],
            "recall": recall[index],
            "f1": f1[index],
            "support": int(support[index]),
        }
        for index, name in enumerate(label_names)
    ]

    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        rows.append(
            {
                "class": f"{average} avg",
                "precision": p,
                "recall": r,
                "f1": f,
                "support": int(support.sum()),
            }
        )

    frame = pd.DataFrame(rows)
    frame[["precision", "recall", "f1"]] = frame[["precision", "recall", "f1"]].round(4)
    return frame


def summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Return the headline scores for one model on one split.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_score: Predicted probabilities, shape ``(n, n_classes)``; enables the
            ranking metrics.
        n_classes: Number of classes; inferred from ``y_score`` or the labels.

    Returns:
        A flat dictionary — ``macro_f1`` first, because it is the metric that
        decides everything in this project.
    """
    n_classes = n_classes or (
        y_score.shape[1] if y_score is not None else int(max(y_true.max(), y_pred.max())) + 1
    )

    metrics: Dict[str, float] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    if y_score is not None and y_score.shape[1] == n_classes:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_score[:, 1]))
            else:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
        except ValueError:
            # A split missing one class makes the ranking metrics undefined.
            pass

    return {key: round(value, 4) for key, value in metrics.items()}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    y_score: Optional[np.ndarray] = None,
    model_name: str = "model",
    split: str = "test",
) -> Dict[str, object]:
    """Score one model on one split and return everything needed to report it.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        label_names: Class names, indexed by numeric label.
        y_score: Predicted probabilities, when available.
        model_name: Identifier carried into the comparison table.
        split: Split name.

    Returns:
        ``{"summary": dict, "per_class": DataFrame, "confusion": DataFrame,
        "confusion_normalised": DataFrame}``.
    """
    summary = summary_metrics(y_true, y_pred, y_score, n_classes=len(label_names))
    summary = {"model": model_name, "split": split, "n": int(len(y_true)), **summary}
    return {
        "summary": summary,
        "per_class": per_class_report(y_true, y_pred, label_names),
        "confusion": confusion_matrix(y_true, y_pred, label_names),
        "confusion_normalised": confusion_matrix(y_true, y_pred, label_names, normalize="true"),
    }


def comparison_table(results: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Stack the ``summary`` dictionaries of several evaluations, best first.

    Args:
        results: Outputs of :func:`evaluate`, or bare summary dictionaries.

    Returns:
        A DataFrame sorted by :data:`PRIMARY_METRIC`, descending.
    """
    rows: List[Dict[str, object]] = []
    for result in results:
        rows.append(result["summary"] if "summary" in result else result)  # type: ignore[index]
    frame = pd.DataFrame(rows)
    if PRIMARY_METRIC in frame.columns:
        frame = frame.sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)
    return frame


def majority_class_baseline(y_true: np.ndarray, label_names: Sequence[str]) -> Dict[str, object]:
    """Score the "always answer the majority class" strategy.

    Included in every comparison on purpose: it is the number that makes the
    difference between accuracy and macro-F1 impossible to miss.
    """
    majority = int(np.bincount(y_true).argmax())
    y_pred = np.full_like(y_true, majority)
    return evaluate(y_true, y_pred, label_names, model_name="majority-class", split="test")
