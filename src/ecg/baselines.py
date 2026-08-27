"""Classical Spark ML baselines on the 194-dimensional feature vectors.

These models establish the floor the deep network has to beat, and they do it on
exactly the artefacts produced in ``notebooks/04_preprocessing.ipynb``: the scaled
``features`` vector (187 waveform samples ⊕ 7 descriptors) and the balanced
``class_weight`` column.

Class weighting rather than resampling is used throughout, for the reason recorded
in the EDA: undersampling MIT-BIH to its minority class would discard 96% of the
training data, and every estimator here accepts a ``weightCol`` instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
from pyspark.ml import Model
from pyspark.ml.classification import (
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    OneVsRest,
    RandomForestClassifier,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ecg import metrics

#: Column produced by :func:`ecg.preprocessing.build_dataset`.
FEATURES_COL = "features"
LABEL_COL = "label"
WEIGHT_COL = "class_weight"
SPLIT_COL = "split_final"


@dataclass
class BaselineSpec:
    """A named estimator plus what the evaluation needs to know about it.

    Attributes:
        name: Identifier used in tables, figures and MLflow runs.
        build: Factory taking ``(n_classes, seed)`` and returning an unfitted estimator.
        probability_col: Name of the probability column, or ``None`` when the model
            does not produce calibrated scores (``LinearSVC`` via one-vs-rest).
        description: One line for the report.
    """

    name: str
    build: Callable[[int, int], object]
    probability_col: Optional[str] = "probability"
    description: str = ""
    params: Dict[str, object] = field(default_factory=dict)


def _logistic_regression(n_classes: int, seed: int):
    return LogisticRegression(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        family="multinomial" if n_classes > 2 else "binomial",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
        tol=1e-6,
    )


def _random_forest(n_classes: int, seed: int):
    return RandomForestClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        numTrees=120,
        maxDepth=12,
        minInstancesPerNode=2,
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.8,
        seed=seed,
    )


def _gradient_boosted_trees(n_classes: int, seed: int):
    gbt = GBTClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=25,
        maxDepth=5,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=seed,
    )
    if n_classes == 2:
        return gbt
    # Spark's GBTClassifier is binary-only; one-vs-rest lifts it to 5 classes at
    # the cost of fitting one booster per class.
    return OneVsRest(
        classifier=gbt,
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        parallelism=1,
    )


def _linear_svc(n_classes: int, seed: int):
    svc = LinearSVC(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        maxIter=60,
        regParam=0.01,
    )
    if n_classes == 2:
        return svc
    return OneVsRest(
        classifier=svc,
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol=WEIGHT_COL,
        parallelism=1,
    )


#: The baseline registry, in report order.
BASELINES: Dict[str, BaselineSpec] = {
    "logistic-regression": BaselineSpec(
        name="logistic-regression",
        build=_logistic_regression,
        description="Multinomial logistic regression, L2, class-weighted",
        params={"maxIter": 100, "regParam": 0.01},
    ),
    "random-forest": BaselineSpec(
        name="random-forest",
        build=_random_forest,
        description="120 trees, depth 12, sqrt feature subsampling, class-weighted",
        params={"numTrees": 120, "maxDepth": 12, "subsamplingRate": 0.8},
    ),
    "linear-svc": BaselineSpec(
        name="linear-svc",
        build=_linear_svc,
        probability_col=None,
        description="Linear SVM, one-vs-rest for the 5-class task, class-weighted",
        params={"maxIter": 60, "regParam": 0.01},
    ),
    "gradient-boosted-trees": BaselineSpec(
        name="gradient-boosted-trees",
        build=_gradient_boosted_trees,
        probability_col=None,
        description="Gradient-boosted trees, one-vs-rest for the 5-class task",
        params={"maxIter": 25, "maxDepth": 5, "stepSize": 0.1},
    ),
}

#: Baselines run by default: two linear models, a bagged tree ensemble and a
#: boosted one. Together they bracket the bias/variance range, which is what makes
#: the gap to the deep model interpretable rather than just a number.
DEFAULT_BASELINES: List[str] = [
    "logistic-regression",
    "linear-svc",
    "random-forest",
    "gradient-boosted-trees",
]


def split_frames(dataset: DataFrame, split_col: str = SPLIT_COL) -> Dict[str, DataFrame]:
    """Return ``{split_name: DataFrame}`` for a dataset carrying a split column."""
    return {
        name: dataset.where(F.col(split_col) == name) for name in ("train", "val", "test")
    }


def fit_baseline(spec: BaselineSpec, train: DataFrame, n_classes: int, seed: int = 42):
    """Fit one baseline and report how long it took.

    Args:
        spec: The baseline to fit.
        train: Training split.
        n_classes: Number of classes.
        seed: Random seed.

    Returns:
        ``(fitted_model, seconds)``.
    """
    estimator = spec.build(n_classes, seed)
    started = time.time()
    model = estimator.fit(train)
    return model, time.time() - started


def evaluate_baseline(
    model: Model,
    data: DataFrame,
    label_names: Sequence[str],
    spec: BaselineSpec,
    split: str,
) -> Dict[str, object]:
    """Score a fitted Spark model on one split with :mod:`ecg.metrics`."""
    predictions = model.transform(data)
    y_true, y_pred, y_score = metrics.collect_predictions(
        predictions, label_col=LABEL_COL, probability_col=spec.probability_col
    )
    return metrics.evaluate(
        y_true, y_pred, label_names, y_score=y_score, model_name=spec.name, split=split
    )


def run_baselines(
    dataset: DataFrame,
    label_names: Sequence[str],
    names: Optional[Sequence[str]] = None,
    seed: int = 42,
    splits: Sequence[str] = ("val", "test"),
    verbose: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Fit and score every requested baseline.

    Args:
        dataset: Output of :func:`ecg.preprocessing.build_dataset`.
        label_names: Class names, indexed by numeric label.
        names: Baseline keys; defaults to :data:`DEFAULT_BASELINES`.
        seed: Random seed.
        splits: Splits to score.
        verbose: Print progress as each model finishes.

    Returns:
        ``{baseline_name: {"model": ..., "fit_seconds": float,
        "val": evaluation, "test": evaluation}}``.
    """
    frames = split_frames(dataset)
    train = frames["train"].cache()
    train.count()
    n_classes = len(label_names)

    results: Dict[str, Dict[str, object]] = {}
    for name in list(names or DEFAULT_BASELINES):
        spec = BASELINES[name]
        model, seconds = fit_baseline(spec, train, n_classes, seed)
        entry: Dict[str, object] = {"model": model, "fit_seconds": round(seconds, 1), "spec": spec}
        for split in splits:
            entry[split] = evaluate_baseline(model, frames[split], label_names, spec, split)
        results[name] = entry
        if verbose:
            summary = entry[splits[-1]]["summary"]  # type: ignore[index]
            print(
                f"  {name:<24} fit {seconds:6.1f}s   "
                f"macro-F1 {summary['macro_f1']:.4f}   "
                f"balanced acc {summary['balanced_accuracy']:.4f}"
            )

    train.unpersist()
    return results


def save_model(model: Model, cfg, source: str, name: str) -> str:
    """Persist a fitted Spark ML model next to the data it was fitted on.

    Saving the baselines matters as much as saving the deep model: the notebooks
    reload them to re-run inference live, so the tables they display come from the
    fitted estimator rather than from a cached CSV that could drift from it.
    """
    destination = Path(cfg.models_dir) / f"spark_{source}_{name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(destination))
    return str(destination)


def load_model(cfg, source: str, name: str) -> Model:
    """Reload a model saved by :func:`save_model`.

    The Spark ML class differs per estimator, so the loader is chosen from the
    baseline registry rather than hard-coded.
    """
    from pyspark.ml.classification import (
        GBTClassificationModel,
        LinearSVCModel,
        LogisticRegressionModel,
        OneVsRestModel,
        RandomForestClassificationModel,
    )

    loaders = {
        "logistic-regression": LogisticRegressionModel,
        "random-forest": RandomForestClassificationModel,
        "linear-svc": OneVsRestModel,
        "gradient-boosted-trees": OneVsRestModel,
    }
    if source == "ptbdb":  # binary: no one-vs-rest wrapper
        loaders["linear-svc"] = LinearSVCModel
        loaders["gradient-boosted-trees"] = GBTClassificationModel

    destination = Path(cfg.models_dir) / f"spark_{source}_{name}"
    return loaders[name].load(str(destination))


def results_table(results: Dict[str, Dict[str, object]], split: str = "test") -> pd.DataFrame:
    """Stack the summaries of :func:`run_baselines` into one comparison table."""
    rows = []
    for name, entry in results.items():
        summary = dict(entry[split]["summary"])  # type: ignore[index]
        summary["fit_seconds"] = entry["fit_seconds"]
        summary["description"] = entry["spec"].description  # type: ignore[union-attr]
        rows.append(summary)
    return metrics.comparison_table(rows)


def feature_importances(
    model: Model,
    feature_names: Sequence[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the ranked feature importances of a tree ensemble.

    Args:
        model: A fitted ``RandomForestClassificationModel`` or ``GBTClassificationModel``.
        feature_names: Names of the assembled feature vector, in order.
        top_n: How many rows to return.

    Returns:
        Columns ``feature``, ``importance``, ``rank``.

    Raises:
        AttributeError: If the model does not expose ``featureImportances``.
    """
    importances = model.featureImportances.toArray()
    frame = pd.DataFrame({"feature": list(feature_names), "importance": importances})
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    return frame.head(top_n)


def assembled_feature_names(descriptor_columns: Sequence[str], n_samples: int = 187) -> List[str]:
    """Return the names of the assembled feature vector, in vector order.

    The ``VectorAssembler`` places the 187 waveform samples first, then the
    descriptors, so the names have to follow the same order to be readable.
    """
    return [f"s{index:03d}" for index in range(n_samples)] + list(descriptor_columns)
