"""Persist evaluation results as tables and figures under ``reports/``.

Every model — Spark baseline, CNN or transfer arm — writes the same set of
artefacts through these helpers, so ``reports/`` stays browsable and the README
can point at a predictable file name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from ecg import viz
from ecg.config import Config


def save_table(frame: pd.DataFrame, cfg: Config, name: str, index: bool = False) -> Path:
    """Write a DataFrame to ``reports/tables/<name>.csv``."""
    destination = Path(cfg.tables_dir) / f"{name}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=index)
    return destination


def save_json(payload: Mapping[str, object], cfg: Config, name: str) -> Path:
    """Write a dictionary to ``reports/tables/<name>.json``."""
    destination = Path(cfg.tables_dir) / f"{name}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    return destination


def save_evaluation(
    evaluation: Mapping[str, object],
    cfg: Config,
    slug: str,
    label_order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    make_figures: bool = True,
) -> Dict[str, str]:
    """Write the tables and figures of one model-on-one-split evaluation.

    Args:
        evaluation: Output of :func:`ecg.metrics.evaluate`.
        cfg: Project configuration.
        slug: File-name stem, e.g. ``"mitbih_cnn_test"``.
        label_order: Class order for the per-class figure.
        title: Human-readable model name used in the figure titles.
        make_figures: Render the confusion matrix and per-class panels.

    Returns:
        ``{artefact: path}``.
    """
    cfg.ensure_dirs()
    title = title or slug.replace("_", " ")
    written: Dict[str, str] = {}

    per_class: pd.DataFrame = evaluation["per_class"]  # type: ignore[assignment]
    confusion: pd.DataFrame = evaluation["confusion"]  # type: ignore[assignment]
    normalised: pd.DataFrame = evaluation["confusion_normalised"]  # type: ignore[assignment]

    written["per_class"] = str(save_table(per_class, cfg, f"{slug}_per_class"))
    written["confusion"] = str(save_table(confusion.reset_index(), cfg, f"{slug}_confusion"))
    written["summary"] = str(save_json(evaluation["summary"], cfg, f"{slug}_summary"))  # type: ignore[arg-type]

    if not make_figures:
        return written

    summary = evaluation["summary"]  # type: ignore[index]
    caption = (
        f"macro-F1 {summary['macro_f1']:.4f} · "  # type: ignore[index]
        f"balanced accuracy {summary['balanced_accuracy']:.4f} · "  # type: ignore[index]
        f"{summary['n']:,} beats"  # type: ignore[index]
    )

    written["confusion_figure"] = viz.save_figure(
        viz.plot_confusion_matrix(
            normalised, title=f"Confusion matrix — {title}", subtitle=caption, normalised=True
        ),
        Path(cfg.figures_dir) / f"{slug}_confusion.png",
    )
    written["per_class_figure"] = viz.save_figure(
        viz.plot_per_class_metrics(
            per_class, order=label_order, title=f"Per-class performance — {title}", subtitle=caption
        ),
        Path(cfg.figures_dir) / f"{slug}_per_class.png",
    )
    return written


def save_history(history: pd.DataFrame, cfg: Config, slug: str, best_epoch: Optional[int] = None,
                 title: Optional[str] = None) -> Dict[str, str]:
    """Write a training history as a table and a curve figure."""
    cfg.ensure_dirs()
    written = {"history": str(save_table(history, cfg, f"{slug}_history"))}
    written["history_figure"] = viz.save_figure(
        viz.plot_training_curves(history, title=title or f"Training — {slug}", best_epoch=best_epoch),
        Path(cfg.figures_dir) / f"{slug}_training.png",
    )
    return written
