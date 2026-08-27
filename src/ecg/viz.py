"""Matplotlib rendering of the summaries produced by :mod:`ecg.eda`.

Design rules applied throughout, so every figure in ``reports/figures`` reads as
one system:

* **Categorical colour is assigned by identity, in fixed slot order** — a class
  keeps its hue no matter which figure it appears in, and hues are never cycled.
  The five-slot palette below was validated for colour-vision deficiency
  (worst adjacent CVD ΔE 9.1, normal-vision ΔE 19.6 on the light surface).
* **Three of those hues fall below 3:1 contrast against the surface**, so every
  categorical mark also carries a visible direct label — identity is never
  colour-alone.
* **One axis per chart**, recessive grid and spines, thin marks, and small
  multiples instead of five overlapping lines whenever morphology is compared.

Every function takes a pandas DataFrame (already aggregated in Spark), returns a
:class:`matplotlib.figure.Figure`, and never touches Spark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# ------------------------------------------------------------------ palette

#: Categorical hues in fixed slot order (validated for CVD on the light surface).
CATEGORICAL: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

#: Chart surface and ink.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#78766f"
GRID = "#e6e5e1"
REFERENCE = "#b9b7b0"

#: Diverging poles used by the correlation heatmap (blue ↔ red, neutral middle).
DIVERGING = LinearSegmentedColormap.from_list(
    "ecg_diverging", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#a32220"]
)

#: Stable class → hue assignment, per collection.
CLASS_COLORS: Dict[str, Dict[str, str]] = {
    "mitbih": {
        "N": CATEGORICAL[0],
        "S": CATEGORICAL[1],
        "V": CATEGORICAL[2],
        "F": CATEGORICAL[3],
        "Q": CATEGORICAL[4],
    },
    "ptbdb": {
        "normal": CATEGORICAL[0],
        "abnormal": CATEGORICAL[1],
    },
}


def class_color(source: str, label_name: str) -> str:
    """Return the fixed hue assigned to a class."""
    return CLASS_COLORS.get(source, {}).get(label_name, CATEGORICAL[0])


def _label_bottom_panels(axes, n_used: int, n_rows: int, n_cols: int, xlabel: str) -> None:
    """Give every column's lowest *used* panel its x tick labels back.

    With ``sharex=True`` matplotlib hides tick labels on every row but the last.
    When the grid is ragged — five classes in a 3×2 grid — the panels in the last
    populated row of a short column end up with no labelled axis anywhere below
    them, so the labels are restored explicitly here.
    """
    for col in range(n_cols):
        rows = [row for row in range(n_rows) if row * n_cols + col < n_used]
        if not rows:
            continue
        ax = axes[max(rows)][col]
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(xlabel)


def apply_style() -> None:
    """Install the project's matplotlib defaults.

    Call once per notebook or script before plotting.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "text.color": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
        }
    )


def _finish(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
    """Apply shared axis furniture."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _figure_header(
    fig: Figure,
    title: str,
    subtitle: str = "",
    header_inches: float = 0.78,
) -> None:
    """Reserve a fixed-height header above a grid of panels and fill it.

    ``suptitle`` positions itself in figure fractions, which collides with a
    subtitle whenever the figure height changes with the number of panels.
    Reserving the header in *inches* keeps the spacing identical across every
    figure in the report.
    """
    height = fig.get_figheight()
    fig.subplots_adjust(top=1.0 - header_inches / height)
    fig.text(
        0.012,
        1.0 - 0.16 / height,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="600",
        color=TEXT_PRIMARY,
    )
    if subtitle:
        fig.text(
            0.012,
            1.0 - 0.42 / height,
            subtitle,
            ha="left",
            va="top",
            fontsize=9,
            color=TEXT_MUTED,
        )


def save_figure(fig: Figure, path: Path | str, close: bool = True) -> str:
    """Save a figure, creating parent directories as needed.

    Args:
        fig: Figure to save.
        path: Destination file (``.png`` recommended for GitHub rendering).
        close: Close the figure afterwards to free memory.

    Returns:
        The destination path as a string.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    if close:
        plt.close(fig)
    return str(destination)


# ------------------------------------------------------------- distributions


def plot_class_distribution(
    distribution: pd.DataFrame,
    source: str,
    order: Optional[Sequence[str]] = None,
    split: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> Figure:
    """Horizontal bars of beats per class, with counts and shares labelled.

    Args:
        distribution: Output of :func:`ecg.eda.class_distribution`.
        source: Collection to plot (``mitbih`` / ``ptbdb``).
        order: Class order; defaults to the order found in the data.
        split: Optional split filter.
        title: Chart title.
        figsize: Figure size in inches; scaled to the class count when omitted.

    Returns:
        The figure.
    """
    data = distribution[distribution["source"] == source]
    if split is not None:
        data = data[data["split"] == split]
    data = data.groupby("label_name", as_index=False)["n_beats"].sum()
    total = data["n_beats"].sum()
    data["pct"] = data["n_beats"] / total * 100

    order = list(order) if order is not None else list(data["label_name"])
    data = data.set_index("label_name").reindex(order).reset_index()

    figsize = figsize or (7.6, 1.5 + 0.46 * len(data))
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(data))[::-1]
    colors = [class_color(source, name) for name in data["label_name"]]
    ax.barh(positions, data["n_beats"], height=0.62, color=colors)

    span = float(data["n_beats"].max())
    for y, (count, pct) in zip(positions, zip(data["n_beats"], data["pct"])):
        ax.text(
            count + span * 0.012,
            y,
            f"{count:,}  ·  {pct:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    ax.set_yticks(positions, list(data["label_name"]))
    ax.set_xlim(0, span * 1.22)
    ax.grid(axis="y", visible=False)
    _finish(ax, xlabel="Heartbeats")
    fig.tight_layout()
    _figure_header(
        fig,
        title or f"Class distribution — {source}",
        f"{total:,} beats" + (f" · split: {split}" if split else ""),
    )
    return fig


def plot_split_comparison(
    distribution: pd.DataFrame,
    source: str = "mitbih",
    order: Optional[Sequence[str]] = None,
    figsize: tuple = (8.0, 4.2),
) -> Figure:
    """Grouped bars comparing the class shares of the train and test splits.

    Two series only, so the two leading hues are used and both are direct-labelled.
    """
    data = distribution[distribution["source"] == source]
    splits = [s for s in ("train", "test") if s in set(data["split"])]
    order = list(order) if order is not None else sorted(set(data["label_name"]))

    fig, ax = plt.subplots(figsize=figsize)
    width = 0.38
    positions = np.arange(len(order))

    for index, split in enumerate(splits):
        chunk = (
            data[data["split"] == split]
            .set_index("label_name")
            .reindex(order)
            .reset_index()
        )
        offset = (index - (len(splits) - 1) / 2) * (width + 0.02)
        bars = ax.bar(
            positions + offset,
            chunk["pct"],
            width=width,
            color=CATEGORICAL[index],
            label=f"{split} ({int(chunk['n_beats'].sum()):,} beats)",
        )
        for bar, pct in zip(bars, chunk["pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{pct:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=TEXT_SECONDARY,
            )

    ax.set_xticks(positions, order)
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    _finish(ax, xlabel="Class", ylabel="Share of split (%)")
    fig.tight_layout()
    _figure_header(
        fig,
        f"Train vs test class balance — {source}",
        "Bar labels give the share of each split, so a matching pair means the split is stratified",
    )
    return fig


# ----------------------------------------------------------------- waveforms


def plot_sample_beats(
    samples: pd.DataFrame,
    source: str,
    order: Optional[Sequence[str]] = None,
    n_per_class: int = 4,
    figsize_per_panel: tuple = (2.5, 1.6),
) -> Figure:
    """Small multiples of individual beats: one row per class.

    Args:
        samples: Output of :func:`ecg.eda.sample_beats`.
        source: Collection, used for the colour assignment.
        order: Class order.
        n_per_class: Columns in the grid.
        figsize_per_panel: Size of each panel, in inches.

    Returns:
        The figure.
    """
    order = list(order) if order is not None else sorted(set(samples["label_name"]))
    n_rows, n_cols = len(order), n_per_class
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, label_name in enumerate(order):
        chunk = samples[samples["label_name"] == label_name].head(n_cols)
        color = class_color(source, label_name)
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(chunk):
                beat = chunk.iloc[col]
                length = int(beat["signal_length"])
                signal = np.asarray(beat["signal"], dtype=float)[:length]
                ax.plot(np.arange(length) / 125.0, signal, color=color, linewidth=1.4)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(visible=False)
            ax.tick_params(labelsize=8)
            if col == 0:
                # The class name lives outside the panel so it can never collide
                # with the waveform, which reaches the top of the axis by design.
                ax.set_ylabel(
                    label_name,
                    color=color,
                    fontsize=13,
                    fontweight="600",
                    rotation=0,
                    labelpad=18,
                    va="center",
                    ha="right",
                )
            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8.5)

    fig.tight_layout()
    _figure_header(
        fig,
        f"Individual heartbeats by class — {source}",
        "Random beats, padding removed · amplitude min-max normalised to [0, 1]",
    )
    return fig


def plot_waveform_profiles(
    profile: pd.DataFrame,
    source: str,
    order: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
    n_cols: int = 3,
    figsize_per_panel: tuple = (3.3, 2.4),
) -> Figure:
    """Small multiples of the average beat morphology, one panel per class.

    Each panel shows the per-sample median with its interquartile band, plus the
    reference class median in grey so shapes can be compared without putting five
    coloured lines on one axis.

    Args:
        profile: Output of :func:`ecg.eda.waveform_profile`.
        source: Collection, used for the colour assignment.
        order: Class order.
        reference: Class drawn as the grey reference in every panel; defaults to
            the first class in ``order``.
        n_cols: Panels per row.
        figsize_per_panel: Size of each panel, in inches.

    Returns:
        The figure.
    """
    order = list(order) if order is not None else sorted(set(profile["label_name"]))
    reference = reference or order[0]
    ref = profile[profile["label_name"] == reference].sort_values("t")

    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for index, label_name in enumerate(order):
        ax = axes[index // n_cols][index % n_cols]
        chunk = profile[profile["label_name"] == label_name].sort_values("t")
        color = class_color(source, label_name)

        if label_name != reference and not ref.empty:
            ax.plot(ref["time_s"], ref["q50"], color=REFERENCE, linewidth=1.2, zorder=1)
        ax.fill_between(
            chunk["time_s"], chunk["q25"], chunk["q75"], color=color, alpha=0.22, linewidth=0, zorder=2
        )
        ax.plot(chunk["time_s"], chunk["q50"], color=color, linewidth=1.8, zorder=3)

        n_beats = int(chunk["n"].max()) if not chunk.empty else 0
        ax.text(
            0.97,
            0.92,
            label_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            fontweight="600",
            color=color,
        )
        ax.text(
            0.97,
            0.78,
            f"{n_beats:,} beats",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=TEXT_MUTED,
        )
        ax.set_ylim(-0.05, 1.05)

    for index in range(len(order), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    _label_bottom_panels(axes, len(order), n_rows, n_cols, "Time (s)")
    for row in range(n_rows):
        axes[row][0].set_ylabel("Normalised amplitude")

    fig.tight_layout()
    _figure_header(
        fig,
        f"Average beat morphology by class — {source}",
        f"Line = per-sample median · band = interquartile range · grey = class “{reference}” median",
    )
    return fig


def plot_length_distribution(
    histogram: pd.DataFrame,
    source: str,
    order: Optional[Sequence[str]] = None,
    n_cols: int = 3,
    figsize_per_panel: tuple = (3.3, 2.2),
) -> Figure:
    """Small multiples of the effective-length histogram, one panel per class."""
    order = list(order) if order is not None else sorted(set(histogram["label_name"]))
    n_rows = int(np.ceil(len(order) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    bin_width = int(histogram["bin_end"].iloc[0] - histogram["bin_start"].iloc[0])

    for index, label_name in enumerate(order):
        ax = axes[index // n_cols][index % n_cols]
        chunk = histogram[histogram["label_name"] == label_name].sort_values("bin_start")
        color = class_color(source, label_name)
        ax.bar(
            chunk["bin_start"],
            chunk["pct"],
            width=bin_width * 0.86,
            align="edge",
            color=color,
        )
        ax.text(
            0.97,
            0.92,
            label_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            fontweight="600",
            color=color,
        )
        ax.grid(axis="x", visible=False)

    for index in range(len(order), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    _label_bottom_panels(axes, len(order), n_rows, n_cols, "Effective length (samples)")
    for row in range(n_rows):
        axes[row][0].set_ylabel("Share of class (%)")

    fig.tight_layout()
    _figure_header(
        fig,
        f"Beat length before zero padding — {source}",
        f"Bin width {bin_width} samples ({bin_width / 125:.2f} s at 125 Hz)",
    )
    return fig


# ---------------------------------------------------------------- descriptors


def plot_feature_boxes(
    summary: pd.DataFrame,
    source: str,
    features: Iterable[str],
    order: Optional[Sequence[str]] = None,
    n_cols: int = 3,
    figsize_per_panel: tuple = (3.4, 2.6),
) -> Figure:
    """Percentile boxes per class, built from the Spark-computed quantiles.

    The boxes are drawn with ``Axes.bxp`` from statistics that were already
    aggregated in Spark, so no raw beat is ever collected to the driver. Box =
    interquartile range, whiskers = 5th–95th percentile, notch line = median.

    Args:
        summary: Output of :func:`ecg.eda.feature_summary`.
        source: Collection, used for the colour assignment.
        features: Descriptors to plot, one panel each.
        order: Class order.
        n_cols: Panels per row.
        figsize_per_panel: Size of each panel, in inches.

    Returns:
        The figure.
    """
    features = list(features)
    data = summary[summary["source"] == source] if "source" in summary.columns else summary
    order = list(order) if order is not None else sorted(set(data["label_name"]))

    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for index, feature in enumerate(features):
        ax = axes[index // n_cols][index % n_cols]
        chunk = data[data["feature"] == feature].set_index("label_name").reindex(order)
        stats = [
            {
                "label": name,
                "med": float(row["q50"]),
                "q1": float(row["q25"]),
                "q3": float(row["q75"]),
                "whislo": float(row["q05"]),
                "whishi": float(row["q95"]),
                "fliers": [],
            }
            for name, row in chunk.iterrows()
        ]
        artists = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.55)
        for patch, name in zip(artists["boxes"], order):
            color = class_color(source, name)
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1.4)
        for element in ("whiskers", "caps"):
            for artist in artists[element]:
                artist.set_color(TEXT_SECONDARY)
                artist.set_linewidth(1.0)
        for artist in artists["medians"]:
            artist.set_color(SURFACE)
            artist.set_linewidth(1.6)

        ax.set_title(feature, loc="left", fontsize=10.5)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=9)

    for index in range(len(features), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    fig.tight_layout()
    _figure_header(
        fig,
        f"Per-beat descriptors by class — {source}",
        "Box = interquartile range · whiskers = 5th–95th percentile · line = median",
        header_inches=1.05,  # extra room: these panels carry their own titles
    )
    return fig


def plot_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str = "Descriptor correlation",
    figsize: tuple = (7.6, 6.4),
    annotate: bool = True,
) -> Figure:
    """Diverging heatmap of a correlation matrix.

    Diverging encoding with a neutral grey midpoint, symmetric limits at ±1, and
    every cell labelled so the reading never depends on colour alone.
    """
    labels = list(correlation.columns)
    values = correlation.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(values, cmap=DIVERGING, vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8.5)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}".replace("0.", ".").replace("-.", "−."),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=SURFACE if abs(value) > 0.55 else TEXT_SECONDARY,
                )

    bar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.5, color=TEXT_SECONDARY)
    bar.set_label("Pearson r", fontsize=9, color=TEXT_SECONDARY)

    fig.tight_layout()
    _figure_header(
        fig,
        title,
        "Diverging scale · every cell labelled, so the reading never depends on colour alone",
    )
    return fig


# ------------------------------------------------------------------ modelling

#: Sequential blue ramp (100 → 700) used for magnitude encodings.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "ecg_sequential",
    ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

#: Fixed hues for the three per-class metrics; the first three slots are the ones
#: that clear the all-pairs colour-vision gates.
METRIC_COLORS: Dict[str, str] = {
    "precision": CATEGORICAL[0],
    "recall": CATEGORICAL[1],
    "f1": CATEGORICAL[2],
}


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    title: str = "Confusion matrix",
    subtitle: str = "",
    normalised: bool = False,
    figsize: Optional[tuple] = None,
) -> Figure:
    """Heatmap of a confusion matrix, every cell labelled.

    Magnitude is a sequential encoding, so a single hue runs light to dark — never
    a rainbow. Cell values are always printed, so the reading never depends on
    colour, and the diagonal is what the eye should follow.

    Args:
        matrix: Output of :func:`ecg.metrics.confusion_matrix`, true classes on the
            index and predicted classes on the columns.
        title: Chart title.
        subtitle: Line under the title.
        normalised: ``True`` when the matrix holds row-wise shares rather than
            counts, which switches the cell format to percentages.
        figsize: Figure size in inches; scaled to the class count when omitted.

    Returns:
        The figure.
    """
    labels = list(matrix.columns)
    values = matrix.to_numpy(dtype=float)
    size = len(labels)
    figsize = figsize or (1.9 + 0.85 * size, 1.7 + 0.85 * size)

    fig, ax = plt.subplots(figsize=figsize)
    vmax = 1.0 if normalised else values.max()
    image = ax.imshow(values, cmap=SEQUENTIAL_BLUE, vmin=0.0, vmax=vmax)

    for i in range(size):
        for j in range(size):
            value = values[i, j]
            text = f"{value * 100:.1f}%" if normalised else f"{int(value):,}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="600" if i == j else "normal",
                color=SURFACE if value > vmax * 0.55 else TEXT_SECONDARY,
            )

    ax.set_xticks(np.arange(size), labels)
    ax.set_yticks(np.arange(size), labels)
    ax.set_xticks(np.arange(size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(size + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    bar = fig.colorbar(image, ax=ax, shrink=0.7, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.5, color=TEXT_SECONDARY)
    bar.set_label("Share of true class" if normalised else "Beats", fontsize=9, color=TEXT_SECONDARY)

    fig.tight_layout()
    _figure_header(
        fig,
        title,
        subtitle or ("Row-normalised: each row is one true class, so the diagonal is recall"
                     if normalised else "Counts · rows are true classes, columns predictions"),
    )
    return fig


def plot_training_curves(
    history: pd.DataFrame,
    title: str = "Training",
    subtitle: str = "",
    best_epoch: Optional[int] = None,
    figsize: tuple = (9.2, 3.4),
) -> Figure:
    """Two panels: losses on the left, validation macro-F1 on the right.

    Two panels rather than one chart with two y-axes — a dual-axis plot invites the
    reader to compare quantities that share no scale.

    Args:
        history: Per-epoch history from :func:`ecg.training.train`.
        title: Chart title.
        subtitle: Line under the title.
        best_epoch: Epoch to mark as the selected checkpoint.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(history["epoch"], history["train_loss"], color=CATEGORICAL[0], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], color=CATEGORICAL[1], label="validation")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(loc="upper right")

    axes[1].plot(history["epoch"], history["val_macro_f1"], color=CATEGORICAL[2])
    axes[1].set_ylabel("Validation macro-F1")

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, color=REFERENCE, linewidth=1.0, zorder=0)
        row = history[history["epoch"] == best_epoch]
        if not row.empty:
            value = float(row["val_macro_f1"].iloc[0])
            axes[1].scatter([best_epoch], [value], s=42, color=CATEGORICAL[2], zorder=3)
            axes[1].annotate(
                f"best {value:.4f}\nepoch {best_epoch}",
                (best_epoch, value),
                textcoords="offset points",
                xytext=(-8, -28),
                ha="right",
                fontsize=8.5,
                color=TEXT_SECONDARY,
            )

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(axis="x", visible=False)

    fig.tight_layout()
    _figure_header(fig, title, subtitle or "Checkpoint selected on validation macro-F1")
    return fig


def plot_per_class_metrics(
    per_class: pd.DataFrame,
    order: Optional[Sequence[str]] = None,
    title: str = "Per-class performance",
    subtitle: str = "",
    figsize: Optional[tuple] = None,
) -> Figure:
    """Grouped bars of precision, recall and F1 for each class.

    Args:
        per_class: Output of :func:`ecg.metrics.per_class_report`; the average rows
            are dropped.
        order: Class order.
        title: Chart title.
        subtitle: Line under the title.
        figsize: Figure size in inches; scaled to the class count when omitted.

    Returns:
        The figure.
    """
    data = per_class[~per_class["class"].str.contains("avg")].copy()
    order = list(order) if order is not None else list(data["class"])
    data = data.set_index("class").reindex(order).reset_index()

    figsize = figsize or (2.4 + 1.35 * len(order), 4.0)
    fig, ax = plt.subplots(figsize=figsize)

    metric_names = ["precision", "recall", "f1"]
    width = 0.26
    positions = np.arange(len(order))

    for index, metric in enumerate(metric_names):
        offset = (index - 1) * (width + 0.015)
        bars = ax.bar(
            positions + offset,
            data[metric],
            width=width,
            color=METRIC_COLORS[metric],
            label=metric,
        )
        for bar, value in zip(bars, data[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}".lstrip("0"),
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_SECONDARY,
            )

    ax.set_xticks(
        positions,
        [f"{name}\nn={int(support):,}" for name, support in zip(order, data["support"])],
    )
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", ncol=3)
    _finish(ax, ylabel="Score")

    fig.tight_layout()
    _figure_header(fig, title, subtitle or "Support shown under each class")
    return fig


def plot_model_comparison(
    table: pd.DataFrame,
    metric: str = "macro_f1",
    label_col: str = "model",
    title: str = "Model comparison",
    subtitle: str = "",
    reference: Optional[float] = None,
    reference_label: str = "majority class",
    figsize: Optional[tuple] = None,
) -> Figure:
    """Horizontal bars ranking models on one metric, values labelled.

    Args:
        table: Output of :func:`ecg.metrics.comparison_table`.
        metric: Column to rank on.
        label_col: Column holding the model names.
        title: Chart title.
        subtitle: Line under the title.
        reference: Optional vertical reference line, e.g. a trivial baseline.
        reference_label: Label for that line.
        figsize: Figure size in inches; scaled to the row count when omitted.

    Returns:
        The figure.
    """
    data = table.sort_values(metric, ascending=True).reset_index(drop=True)
    figsize = figsize or (8.0, 1.6 + 0.46 * len(data))
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(len(data))
    best = data[metric].max()
    colors = [
        CATEGORICAL[0] if value >= best - 1e-12 else REFERENCE for value in data[metric]
    ]
    ax.barh(positions, data[metric], height=0.6, color=colors)

    for y, value in zip(positions, data[metric]):
        ax.text(
            value + 0.012,
            y,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    if reference is not None:
        ax.axvline(reference, color=CATEGORICAL[1], linewidth=1.4, linestyle=(0, (4, 3)))
        ax.text(
            reference,
            len(data) - 0.35,
            f"  {reference_label} ({reference:.3f})",
            color=CATEGORICAL[1],
            fontsize=8.5,
            va="center",
        )

    ax.set_yticks(positions, list(data[label_col]))
    ax.set_xlim(0, min(1.0, max(best, reference or 0) * 1.28))
    ax.grid(axis="y", visible=False)
    _finish(ax, xlabel=metric.replace("_", " "))

    fig.tight_layout()
    _figure_header(fig, title, subtitle or "Higher is better · best model highlighted")
    return fig


def plot_error_examples(
    signals: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    source: str,
    confusions: Optional[Sequence[tuple]] = None,
    n_examples: int = 3,
    seed: int = 42,
    figsize_per_panel: tuple = (2.6, 1.7),
) -> Figure:
    """Grid of misclassified beats, one row per confusion pair.

    Reading the actual waveforms a model got wrong is usually more informative
    than another aggregate: it shows whether the errors are borderline morphology
    or something the model should obviously have caught.

    Args:
        signals: Waveforms, shape ``(n, 187)``.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        label_names: Class names, indexed by numeric label.
        source: Collection, for the colour assignment.
        confusions: ``(true, predicted)`` index pairs to show; defaults to the
            most frequent off-diagonal cells.
        n_examples: Beats per row.
        seed: Sampling seed.
        figsize_per_panel: Size of each panel, in inches.

    Returns:
        The figure.
    """
    rng = np.random.default_rng(seed)
    errors = y_true != y_pred

    if confusions is None:
        pairs, counts = np.unique(
            np.stack([y_true[errors], y_pred[errors]], axis=1), axis=0, return_counts=True
        )
        confusions = [tuple(pair) for pair in pairs[np.argsort(-counts)][:4]]

    n_rows = len(confusions)
    fig, axes = plt.subplots(
        n_rows,
        n_examples,
        figsize=(figsize_per_panel[0] * n_examples, figsize_per_panel[1] * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, (true_label, pred_label) in enumerate(confusions):
        mask = (y_true == true_label) & (y_pred == pred_label)
        indices = np.flatnonzero(mask)
        if len(indices) > n_examples:
            indices = rng.choice(indices, n_examples, replace=False)

        true_name, pred_name = label_names[int(true_label)], label_names[int(pred_label)]
        for col in range(n_examples):
            ax = axes[row][col]
            if col < len(indices):
                signal = signals[indices[col]]
                length = int(np.max(np.nonzero(signal)[0]) + 1) if signal.any() else len(signal)
                ax.plot(
                    np.arange(length) / 125.0,
                    signal[:length],
                    color=class_color(source, true_name),
                    linewidth=1.3,
                )
            ax.set_ylim(-0.05, 1.05)
            ax.grid(visible=False)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(
                    f"{true_name} → {pred_name}\n({int(mask.sum())} beats)",
                    fontsize=9,
                    color=TEXT_SECONDARY,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=42,
                )
            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8.5)

    fig.tight_layout()
    _figure_header(
        fig,
        f"Misclassified beats — {source}",
        "Most frequent confusions · label reads true class → predicted class",
    )
    return fig


def plot_low_data_curve(
    curve: pd.DataFrame,
    arms: Optional[Sequence[str]] = None,
    metric: str = "macro_f1",
    title: str = "Transfer under a shrinking target dataset",
    subtitle: str = "",
    figsize: tuple = (8.4, 4.4),
) -> Figure:
    """One line per transfer arm as the target training set shrinks.

    Three series, so the three leading hues are used — the only slots that clear
    the all-pairs colour-vision gates — and each line is direct-labelled at its
    right end so identity never rests on colour alone.

    Args:
        curve: Output of :func:`ecg.transfer.low_data_curve`.
        arms: Arm order; defaults to the order found in the data.
        metric: Column to plot.
        title: Chart title.
        subtitle: Line under the title.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    arms = list(arms) if arms is not None else list(dict.fromkeys(curve["arm"]))
    fig, ax = plt.subplots(figsize=figsize)

    for index, arm in enumerate(arms):
        chunk = curve[curve["arm"] == arm].sort_values("n_train")
        color = CATEGORICAL[index % 3]
        ax.plot(chunk["n_train"], chunk[metric], color=color, marker="o", markersize=5, label=arm)
        last = chunk.iloc[-1]
        ax.annotate(
            arm,
            (last["n_train"], last[metric]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=9.5,
            fontweight="600",
            color=color,
        )

    ax.set_xscale("log")
    ax.set_xticks(sorted(curve["n_train"].unique()))
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=8.5)
    ax.set_xlim(curve["n_train"].min() * 0.85, curve["n_train"].max() * 1.9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    _finish(ax, xlabel="PTB training beats (log scale)", ylabel=metric.replace("_", " "))

    fig.tight_layout()
    _figure_header(
        fig,
        title,
        subtitle or "Validation and test splits held at full size; only the training set shrinks",
    )
    return fig
