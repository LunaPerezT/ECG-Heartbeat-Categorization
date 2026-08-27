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
