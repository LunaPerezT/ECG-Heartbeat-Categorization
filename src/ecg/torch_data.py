"""Bridge from the Spark Parquet datasets to PyTorch tensors.

The deep model reads the **raw 187-sample waveform**, not the scaled 194-dimensional
vector the Spark baselines use: convolutions are what extract shape, and the
signals are already min-max normalised to ``[0, 1]`` by the dataset authors, so no
further scaling is required or wanted.

What *is* reused from the Spark stage is the split assignment. Both model families
therefore train, validate and test on exactly the same beats, which is what makes
the final comparison table meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg.config import Config
from ecg.schema import N_SAMPLES

#: Splits materialised by :func:`load_arrays`.
SPLITS: Tuple[str, ...] = ("train", "val", "test")


def _cache_path(cfg: Config, source: str) -> Path:
    return Path(cfg.processed_dir) / f"arrays_{source}.npz"


def load_arrays(
    spark,
    cfg: Config,
    source: str = "mitbih",
    splits: Sequence[str] = SPLITS,
    use_cache: bool = True,
    dataset_name: str = "features",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Materialise the waveforms and labels of each split as numpy arrays.

    The whole collection is 109,446 × 187 float32 — about 82 MB — so it is loaded
    into driver memory once and cached to a ``.npz`` beside the Parquet. Reading
    the cache takes well under a second, which keeps notebook re-runs cheap.

    Args:
        spark: Active Spark session (only used on a cache miss).
        cfg: Project configuration.
        source: ``"mitbih"`` or ``"ptbdb"``.
        splits: Splits to return.
        use_cache: Read and write the ``.npz`` cache.
        dataset_name: Parquet dataset prefix written by
            :func:`ecg.preprocessing.build_dataset`.

    Returns:
        ``{split: (X, y)}`` with ``X`` of shape ``(n, 187)`` float32 and ``y`` of
        shape ``(n,)`` int64.
    """
    cache = _cache_path(cfg, source)
    if use_cache and cache.exists():
        stored = np.load(cache)
        return {split: (stored[f"X_{split}"], stored[f"y_{split}"]) for split in splits}

    path = Path(cfg.processed_dir) / f"{dataset_name}_{source}"
    pdf = (
        spark.read.parquet(str(path))
        .select("split_final", "label", "signal")
        .toPandas()
    )

    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        chunk = pdf[pdf["split_final"] == split]
        X = np.vstack(chunk["signal"].to_numpy()).astype(np.float32, copy=False)
        y = chunk["label"].to_numpy().astype(np.int64, copy=False)
        if X.shape[1] != N_SAMPLES:  # pragma: no cover - defensive
            raise ValueError(f"expected {N_SAMPLES} samples per beat, got {X.shape[1]}")
        arrays[split] = (X, y)

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            **{f"X_{split}": arrays[split][0] for split in SPLITS},
            **{f"y_{split}": arrays[split][1] for split in SPLITS},
        )

    return {split: arrays[split] for split in splits}


def make_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    """Wrap ``(X, y)`` as a ``TensorDataset`` shaped ``(n, 1, 187)``."""
    tensors = torch.from_numpy(np.ascontiguousarray(X)).unsqueeze(1)
    labels = torch.from_numpy(np.ascontiguousarray(y))
    return TensorDataset(tensors, labels)


def make_loaders(
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Build one ``DataLoader`` per split.

    Args:
        arrays: Output of :func:`load_arrays`.
        batch_size: Training batch size.
        eval_batch_size: Batch size for validation and test; defaults to ``4 ×
            batch_size`` since no gradients are held.
        shuffle_train: Shuffle the training split.
        num_workers: Worker processes. Zero is the right answer on a small CPU
            box: the tensors are already in memory, so workers would only add
            process overhead.
        seed: Seed for the shuffling generator, so epochs are reproducible.

    Returns:
        ``{split: DataLoader}``.
    """
    eval_batch_size = eval_batch_size or batch_size * 4
    generator = torch.Generator().manual_seed(seed)

    loaders: Dict[str, DataLoader] = {}
    for split, (X, y) in arrays.items():
        is_train = split == "train"
        loaders[split] = DataLoader(
            make_dataset(X, y),
            batch_size=batch_size if is_train else eval_batch_size,
            shuffle=is_train and shuffle_train,
            num_workers=num_workers,
            drop_last=False,
            generator=generator if is_train and shuffle_train else None,
        )
    return loaders


def class_weights(y: np.ndarray, n_classes: Optional[int] = None) -> torch.Tensor:
    """Return balanced class weights, ``n / (k · n_class)``, as a float tensor.

    Identical to the formula used by :func:`ecg.preprocessing.class_weights` for
    the Spark models, so both families see the same correction for the 113:1
    imbalance.
    """
    n_classes = n_classes or int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = np.nan  # avoid a divide-by-zero for absent classes
    weights = len(y) / (n_classes * counts)
    return torch.tensor(np.nan_to_num(weights, nan=0.0), dtype=torch.float32)


def class_counts(y: np.ndarray, label_names: Sequence[str]) -> Dict[str, int]:
    """Return ``{class_name: count}`` for a label array."""
    counts = np.bincount(y, minlength=len(label_names))
    return {name: int(counts[index]) for index, name in enumerate(label_names)}


def describe_arrays(
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]], label_names: Sequence[str]
):
    """Return a small pandas summary of what was loaded, for the notebooks."""
    import pandas as pd

    rows = []
    for split, (X, y) in arrays.items():
        row: Dict[str, object] = {"split": split, "n_beats": len(y), "samples": X.shape[1]}
        row.update(class_counts(y, label_names))
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int = 42,
    min_per_class: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a class-proportional random subset of a split.

    Used for the low-data experiment in :func:`ecg.transfer.low_data_curve`, where
    the question is how the arms behave when the target dataset is genuinely
    small. Sampling per class keeps the prior intact, so a shrinking training set
    does not silently become a different problem.

    Args:
        X: Waveforms.
        y: Labels.
        fraction: Share of each class to keep, in ``(0, 1]``.
        seed: Random seed.
        min_per_class: Floor on the number of beats kept per class.

    Returns:
        ``(X_subset, y_subset)``, shuffled.

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]``.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return X, y

    rng = np.random.default_rng(seed)
    keep: list = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        n_keep = max(min_per_class, int(round(len(indices) * fraction)))
        n_keep = min(n_keep, len(indices))
        keep.append(rng.choice(indices, n_keep, replace=False))

    selected = np.concatenate(keep)
    rng.shuffle(selected)
    return X[selected], y[selected]
