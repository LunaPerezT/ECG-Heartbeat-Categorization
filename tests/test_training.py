"""Training loop, data bridge and transfer wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg import torch_data, training, transfer
from ecg.config import Config
from ecg.models import ResidualCNN, count_parameters

LABELS = ["a", "b", "c"]


@pytest.fixture(scope="module")
def tiny_arrays():
    """Three linearly separable classes, small enough to train in seconds."""
    rng = np.random.default_rng(0)
    per_class, length = 60, 187
    X, y = [], []
    for label in range(3):
        base = np.zeros(length, dtype=np.float32)
        base[label * 40 : label * 40 + 30] = 1.0
        X.append(base + rng.normal(0, 0.05, size=(per_class, length)).astype(np.float32))
        y.append(np.full(per_class, label, dtype=np.int64))
    X = np.clip(np.vstack(X), 0, 1)
    y = np.concatenate(y)

    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    cut = int(0.7 * len(y))
    return {
        "train": (X[:cut], y[:cut]),
        "val": (X[cut:], y[cut:]),
        "test": (X[cut:], y[cut:]),
    }


@pytest.fixture(scope="module")
def tiny_loaders(tiny_arrays):
    return torch_data.make_loaders(tiny_arrays, batch_size=32, seed=0)


# --------------------------------------------------------------------- loaders


def test_make_dataset_adds_the_channel_dimension(tiny_arrays):
    X, y = tiny_arrays["train"]
    dataset = torch_data.make_dataset(X, y)
    sample, label = dataset[0]
    assert sample.shape == (1, 187)
    assert label.dtype == torch.int64


def test_loaders_cover_every_row(tiny_arrays, tiny_loaders):
    for split, loader in tiny_loaders.items():
        seen = sum(labels.numel() for _, labels in loader)
        assert seen == len(tiny_arrays[split][1])


def test_class_weights_follow_the_balanced_formula():
    y = np.array([0] * 90 + [1] * 10)
    weights = torch_data.class_weights(y, n_classes=2)
    assert float(weights[0]) == pytest.approx(100 / (2 * 90))
    assert float(weights[1]) == pytest.approx(100 / (2 * 10))
    assert weights[1] > weights[0]


def test_class_weights_tolerate_an_absent_class():
    weights = torch_data.class_weights(np.array([0, 0, 1]), n_classes=3)
    assert float(weights[2]) == 0.0
    assert torch.isfinite(weights).all()


def test_describe_arrays_lists_every_split(tiny_arrays):
    frame = torch_data.describe_arrays(tiny_arrays, LABELS)
    assert set(frame["split"]) == {"train", "val", "test"}
    assert frame["n_beats"].sum() == sum(len(y) for _, y in tiny_arrays.values())


# -------------------------------------------------------------------- training


def test_train_improves_and_records_a_history(tiny_loaders):
    model = ResidualCNN(n_classes=3)
    settings = training.TrainingSettings(max_epochs=8, patience=8, learning_rate=3e-3, seed=0)
    result = training.train(model, tiny_loaders, LABELS, settings=settings, verbose=False)

    assert list(result.history["epoch"]) == list(range(1, 9))
    assert {"train_loss", "val_loss", "val_macro_f1"}.issubset(result.history.columns)
    assert result.best_score > 0.5
    # The *minimum* rather than the last epoch: on 126 synthetic beats the loss
    # oscillates from batch to batch, and asserting monotonicity would be flaky.
    assert result.history["train_loss"].min() < result.history["train_loss"].iloc[0]


def test_train_requires_train_and_val_loaders(tiny_loaders):
    with pytest.raises(KeyError, match="train"):
        training.train(ResidualCNN(3), {"test": tiny_loaders["test"]}, LABELS, verbose=False)


def test_early_stopping_triggers_with_zero_patience(tiny_loaders):
    settings = training.TrainingSettings(max_epochs=10, patience=1, learning_rate=1e-9, seed=0)
    result = training.train(ResidualCNN(3), tiny_loaders, LABELS, settings=settings, verbose=False)
    assert result.stopped_early
    assert len(result.history) < 10


def test_training_restores_the_best_checkpoint(tiny_loaders, tmp_path):
    model = ResidualCNN(n_classes=3)
    settings = training.TrainingSettings(max_epochs=3, learning_rate=3e-3, seed=0)
    path = tmp_path / "best.pt"
    result = training.train(
        model, tiny_loaders, LABELS, settings=settings, checkpoint_path=path, verbose=False
    )

    assert path.exists()
    assert path.with_suffix(".json").exists()

    reloaded = training.load_checkpoint(ResidualCNN(n_classes=3), path)
    y_true, y_pred, _ = training.predict(model, tiny_loaders["val"])
    _, y_pred_reloaded, _ = training.predict(reloaded, tiny_loaders["val"])
    assert np.array_equal(y_pred, y_pred_reloaded)
    assert result.best_epoch >= 1


def test_predict_returns_probabilities_that_sum_to_one(tiny_loaders):
    _, y_pred, y_score = training.predict(ResidualCNN(3), tiny_loaders["val"])
    assert np.allclose(y_score.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(y_pred, y_score.argmax(axis=1))


def test_evaluate_model_produces_a_full_evaluation(tiny_loaders):
    result = training.evaluate_model(ResidualCNN(3), tiny_loaders["test"], LABELS, "m", "test")
    assert set(result) == {"summary", "per_class", "confusion", "confusion_normalised"}


def test_set_seed_makes_initialisation_reproducible():
    training.set_seed(7)
    first = ResidualCNN(3).stem.weight.detach().clone()
    training.set_seed(7)
    assert torch.equal(first, ResidualCNN(3).stem.weight.detach())


def test_settings_from_config_reads_the_project_defaults():
    cfg = Config(max_epochs=11, patience=2, learning_rate=0.5)
    settings = training.TrainingSettings.from_config(cfg)
    assert (settings.max_epochs, settings.patience, settings.learning_rate) == (11, 2, 0.5)


def test_mlflow_run_is_a_no_op_when_disabled():
    with training.mlflow_run("x", enabled=False) as run:
        assert run is None


# -------------------------------------------------------------------- transfer


def test_scratch_arm_needs_no_checkpoint():
    model = transfer.build_arm("scratch", n_classes=2)
    assert model.n_classes == 2
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_transfer_arms_require_a_source_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        transfer.build_arm("frozen", n_classes=2)


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="Unknown arm"):
        transfer.build_arm("magic", n_classes=2)


def test_frozen_arm_keeps_the_source_weights_and_trains_only_the_head(tmp_path):
    source = ResidualCNN(n_classes=5)
    path = tmp_path / "source.pt"
    training.save_checkpoint(source, path)

    frozen = transfer.build_arm("frozen", n_classes=2, source_checkpoint=path)
    assert torch.equal(frozen.stem.weight.detach(), source.stem.weight.detach())
    assert frozen.n_classes == 2
    trainable = count_parameters(frozen, trainable_only=True)
    assert 0 < trainable < count_parameters(frozen)


def test_finetune_arm_leaves_everything_trainable(tmp_path):
    path = tmp_path / "source.pt"
    training.save_checkpoint(ResidualCNN(n_classes=5), path)
    model = transfer.build_arm("finetune", n_classes=2, source_checkpoint=path)
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_transfer_gain_requires_the_scratch_reference():
    with pytest.raises(KeyError, match="scratch"):
        transfer.transfer_gain({})
