"""Architecture contracts for the residual CNN."""

from __future__ import annotations

import pytest
import torch

from ecg.models import INPUT_LENGTH, ResidualBlock, ResidualCNN, count_parameters, describe


@pytest.fixture(scope="module")
def model() -> ResidualCNN:
    return ResidualCNN(n_classes=5)


def test_output_shape(model):
    assert model(torch.randn(8, 1, INPUT_LENGTH)).shape == (8, 5)


def test_accepts_unchannelled_input(model):
    """A (N, L) batch is promoted to (N, 1, L) rather than failing."""
    assert model(torch.randn(4, INPUT_LENGTH)).shape == (4, 5)


def test_residual_block_halves_the_sequence(model):
    block = ResidualBlock(channels=32)
    out = block(torch.randn(2, 32, 187))
    assert out.shape == (2, 32, (187 - 5) // 2 + 1)


def test_five_blocks_reduce_187_to_2_positions(model):
    assert model.flat_dim == 32 * 2


def test_parameter_count_matches_the_paper_scale(model):
    """~54k parameters is what makes this trainable on a CPU and transferable."""
    assert count_parameters(model) == 53_957


def test_too_many_blocks_is_rejected():
    with pytest.raises(ValueError, match="reduce a length"):
        ResidualCNN(n_classes=2, n_blocks=9)


def test_freeze_backbone_stops_gradients_only_in_the_backbone():
    model = ResidualCNN(n_classes=5).freeze_backbone(True)
    assert all(not p.requires_grad for p in model.stem.parameters())
    assert all(not p.requires_grad for p in model.blocks.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    model.freeze_backbone(False)
    assert all(p.requires_grad for p in model.stem.parameters())


def test_frozen_backbone_receives_no_gradient():
    model = ResidualCNN(n_classes=3).freeze_backbone(True)
    loss = model(torch.randn(4, 1, INPUT_LENGTH)).sum()
    loss.backward()
    assert model.stem.weight.grad is None
    assert model.head[-1].weight.grad is not None


def test_replace_head_changes_the_output_size_and_keeps_the_backbone():
    model = ResidualCNN(n_classes=5)
    stem_before = model.stem.weight.detach().clone()

    model.replace_head(n_classes=2)
    assert model.n_classes == 2
    assert model(torch.randn(3, 1, INPUT_LENGTH)).shape == (3, 2)
    assert torch.equal(model.stem.weight.detach(), stem_before)


def test_describe_reports_trainable_separately():
    model = ResidualCNN(n_classes=2).freeze_backbone(True)
    info = describe(model)
    assert info["n_classes"] == 2
    assert info["trainable_parameters"] < info["parameters"]


def test_dropout_is_only_added_when_requested():
    assert not any(isinstance(m, torch.nn.Dropout) for m in ResidualCNN(2).head)
    assert any(isinstance(m, torch.nn.Dropout) for m in ResidualCNN(2, dropout=0.3).head)


def test_forward_is_deterministic_in_eval_mode(model):
    model.eval()
    x = torch.randn(2, 1, INPUT_LENGTH)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))
