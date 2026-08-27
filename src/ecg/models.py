"""PyTorch architectures for heartbeat classification.

The reference model is the residual 1-D convolutional network of Kachuee, Fazeli
& Sarrafzadeh (`arXiv:1805.00794 <https://arxiv.org/abs/1805.00794>`_), reproduced
here as described in the paper:

* one 1-D convolution over the raw 187-sample beat,
* five residual blocks, each two convolutions with a skip connection followed by
  max pooling,
* two fully-connected layers and a softmax over the classes.

Every convolution has 32 kernels of size 5; every pooling layer has size 5 and
stride 2. The result is a deliberately small network — about 54k parameters —
which is what makes it trainable on a CPU and transferable to the much smaller
PTB collection.

The model consumes the **raw waveform**, not the engineered feature vector: the
EDA in ``notebooks/02_eda_mitbih.ipynb`` showed that no scalar descriptor
correlates with the label above r = 0.27, so the discriminative information is in
the shape of the signal and the convolutions are what read it.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

#: Length of one beat, in samples.
INPUT_LENGTH: int = 187


class ResidualBlock(nn.Module):
    """Two same-padded convolutions, a skip connection and max pooling.

    Args:
        channels: Number of feature maps in and out (the skip connection requires
            them to match).
        kernel_size: Convolution kernel size.
        pool_size: Max-pooling window.
        pool_stride: Max-pooling stride.

    Shape:
        - Input: :math:`(N, C, L)`
        - Output: :math:`(N, C, \\lfloor (L - pool\\_size)/pool\\_stride \\rfloor + 1)`
    """

    def __init__(
        self,
        channels: int = 32,
        kernel_size: int = 5,
        pool_size: int = 5,
        pool_stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool1d(pool_size, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.relu(out + residual)
        return self.pool(out)


def _pooled_length(length: int, n_blocks: int, pool_size: int, pool_stride: int) -> int:
    """Return the sequence length after ``n_blocks`` pooling operations."""
    for _ in range(n_blocks):
        length = (length - pool_size) // pool_stride + 1
    return length


class ResidualCNN(nn.Module):
    """The Kachuee et al. residual 1-D CNN.

    Args:
        n_classes: Number of output classes (5 for MIT-BIH, 2 for PTB).
        in_channels: Input channels; 1 for a single-lead beat.
        channels: Feature maps per convolution.
        kernel_size: Convolution kernel size.
        n_blocks: Number of residual blocks.
        pool_size: Max-pooling window.
        pool_stride: Max-pooling stride.
        hidden_dim: Width of the first fully-connected layer.
        input_length: Samples per beat.
        dropout: Dropout applied before the classifier head; ``0.0`` reproduces
            the paper exactly.

    Example:
        >>> model = ResidualCNN(n_classes=5)
        >>> model(torch.randn(8, 1, 187)).shape
        torch.Size([8, 5])
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_channels: int = 1,
        channels: int = 32,
        kernel_size: int = 5,
        n_blocks: int = 5,
        pool_size: int = 5,
        pool_stride: int = 2,
        hidden_dim: int = 32,
        input_length: int = INPUT_LENGTH,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.input_length = input_length

        self.stem = nn.Conv1d(in_channels, channels, kernel_size, padding="same")
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(channels, kernel_size, pool_size, pool_stride)
                for _ in range(n_blocks)
            ]
        )

        pooled = _pooled_length(input_length, n_blocks, pool_size, pool_stride)
        if pooled < 1:
            raise ValueError(
                f"{n_blocks} pooling blocks reduce a length-{input_length} input to nothing; "
                "use fewer blocks or a shorter stride"
            )
        self.flat_dim = channels * pooled

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if x.dim() == 2:  # (N, L) -> (N, 1, L)
            x = x.unsqueeze(1)
        return self.head(self.blocks(self.stem(x)))

    # ------------------------------------------------------------- transfer API

    @property
    def backbone_modules(self) -> List[nn.Module]:
        """The layers that learn the representation, as opposed to the classifier."""
        return [self.stem, self.blocks]

    def freeze_backbone(self, freeze: bool = True) -> "ResidualCNN":
        """Freeze (or unfreeze) the convolutional stack.

        This is the transfer-learning setup of the source paper: keep the
        representation learned on 109k arrhythmia beats fixed and retrain only the
        two fully-connected layers on the much smaller PTB collection.

        Args:
            freeze: ``True`` to stop gradients in the backbone.

        Returns:
            ``self``, for chaining.
        """
        for module in self.backbone_modules:
            for parameter in module.parameters():
                parameter.requires_grad = not freeze
        return self

    def replace_head(self, n_classes: int, hidden_dim: int = 32) -> "ResidualCNN":
        """Swap the classifier for a freshly initialised one with ``n_classes`` outputs.

        Args:
            n_classes: Number of classes of the new task.
            hidden_dim: Width of the first fully-connected layer.

        Returns:
            ``self``, for chaining.
        """
        dropout = next(
            (module for module in self.head if isinstance(module, nn.Dropout)), None
        )
        layers: List[nn.Module] = [nn.Flatten()]
        if dropout is not None:
            layers.append(nn.Dropout(dropout.p))
        layers += [
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, n_classes),
        ]
        self.head = nn.Sequential(*layers)
        self.n_classes = n_classes
        return self


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of parameters in a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def describe(model: ResidualCNN) -> dict:
    """Return a small dictionary summarising a model, for logging."""
    return {
        "architecture": type(model).__name__,
        "n_classes": model.n_classes,
        "input_length": model.input_length,
        "flat_dim": model.flat_dim,
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }
