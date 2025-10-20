"""
modules.py
The model has four main networks. A generator, discriminator, embedder, and recovery network. There is an additional supervisor network that enforces
step-wise latent transition dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class TimeGANConfig:
    """ Creating a config dataclass for later use
    """
    # Input/latent dims
    x_dim: int
    z_dim: int = 16
    h_dim: int = 64

    # RNN topology
    rnn_layers: int = 2
    dropout: float = 0.0

    lr: float = 1e-3
    beta1: float = 0.5
    beta2: float = 0.9

    # Loss weights 
    lambda_recon: float = 10.0
    lambda_sup: float = 1.0
    lambda_fm: float = 1.0


class LSTMBlock(nn.Module):
    """Wrapper around nn.LSTM with batch_first=True for convenience.

    Input:  x  [B, T, D_in]
    Output: y  [B, T, H]
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return y


class Embedder(nn.Module):
    """X -> H (latent). Typically used with a paired Recovery for reconstruction.
    """
    def __init__(self, x_dim: int, h_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = LSTMBlock(x_dim, h_dim, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)


class Recovery(nn.Module):
    """H -> X (reconstruction/decoder)."""
    def __init__(self, h_dim: int, x_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = LSTMBlock(h_dim, h_dim, num_layers, dropout)
        self.proj = nn.Linear(h_dim, x_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.rnn(h)
        return self.proj(y)


class Generator(nn.Module):
    """Z -> H~ (fake latent sequence)."""
    def __init__(self, z_dim: int, h_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = LSTMBlock(z_dim, h_dim, num_layers, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.rnn(z)


class Supervisor(nn.Module):
    """Temporal supervisor that encourages next-step prediction in latent space.
    Given H or H~, outputs hat H with stronger temporal dependencies.
    """
    def __init__(self, h_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = LSTMBlock(h_dim, h_dim, num_layers, dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.rnn(h)


class Discriminator(nn.Module):
    """Binary classifier on latent sequences.
    Returns per-timestep logits [B, T]
    """
    def __init__(self, h_dim: int):
        super().__init__()
        self.rnn = LSTMBlock(h_dim, h_dim, num_layers=1, dropout=0.0)
        self.out = nn.Linear(h_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.rnn(h)                  # [B, T, H]
        logits = self.out(y).squeeze(-1) # [B, T]
        return logits