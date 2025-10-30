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

    
class PatchDisc(nn.Module):
    """
    Feature-space patch discriminator.
    Expects x of shape (B, T, C). We treat C as channels and convolve over time.
    """
    def __init__(self, c_in: int, k_t: int = 5, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # input: (B, C, T, 1)
            nn.Conv2d(c_in, ch, kernel_size=(k_t, 1), padding=(k_t // 2, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, kernel_size=(1, 1))  # logits map
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T, 1)
        x = x.transpose(1, 2).unsqueeze(-1)
        out = self.net(x)              # (B, 1, T, 1)
        return out.mean(dim=(2, 3))    # (B, 1) scalar logit per sample



def build_modules(cfg: TimeGANConfig):
    embedder = Embedder(cfg.x_dim, cfg.h_dim, cfg.rnn_layers, cfg.dropout)
    recovery = Recovery(cfg.h_dim, cfg.x_dim, cfg.rnn_layers, cfg.dropout)
    generator = Generator(cfg.z_dim, cfg.h_dim, cfg.rnn_layers, cfg.dropout)
    supervisor = Supervisor(cfg.h_dim, cfg.rnn_layers, cfg.dropout)
    discriminator = Discriminator(cfg.h_dim)
    return embedder, recovery, generator, supervisor, discriminator


class TimeGAN(nn.Module):
    """Wrapper that groups the TimeGAN components.

    Exposes a small, inference-friendly API:
    - encode(x) : real → latent
    - reconstruct(x) : autoencoder path
    - generate(B, T) : noise → synthetic sequence 
    """
    def __init__(self, cfg: TimeGANConfig):
        super().__init__()
        self.cfg = cfg
        (self.embedder,
         self.recovery,
         self.generator,
         self.supervisor,
         self.discriminator) = build_modules(cfg)

    @torch.no_grad()
    def sample_noise(self, batch: int, seq_len: int) -> torch.Tensor:
        return torch.randn(batch, seq_len, self.cfg.z_dim, device=next(self.parameters()).device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedder(x)
        return self.recovery(h)

    @torch.no_grad()
    def generate(self, batch: int, seq_len: int) -> torch.Tensor:
        z = self.sample_noise(batch, seq_len)
        h_tilde = self.generator(z)
        h_hat = self.supervisor(h_tilde)
        x_hat = self.recovery(h_hat)
        return x_hat
 
__all__ = [
    "TimeGANConfig",
    "LSTMBlock",
    "Embedder",
    "Recovery",
    "Generator",
    "Supervisor",
    "Discriminator",
    "build_modules",
    "TimeGAN",
]