"""
train.py

- Uses dataset.py to build loaders from LOBSTER CSVs
- Implements the three-stage TimeGAN training loop:
    1) Embedder/Recovery pretrain (reconstruction)
    2) Supervised pretrain (H → S(H) next-step)
    3) Joint training (G/S vs D, plus reconstruction + moment losses)
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple
import json, numpy as np, matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_loaders, CONT_KEYS
from modules import TimeGAN, TimeGANConfig


# Utils
def rand_like(x):
    return torch.randn_like(x)

def moments_mean_std(x, dims=(0,1), eps: float = 1e-6):
    # mean & std over batch and time by default
    m = x.mean(dim=dims, keepdim=False)
    v = x.var(dim=dims, unbiased=False, keepdim=False)
    return m, torch.sqrt(v + eps)


#TimeGAN losses
@dataclass
class LossWeights:
    gamma: float = 1.0
    sup: float = 100.0
    mom: float = 100.0


class TimeGANLoss:
    def __init__(self, gamma=1.0, sup_w=100.0, mom_w=100.0):
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.gamma = gamma
        self.sup_w = sup_w
        self.mom_w = mom_w

    def d_loss(self, y_real, y_fake, y_fake_e):
        ones = torch.ones_like(y_real)
        zeros = torch.zeros_like(y_fake)
        loss_real = self.bce(y_real, ones)
        loss_fake = self.bce(y_fake, zeros)
        loss_fake_e = self.bce(y_fake_e, zeros)
        return loss_real + loss_fake + self.gamma * loss_fake_e

    def g_adv_loss(self, y_fake, y_fake_e):
        ones_fake = torch.ones_like(y_fake)
        ones_fake_e = torch.ones_like(y_fake_e)
        return self.bce(y_fake, ones_fake) + self.gamma * self.bce(y_fake_e, ones_fake_e)

    def g_sup_loss(self, h, h_sup):
        # next-step supervision: H[:,1:,:] vs S(H)[:,:-1,:]
        return self.mse(h[:, 1:, :], h_sup[:, :-1, :])

    def g_moment_loss(self, x, x_hat):
        m_x, s_x = moments_mean_std(x)
        m_y, s_y = moments_mean_std(x_hat)
        return (m_x - m_y).abs().mean() + (s_x - s_y).abs().mean()

    def e_rec_loss(self, x, x_tilde):
        # reconstruction root-MSE
        return self.mse(x, x_tilde)

def js_divergence(p_samples: np.ndarray, q_samples: np.ndarray, nbins: int = 80) -> float:
    """Symmetrized KL on simple histograms (supports different sample sizes)."""
    # same range for both
    lo = np.nanmin([p_samples.min(), q_samples.min()])
    hi = np.nanmax([p_samples.max(), q_samples.max()])
    if lo == hi:  # degenerate
        return 0.0
    p_hist, edges = np.histogram(p_samples, bins=nbins, range=(lo, hi), density=True)
    q_hist, _     = np.histogram(q_samples, bins=nbins, range=(lo, hi), density=True)
    # smooth to avoid log 0
    eps = 1e-8
    p = (p_hist + eps); p /= p.sum()
    q = (q_hist + eps); q /= q.sum()
    m = 0.5*(p+q)
    kl = lambda a,b: np.sum(a*np.log(a/b))
    return 0.5*kl(p,m) + 0.5*kl(q,m)

def inverse_scale_continuous(x: torch.Tensor, scaler, feat_idx: dict, cont_keys):
    """Inverse only the continuous dims back to original (engineered) units."""
    x_np = x.reshape(-1, x.shape[-1]).clone()
    if hasattr(scaler, "inverse_transform"):
        x_np = scaler.inverse_transform(x_np)
    return x_np.reshape(x.shape)

def extract_cols(x: torch.Tensor, feat_idx: dict, keys):
    idxs = [feat_idx[k] for k in keys]
    return x[..., idxs]


# Train
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Build data loaders + scaler + feature map
    train_loader, val_loader, test_loader, scaler, feat_idx = build_loaders(
        messages_csv=args.messages,
        orderbook_csv=args.orderbook,
        depth=args.depth,
        seq_len=args.seq_len,
        step=args.step,
        batch_size=args.batch_size,
        shuffle_train=True,
    )

    input_dim = len(feat_idx)  
    cfg = TimeGANConfig(
        x_dim=input_dim,
        z_dim=input_dim,     
        h_dim=args.hidden,
        rnn_layers=args.layers,
        dropout=0.0,
    )
    z_dim = cfg.z_dim

    model = TimeGAN(cfg).to(device)
    E = model.embedder
    R = model.recovery
    G = model.generator
    S = model.supervisor
    D = model.discriminator

    loss_fn = TimeGANLoss(gamma=args.gamma, sup_w=args.sup_w, mom_w=args.mom_w)

    # Optims
    e_params = list(E.parameters())
    r_params = list(R.parameters())
    g_params = list(G.parameters()) + list(S.parameters())
    d_params = list(D.parameters())

    opt_E0 = optim.Adam(e_params + r_params, lr=args.lr)
    opt_E  = optim.Adam(e_params + r_params, lr=args.lr)
    opt_G  = optim.Adam(g_params, lr=args.lr)
    opt_GS = optim.Adam(g_params, lr=args.lr)
    opt_D  = optim.Adam(d_params, lr=args.lr)

    # Embedding pretrain
    E.train(); R.train()
    for it in range(args.iters_pre_embed):
        for X in train_loader:
            X = X.to(device)
            H = E(X)
            X_tilde = R(H)
            rec = loss_fn.e_rec_loss(X, X_tilde)
            opt_E0.zero_grad()
            rec.backward()
            opt_E0.step()
        if (it+1) % args.log_every == 0:
            print(f"[E-pre] it={it+1}/{args.iters_pre_embed} rec={rec.sqrt().item():.4f}")

    # Supervised pretrain (G/S only, supervised loss)
    G.train(); S.train(); E.eval()  # H computed by frozen E here
    for it in range(args.iters_pre_sup):
        for X in train_loader:
            X = X.to(device)
            with torch.no_grad():
                H = E(X)
            H_sup = S(H)
            sup = loss_fn.g_sup_loss(H, H_sup)
            opt_GS.zero_grad()
            sup.backward()
            opt_GS.step()
        if (it+1) % args.log_every == 0:
            print(f"[S-pre] it={it+1}/{args.iters_pre_sup} sup={sup.sqrt().item():.4f}")

    # Joint training
    for it in range(args.iters_joint):
        # G/S (twice per D step as in reference implementation)
        for _ in range(2):
            for X in train_loader:
                X = X.to(device)
                B, N, F = X.shape
                Z = torch.randn(B, N, z_dim, device=device)

                # forward
                H = E(X)
                X_tilde = R(H)
                E_hat = G(Z)
                H_hat = S(E_hat)
                X_hat = R(H_hat)

                # discriminator logits
                y_real = D(H)
                y_fake = D(H_hat.detach())
                y_fake_e = D(E_hat.detach())

                y_fake_g = D(H_hat)          # grads to G/S
                y_fake_e_g = D(E_hat)          # grads to G
                g_adv = loss_fn.g_adv_loss(y_fake_g, y_fake_e_g)
                g_sup = loss_fn.g_sup_loss(H, S(H))         # supervised term (with grads)
                g_mom = loss_fn.g_moment_loss(X, X_hat)

                g_total = g_adv + args.sup_w * g_sup.sqrt() + args.mom_w * g_mom

                # auxiliary reconstruction (for E/R) - detach to avoid second backprop through S
                e_rec = loss_fn.e_rec_loss(X, X_tilde)
                e_total = 10.0 * e_rec.sqrt() + 0.1 * g_sup.detach()  

                opt_G.zero_grad()
                g_total.backward(retain_graph=True)
                opt_G.step()

                opt_E.zero_grad()
                e_total.backward()
                opt_E.step()

        # D step 
        for X in train_loader:
            X = X.to(device)
            B, N, F = X.shape
            Z = torch.randn(B, N, z_dim, device=device)
            with torch.no_grad():
                H = E(X)
                E_hat = G(Z)
                H_hat = S(E_hat)
            y_real = D(H)
            y_fake = D(H_hat)
            y_fake_e = D(E_hat)
            d_total = loss_fn.d_loss(y_real, y_fake, y_fake_e)
            if d_total.item() > 0.15:  # heuristic from reference implementation
                opt_D.zero_grad(); d_total.backward(); opt_D.step()

        if (it+1) % args.log_every == 0:
            print(f"[Joint] it={it+1}/{args.iters_joint} d={d_total.item():.4f} g_adv={g_adv.item():.4f} g_sup={g_sup.sqrt().item():.4f} g_mom={g_mom.item():.4f} rec={e_rec.sqrt().item():.4f}")

    # Synthesis
    E.eval(); R.eval(); G.eval(); S.eval()
    synth = []
    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            B, N, F = X.shape
            Z = torch.randn(B, N, z_dim, device=device)
            H_hat = S(G(Z))
            X_hat = R(H_hat)
            synth.append(X_hat.cpu())
    synth = torch.cat(synth, dim=0)
    torch.save({
        "synth_windows": synth,
        "feat_idx": feat_idx,
        "cont_keys": CONT_KEYS,
    }, args.out)
    print(f"Saved synthetic windows (feature space) to {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--messages", type=str, required=True)
    p.add_argument("--orderbook", type=str, required=True)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=200)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters_pre_embed", type=int, default=5)   
    p.add_argument("--iters_pre_sup", type=int, default=5)
    p.add_argument("--iters_joint", type=int, default=10)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--sup_w", type=float, default=100.0)
    p.add_argument("--mom_w", type=float, default=100.0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--out", type=str, default="synth.pt")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
  args = parse_args()
  train(args)