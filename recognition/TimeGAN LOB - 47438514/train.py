"""
train.py

- Uses dataset.py to build loaders from LOBSTER CSVs
- Implements the three-stage TimeGAN training loop:
    1) Embedder/Recovery pretrain (reconstruction)
    2) Supervised pretrain (H → S(H) next-step)
    3) Joint training (G/S vs D, plus reconstruction + moment losses)
- Adds simple validation (JS divergence on mid-return & spread) and plots loss curves
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_loaders, CONT_KEYS
from modules import TimeGAN, TimeGANConfig

import os, time, json, sys, platform
from collections import defaultdict


# Utils
def rand_like(x):
    return torch.randn_like(x)


def moments_mean_std(x, dims=(0, 1), eps: float = 1e-6):
    # mean & std over batch and time by default
    m = x.mean(dim=dims, keepdim=False)
    v = x.var(dim=dims, unbiased=False, keepdim=False)
    return m, torch.sqrt(v + eps)

def count_params(m):
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    return {"trainable": int(trainable), "total": int(total)}

def gpu_env():
    info = {
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info.update({
            "gpu_index": idx,
            "gpu_name": torch.cuda.get_device_name(idx),
            "cuda_runtime": torch.version.cuda,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(idx))),
        })
    return info

# TimeGAN losses
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
        # reconstruction MSE
        return self.mse(x, x_tilde)


# Simple validation metrics (JS on engineered features)
def js_divergence(p_samples: np.ndarray, q_samples: np.ndarray, nbins: int = 80) -> float:
    """Symmetrized KL on histograms (common range)."""
    lo = float(min(np.min(p_samples), np.min(q_samples)))
    hi = float(max(np.max(p_samples), np.max(q_samples)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    p_hist, edges = np.histogram(p_samples, bins=nbins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q_samples, bins=nbins, range=(lo, hi), density=True)
    eps = 1e-8
    p = (p_hist + eps); p /= p.sum()
    q = (q_hist + eps); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def inverse_scale_continuous(x: torch.Tensor, scaler, feat_idx: dict, cont_keys):
    """Inverse only the continuous dims back to original (engineered) units."""
    x2d = x.reshape(-1, x.shape[-1]).clone()
    if hasattr(scaler, "inverse_transform"):
        x2d = scaler.inverse_transform(x2d)
    return x2d.reshape(x.shape)


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
        depth_levels=args.depth_levels,
    )

    input_dim = len(feat_idx)
    cfg = TimeGANConfig(
        x_dim=input_dim,
        z_dim=input_dim,  # match ref impl (z same dim as x)
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
    opt_E = optim.Adam(e_params + r_params, lr=args.lr)
    opt_G = optim.Adam(g_params, lr=args.lr)
    opt_GS = optim.Adam(g_params, lr=args.lr)
    opt_D = optim.Adam(d_params, lr=args.lr)

    # run log scaffold
    run = {}
    run["config"] = {
        "x_dim": cfg.x_dim, "z_dim": cfg.z_dim, "h_dim": cfg.h_dim,
        "rnn_layers": cfg.rnn_layers, "dropout": cfg.dropout,
        "lr": args.lr, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "step": args.step,
        "depth": args.depth, "depth_levels": getattr(args, "depth_levels", None),
    }
    run["environment"] = gpu_env()

    module_params = {
        "embedder": count_params(E),
        "recovery": count_params(R),
        "generator": count_params(G),
        "supervisor": count_params(S),
        "discriminator": count_params(D),
    }
    total_trainable = sum(v["trainable"] for v in module_params.values())
    total = sum(v["total"] for v in module_params.values())

    run["params"] = dict(module_params)  # copy to keep the module stats clean
    run["params"]["total_trainable"] = total_trainable
    run["params"]["total"] = total



    # save a human-readable architecture dump
    arch_txt = []
    arch_txt += ["=== Embedder ===", str(E), "", "=== Recovery ===", str(R), "",
                "=== Generator ===", str(G), "", "=== Supervisor ===", str(S), "",
                "=== Discriminator ===", str(D), ""]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + "_architecture.txt", "w") as f:
        f.write("\n".join(arch_txt))

    # training strategy
    run["strategy"] = {
        "pretrain_embed_iters": args.iters_pre_embed,
        "pretrain_supervised_iters": args.iters_pre_sup,
        "joint_iters": args.iters_joint,
        "loss_weights": {"gamma": args.gamma, "sup_w": args.sup_w, "mom_w": args.mom_w},
        "variant_notes": "Full TimeGAN: reconstruction + supervised + adversarial (with moment matching).",
    }

    # timers and VRAM tracking
    phase_times = defaultdict(float)
    t0_all = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # history holders for plots/metrics
    his_d, his_gadv, his_gsup, his_gmom, his_rec = [], [], [], [], []
    his_js_mid, his_js_spread, his_steps = [], [], []

    steps = {"pretrain_embed": 0, "pretrain_supervised": 0, "joint_GS": 0, "joint_D": 0}

    # Embedding pretrain
    t0 = time.time()
    E.train(); R.train()
    for it in range(args.iters_pre_embed):
        for X in train_loader:
            steps["pretrain_embed"] += 1
            X = X.to(device)
            H = E(X)
            X_tilde = R(H)
            rec = loss_fn.e_rec_loss(X, X_tilde)
            opt_E0.zero_grad(); rec.backward(); opt_E0.step()
        if (it + 1) % args.log_every == 0:
            print(f"[E-pre] it={it+1}/{args.iters_pre_embed} rec={rec.sqrt().item():.4f}")
    phase_times["pretrain_embed"] += time.time() - t0

    # Supervised pretrain
    t0 = time.time()
    G.train(); S.train(); E.eval()  # H computed by frozen E here
    for it in range(args.iters_pre_sup):
        for X in train_loader:
            steps["pretrain_supervised"] += 1
            X = X.to(device)
            with torch.no_grad():
                H = E(X)
            H_sup = S(H)
            sup = loss_fn.g_sup_loss(H, H_sup)
            opt_GS.zero_grad(); sup.backward(); opt_GS.step()
        if (it + 1) % args.log_every == 0:
            print(f"[S-pre] it={it+1}/{args.iters_pre_sup} sup={sup.sqrt().item():.4f}")
    phase_times["pretrain_supervised"] += time.time() - t0

    # Joint training
    t0 = time.time()
    E.train(); R.train(); G.train(); S.train(); D.train()
    for it in range(args.iters_joint):
        # Generator/Supervisor updates (twice per D step)
        for _ in range(2):
            for X in train_loader:
                steps["joint_GS"] += 1
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

                # generator/supervisor losses
                y_fake_g = D(H_hat)     # grads to G/S
                y_fake_e_g = D(E_hat)   # grads to G
                g_adv = loss_fn.g_adv_loss(y_fake_g, y_fake_e_g)
                g_sup = loss_fn.g_sup_loss(H, S(H))
                g_mom = loss_fn.g_moment_loss(X, X_hat)
                g_total = g_adv + args.sup_w * g_sup.sqrt() + args.mom_w * g_mom

                # embedder auxiliary (detach supervised term to avoid graph clash)
                e_rec = loss_fn.e_rec_loss(X, X_tilde)
                e_total = 10.0 * e_rec.sqrt() + 0.1 * g_sup.detach()

                opt_G.zero_grad(); g_total.backward(retain_graph=True); opt_G.step()
                opt_E.zero_grad(); e_total.backward(); opt_E.step()

        # Discriminator step (gated as in TF ref)
        for X in train_loader:
            steps["joint_D"] += 1
            X = X.to(device)
            B, N, F = X.shape
            Z = torch.randn(B, N, z_dim, device=device)
            with torch.no_grad():
                H = E(X); E_hat = G(Z); H_hat = S(E_hat)
            y_real = D(H); y_fake = D(H_hat); y_fake_e = D(E_hat)
            d_total = loss_fn.d_loss(y_real, y_fake, y_fake_e)
            if d_total.item() > 0.15:
                opt_D.zero_grad(); d_total.backward(); opt_D.step()

        if (it + 1) % args.log_every == 0:
            rec_rmse = e_rec.sqrt().item(); gs_rmse = g_sup.sqrt().item()
            his_d.append(d_total.item()); his_gadv.append(g_adv.item())
            his_gsup.append(gs_rmse); his_gmom.append(g_mom.item()); his_rec.append(rec_rmse)
            his_steps.append(it + 1)

            # lightweight validation on val split every val_every
            if (it + 1) % args.val_every == 0:
                js_mid, js_spread = run_validation(val_loader, model, scaler, feat_idx, device, z_dim)
                his_js_mid.append(js_mid); his_js_spread.append(js_spread)
                print(f"[Joint] it={it+1}/{args.iters_joint} d={his_d[-1]:.3f} g_adv={his_gadv[-1]:.3f} "
                      f"g_sup={gs_rmse:.3f} g_mom={his_gmom[-1]:.3f} rec={rec_rmse:.3f} | "
                      f"JS(mid)={js_mid:.3f} JS(spread)={js_spread:.3f}")
            else:
                print(f"[Joint] it={it+1}/{args.iters_joint} d={his_d[-1]:.3f} g_adv={his_gadv[-1]:.3f} "
                      f"g_sup={gs_rmse:.3f} g_mom={his_gmom[-1]:.3f} rec={rec_rmse:.3f}")
    phase_times["joint"] += time.time() - t0

    run["phase_times_sec"] = dict(phase_times)
    run["total_time_sec"] = time.time() - t0_all
    run["num_batches"] = {k: int(v) for k, v in steps.items()}
    run["batches_per_iter"] = len(train_loader)

    if torch.cuda.is_available():
        run["environment"]["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        run["environment"]["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())

    # final dump
    with open(os.path.splitext(args.out)[0] + "_training_report.json", "w") as f:
        json.dump(run, f, indent=2)
    print(f"Saved training report to {os.path.splitext(args.out)[0]}_training_report.json")

    # Synthesis
    E.eval(); R.eval(); G.eval(); S.eval()
    synth = []
    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            B, N, F = X.shape
            Z = torch.randn(B, N, z_dim, device=device)
            X_hat = R(S(G(Z)))
            synth.append(X_hat.cpu())
    synth = torch.cat(synth, dim=0)
    torch.save({
        "synth_windows": synth,
        "feat_idx": feat_idx,
        "cont_keys": CONT_KEYS,
    }, args.out)
    print(f"Saved synthetic windows (feature space) to {args.out}")

    # Plots + val metrics dump
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        t = np.arange(len(his_d))
        ax.plot(t, his_d, label="D loss")
        ax.plot(t, his_gadv, label="G adv")
        ax.plot(t, his_gsup, label="G sup RMSE")
        ax.plot(t, his_gmom, label="G moments")
        ax.plot(t, his_rec, label="Recon RMSE")
        ax.set_xlabel("log steps"); ax.set_ylabel("loss"); ax.legend()
        plt.tight_layout(); plt.savefig("training_curves.png", dpi=160)
        print("Saved training_curves.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

    val_report = {
        "JS_mid_delta": his_js_mid[-1] if his_js_mid else None,
        "JS_spread": his_js_spread[-1] if his_js_spread else None,
        "notes": "JS is symmetrized KL on histograms of inverse-scaled engineered features.",
    }
    with open("val_metrics.json", "w") as f:
        json.dump(val_report, f, indent=2)
    print("Saved val_metrics.json")


# Validation routine (uses val loader)
def run_validation(val_loader, model: TimeGAN, scaler, feat_idx, device, z_dim: int):
    model.eval()
    real_mid, real_spread = [], []
    synth_mid, synth_spread = [], []
    with torch.no_grad():
        for X in val_loader:
            X = X.to(device)
            B, N, F = X.shape
            Z = torch.randn(B, N, z_dim, device=device)
            X_hat = model.recovery(model.supervisor(model.generator(Z)))

            # inverse only continuous dims for both real & synth
            X_inv = inverse_scale_continuous(X.cpu(), scaler, feat_idx, CONT_KEYS)
            Xhat_inv = inverse_scale_continuous(X_hat.cpu(), scaler, feat_idx, CONT_KEYS)

            mid_real = extract_cols(X_inv, feat_idx, ["mid_delta_ticks"]).numpy().ravel()
            spread_real = extract_cols(X_inv, feat_idx, ["spread_ticks"]).numpy().ravel()
            mid_synth = extract_cols(Xhat_inv, feat_idx, ["mid_delta_ticks"]).numpy().ravel()
            spread_synth = extract_cols(Xhat_inv, feat_idx, ["spread_ticks"]).numpy().ravel()

            real_mid.append(mid_real); real_spread.append(spread_real)
            synth_mid.append(mid_synth); synth_spread.append(spread_synth)

    real_mid = np.concatenate(real_mid); real_spread = np.concatenate(real_spread)
    synth_mid = np.concatenate(synth_mid); synth_spread = np.concatenate(synth_spread)

    js_mid = js_divergence(real_mid, synth_mid)
    js_spread = js_divergence(real_spread, synth_spread)
    return js_mid, js_spread


# Argparse
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
    p.add_argument("--val_every", type=int, default=200, help="validate every N logged steps")
    p.add_argument("--depth_levels", type=int, default=10, help="append [BidSize1..K, AskSize1..K] as features")
    p.add_argument("--out", type=str, default="synth.pt")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
