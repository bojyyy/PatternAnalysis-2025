"""
predict.py

Evaluate a trained TimeGAN run by using the
saved synthetic windows from train.py (synth.pt). Computes:
  - KL divergence between real vs synthetic distributions of mid_return & spread
  - SSIM between heatmaps (time x depth-level sizes) for a few representative windows
Also saves side-by-side heatmap images.

Usage example:
    python predict.py \
      --messages AMZN_2012-06-21_34200000_57600000_message_10.csv \
      --orderbook AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
      --depth 10 --seq_len 200 --step 50 \
      --depth_levels 10 --synth synth.pt --n_heatmaps 4
"""
from __future__ import annotations
import argparse, json, os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import build_loaders, CONT_KEYS, extract_depth_matrix

from skimage.metrics import structural_similarity as skimage_ssim


def kl_divergence(p_samples, q_samples, nbins=40, eps=1e-12, smooth=True):
    p = np.asarray(p_samples).ravel()
    q = np.asarray(q_samples).ravel()
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if p.size == 0 or q.size == 0:
        return 0.0

    lo = float(min(p.min(), q.min()))
    hi = float(max(p.max(), q.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    # histogram COUNTS (not density) on a common support
    p_hist, edges = np.histogram(p, bins=nbins, range=(lo, hi), density=False)
    q_hist, _     = np.histogram(q, bins=nbins, range=(lo, hi), density=False)

    if smooth and nbins >= 3:
        k = np.array([1.0, 1.0, 1.0], dtype=np.float64) / 3.0
        p_hist = np.convolve(p_hist.astype(np.float64), k, mode="same")
        q_hist = np.convolve(q_hist.astype(np.float64), k, mode="same")

    # No negatives allowed after smoothing
    p_hist = np.clip(p_hist, 0.0, None)
    q_hist = np.clip(q_hist, 0.0, None)

    # Convert to probabilities and ensure strictly positive
    p = p_hist + eps
    q = q_hist + eps
    p /= p.sum()
    q /= q.sum()

    # Mask only truly-zero numerical p’s (shouldn’t happen post-eps)
    mask = p > 0.0

    # Compute KL
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))

def kl_discrete_ticks(real, synth, min_tick=1, max_tick=10, eps=1e-8):
    """
    KL(P || Q) where P,Q are PMFs over integer tick classes [min_tick..max_tick].
    Clips outliers into the end bins and adds eps smoothing for stability.
    """
    r = np.asarray(real).ravel()
    s = np.asarray(synth).ravel()
    r = r[np.isfinite(r)]
    s = s[np.isfinite(s)]
    if r.size == 0 or s.size == 0:
        return 0.0

    # round to nearest tick and clip to the support
    r = np.clip(np.rint(r).astype(int), min_tick, max_tick) - min_tick
    s = np.clip(np.rint(s).astype(int), min_tick, max_tick) - min_tick
    K = max_tick - min_tick + 1

    p = np.bincount(r, minlength=K).astype(np.float64)
    q = np.bincount(s, minlength=K).astype(np.float64)

    p = (p + eps) / (p.sum() + eps * K)
    q = (q + eps) / (q.sum() + eps * K)

    return float(np.sum(p * (np.log(p) - np.log(q))))

def ssim2d(imgA: np.ndarray, imgB: np.ndarray) -> float:
    """Return SSIM in [0,1]. Accepts 2D arrays scaled to [0,1]."""
    if skimage_ssim is not None:
        return float(skimage_ssim(imgA, imgB, data_range=1.0))
    # Tiny fallback: mean-variance-normalized correlation clipped to [0,1]
    A = (imgA - imgA.mean()) / (imgA.std() + 1e-8)
    B = (imgB - imgB.mean()) / (imgB.std() + 1e-8)
    corr = np.mean(A * B)
    return float(np.clip((corr + 1) / 2, 0.0, 1.0))


def minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def main(args):
    # Rebuild loaders & scaler exactly like training did (fit on train rows)
    train_loader, val_loader, test_loader, scaler, feat_idx = build_loaders(
        messages_csv=args.messages,
        orderbook_csv=args.orderbook,
        depth=args.depth,
        seq_len=args.seq_len,
        step=args.step,
        batch_size=args.batch_size,
        shuffle_train=False,
        depth_levels=args.depth_levels,
    )

    # Load synthetic windows produced by train.py
    blob = torch.load(args.synth, map_location="cpu")
    Xhat_test = blob["synth_windows"]  # [num_windows, T, F]
    feat_idx_synth = blob.get("feat_idx", feat_idx)

    # Ensure feature maps match
    if feat_idx_synth != feat_idx:
        print("[warn] feat_idx from synth file differs from dataset; using synth mapping for indexing.")
        feat_idx = feat_idx_synth

    # Build aligned real & synth test tensors (same ordering)
    real_list = []
    for X in test_loader:
        real_list.append(X)
    real_test = torch.cat(real_list, dim=0)                         # (Nw, T, F)
    synth_test = Xhat_test[: real_test.shape[0]]                     # (Nw, T, F)

    # Inverse-scale (scaler should only touch continuous cols; spread excluded)
    R_inv = scaler.inverse_transform(real_test.reshape(-1, real_test.shape[-1])).reshape(real_test.shape)
    S_inv = scaler.inverse_transform(synth_test.reshape(-1, synth_test.shape[-1])).reshape(synth_test.shape)

    # KL over the whole test set (flattened)
    r_mid = R_inv[..., feat_idx["mid_delta_ticks"]].numpy().ravel()
    r_spr = R_inv[..., feat_idx["spread_ticks"]].numpy().ravel()
    s_mid = S_inv[..., feat_idx["mid_delta_ticks"]].numpy().ravel()
    s_spr = S_inv[..., feat_idx["spread_ticks"]].numpy().ravel()
    kl_mid = kl_divergence(r_mid, s_mid)
    kl_spread = kl_discrete_ticks(r_spr, s_spr, min_tick=1, max_tick=10)

    num_w = R_inv.shape[0]
    n = min(args.n_heatmaps, num_w)
    # evenly spaced
    idxs = np.linspace(0, num_w - 1, num=n, dtype=int)

    os.makedirs(args.out_dir, exist_ok=True)
    ssim_scores = []
    for j, idx in enumerate(idxs):
        Rimg = extract_depth_matrix(R_inv[idx], feat_idx, K=args.depth_levels).numpy()
        Simg = extract_depth_matrix(S_inv[idx], feat_idx, K=args.depth_levels).numpy()

        # shared min/max for fair SSIM
        lo = min(Rimg.min(), Simg.min())
        hi = max(Rimg.max(), Simg.max())
        if hi <= lo:
            hi = lo + 1.0
        Rn = (Rimg - lo) / (hi - lo)
        Sn = (Simg - lo) / (hi - lo)

        score = ssim2d(Rn, Sn)
        ssim_scores.append(float(score))

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].imshow(Rn, aspect='auto', origin='lower'); ax[0].set_title('Real depth'); ax[0].set_xlabel('Levels'); ax[0].set_ylabel('Time')
        ax[1].imshow(Sn, aspect='auto', origin='lower'); ax[1].set_title(f'Synth depth  SSIM={score:.3f}'); ax[1].set_xlabel('Levels')
        for a in ax: a.set_yticks([])
        plt.tight_layout(); fig.savefig(os.path.join(args.out_dir, f'heatmap_{j}.png'), dpi=160); plt.close(fig)

    # Dump metrics JSON
    report = {
        'KL_mid_return': float(kl_mid),
        'KL_spread': float(kl_spread),
        'SSIM_depth': [float(s) for s in ssim_scores],
        'notes': 'KL on inverse-scaled engineered features; heatmaps use log1p depth sizes normalized to [0,1].'
    }
    with open(os.path.join(args.out_dir, 'predict_metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"KL(mid)={kl_mid:.3f}  KL(spread)={kl_spread:.3f}")
    if ssim_scores:
        print("SSIM depth:", ", ".join(f"{s:.3f}" for s in ssim_scores))
        print(f"Saved {len(ssim_scores)} heatmaps to {args.out_dir}")
    print(f"Metrics saved to {os.path.join(args.out_dir, 'predict_metrics.json')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--messages', type=str, required=True)
    ap.add_argument('--orderbook', type=str, required=True)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--seq_len', type=int, default=24)
    ap.add_argument('--step', type=int, default=150)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--depth_levels', type=int, default=10, help='use [BidSize1..K, AskSize1..K] for heatmaps & SSIM')
    ap.add_argument('--synth', type=str, default='synth.pt', help='file saved by train.py')
    ap.add_argument('--n_heatmaps', type=int, default=4)
    ap.add_argument('--out_dir', type=str, default='eval_figs')
    args = ap.parse_args()
    main(args)
