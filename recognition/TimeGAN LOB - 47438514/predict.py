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


def kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray, nbins: int = 80, eps: float = 1e-8) -> float:
    """KL(real || synth) on histograms that share a support window."""
    lo = float(min(np.min(p_samples), np.min(q_samples)))
    hi = float(max(np.max(p_samples), np.max(q_samples)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    p_hist, _ = np.histogram(p_samples, bins=nbins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q_samples, bins=nbins, range=(lo, hi), density=True)
    p = p_hist + eps
    q = q_hist + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


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

    # evaluation accumulators
    real_mid, real_spread, synth_mid, synth_spread = [], [], [], []
    heatmap_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    # Choose global window indices for heatmaps
    heat_idxs = set(np.linspace(0, len(Xhat_test) - 1, num=max(1, args.n_heatmaps), dtype=int).tolist())

    # Iterate test split and align synthetic windows by pointer
    synth_ptr = 0
    for i, X in enumerate(test_loader):
        # inverse-scale real
        X_inv = scaler.inverse_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # match synthetic slice for this batch
        B = X.shape[0]
        synth_chunk = Xhat_test[synth_ptr:synth_ptr + B]
        synth_ptr += B
        Xhat_inv = scaler.inverse_transform(synth_chunk.reshape(-1, synth_chunk.shape[-1])).reshape(synth_chunk.shape)

        # KL features (flatten)
        r_mid = X_inv[..., feat_idx["mid_delta_ticks"]].numpy().ravel()
        r_spr = X_inv[..., feat_idx["spread_ticks"]].numpy().ravel()
        s_mid = Xhat_inv[..., feat_idx["mid_delta_ticks"]].numpy().ravel()
        s_spr = Xhat_inv[..., feat_idx["spread_ticks"]].numpy().ravel()
        real_mid.append(r_mid); real_spread.append(r_spr)
        synth_mid.append(s_mid); synth_spread.append(s_spr)

        # heatmaps (first element of selected global windows)
        base_idx = synth_ptr - B
        if args.depth_levels > 0 and base_idx in heat_idxs:
            Rimg = extract_depth_matrix(X_inv[0], feat_idx, K=args.depth_levels).numpy()
            Simg = extract_depth_matrix(Xhat_inv[0], feat_idx, K=args.depth_levels).numpy()
            heatmap_pairs.append((minmax01(Rimg), minmax01(Simg)))

    # Concatenate and compute KL
    real_mid = np.concatenate(real_mid); real_spread = np.concatenate(real_spread)
    synth_mid = np.concatenate(synth_mid); synth_spread = np.concatenate(synth_spread)
    kl_mid = kl_divergence(real_mid, synth_mid)
    kl_spread = kl_divergence(real_spread, synth_spread)

    # Heatmap SSIMs + figures
    os.makedirs(args.out_dir, exist_ok=True)
    ssim_scores = []
    for j, (Rimg, Simg) in enumerate(heatmap_pairs[: args.n_heatmaps]):
        score = ssim2d(Rimg, Simg)
        ssim_scores.append(score)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].imshow(Rimg, aspect='auto', origin='lower'); ax[0].set_title('Real depth'); ax[0].set_xlabel('Levels'); ax[0].set_ylabel('Time')
        ax[1].imshow(Simg, aspect='auto', origin='lower'); ax[1].set_title(f'Synth depth  SSIM={score:.3f}'); ax[1].set_xlabel('Levels')
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

    print(f"KL(mid)={kl_mid:.3f}  JS(spread)={kl_spread:.3f}")
    if ssim_scores:
        print("SSIM depth:", ", ".join(f"{s:.3f}" for s in ssim_scores))
        print(f"Saved {len(ssim_scores)} heatmaps to {args.out_dir}")
    print(f"Metrics saved to {os.path.join(args.out_dir, 'predict_metrics.json')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--messages', type=str, required=True)
    ap.add_argument('--orderbook', type=str, required=True)
    ap.add_argument('--depth', type=int, default=10)
    ap.add_argument('--seq_len', type=int, default=200)
    ap.add_argument('--step', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--depth_levels', type=int, default=10, help='use [BidSize1..K, AskSize1..K] for heatmaps & SSIM')
    ap.add_argument('--synth', type=str, default='synth.pt', help='file saved by train.py')
    ap.add_argument('--n_heatmaps', type=int, default=4)
    ap.add_argument('--out_dir', type=str, default='eval_figs')
    args = ap.parse_args()
    main(args)
