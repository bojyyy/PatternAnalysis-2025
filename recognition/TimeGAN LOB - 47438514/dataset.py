"""
dataset.py
Headerless LOBSTER to windowed Dataset for TimeGAN (fixed-length, tabular).

- Reads messages and orderbook CSVs
- Engineers compact, mostly-stationary features:
    one-hot type {1,2,3,4,5,7}, side∈{0,1}, mid_delta_ticks, spread_ticks, log1p(size), log1p(delta t ms)
- Optional: appends depth sizes for K levels: [BidSize1..K, AskSize1..K] (log1p)
- Scales continuous columns only to [0,1]
- Yields windows without materializing all windows in RAM
- Helpers to fit scaler on train rows and build loaders
- Helper to extract a [T, 2K] depth matrix for heatmaps/SSIM
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Tensor = torch.Tensor


def msg_names() -> List[str]:
    return ["Time", "Type", "OrderID", "Size", "Price", "Direction"]


def ob_names(depth: int) -> List[str]:
    names: List[str] = []
    for L in range(1, depth + 1):
        names += [f"AskPrice{L}", f"AskSize{L}", f"BidPrice{L}", f"BidSize{L}"]
    return names


def depth_size_keys(K: int) -> List[str]:
    return [f"BidSize{i}" for i in range(1, K + 1)] + [f"AskSize{i}" for i in range(1, K + 1)]


# Preprocessing
TYPE_CLASSES = [1, 2, 3, 4, 5, 7]  # keep 7 (halt) for completeness
CONT_KEYS = ["mid_delta_ticks", "log_size", "log_dt"]  # base continuous keys
TICK_SIZE = 100.0
CLIP_TICKS = 50


def _onehot(types: np.ndarray, classes=TYPE_CLASSES) -> np.ndarray:
    m = {c: i for i, c in enumerate(classes)}
    idx = np.array([m.get(int(t), 0) for t in types], dtype=np.int64)
    return np.eye(len(classes), dtype=np.float32)[idx]


def preprocess(msgs: pd.DataFrame, ob: pd.DataFrame, depth_levels: int = 0) -> Tuple[Tensor, Dict[str, int]]:
    """Return engineered features [T,F] (torch.float32) and a name to index map."""
    assert len(msgs) == len(ob), "messages and orderbook must align"

    # Δt (ms) → log1p
    t_ms = msgs["Time"].to_numpy(np.float64) * 1000.0
    dt = np.diff(t_ms, prepend=t_ms[0])
    log_dt = np.log1p(np.maximum(dt, 0.0)).astype(np.float32)

    # side {0,1} from direction -1/+1
    side = ((msgs["Direction"].to_numpy(np.int32) + 1) // 2).astype(np.float32)

    # message size → log1p
    log_size = np.log1p(msgs["Size"].to_numpy(np.float64)).astype(np.float32)

    # top-of-book mid / spread → ticks
    ask1 = ob["AskPrice1"].to_numpy(np.float64)
    bid1 = ob["BidPrice1"].to_numpy(np.float64)
    mid = 0.5 * (ask1 + bid1)
    spread = (ask1 - bid1)
    prev_mid = np.roll(mid, 1); prev_mid[0] = mid[0]
    mid_delta_ticks = ((mid - prev_mid) / TICK_SIZE).astype(np.float32)
    mid_delta_ticks = np.clip(mid_delta_ticks, -CLIP_TICKS, CLIP_TICKS)
    spread_ticks = (spread / TICK_SIZE).astype(np.float32)

    # type one-hot
    type_oh = _onehot(msgs["Type"].to_numpy())

    # base features
    X = np.column_stack([type_oh, side, mid_delta_ticks, spread_ticks, log_size, log_dt]).astype(np.float32)

    # feature index map
    idx: Dict[str, int] = {f"type_{c}": i for i, c in enumerate(TYPE_CLASSES)}
    k = len(TYPE_CLASSES)
    idx.update({
        "side": k,
        "mid_delta_ticks": k + 1,
        "spread_ticks": k + 2,
        "log_size": k + 3,
        "log_dt": k + 4,
    })

    # optional: append depth sizes (log1p) for K levels, used for heatmaps/SSIM
    K = int(depth_levels)
    if K > 0:
        cols = depth_size_keys(K)                    # [BidSize1..K, AskSize1..K]
        sizes = ob[cols].to_numpy(np.float32)        # [T, 2K]
        sizes = np.log1p(sizes)                      # compress; continuous, will be scaled
        X = np.column_stack([X, sizes]).astype(np.float32)

        base = len(idx)
        for j, name in enumerate(cols):
            idx[name] = base + j

    return torch.from_numpy(X), idx


@dataclass
class ContinuousMinMax:
    idx: List[int]
    min_: Optional[Tensor] = None
    max_: Optional[Tensor] = None
    eps: float = 1e-7

    def fit(self, X: Tensor) -> "ContinuousMinMax":
        sub = X[:, self.idx]
        self.min_ = sub.min(dim=0).values
        self.max_ = sub.max(dim=0).values
        return self

    def transform(self, X: Tensor) -> Tensor:
        X = X.clone()
        X[:, self.idx] = (X[:, self.idx] - self.min_) / (self.max_ - self.min_ + self.eps)
        return X

    def inverse_transform(self, X: Tensor) -> Tensor:
        X = X.clone()
        X[:, self.idx] = X[:, self.idx] * (self.max_ - self.min_ + self.eps) + self.min_
        return X


class LOBWindowDataset(Dataset):
    """Headerless LOBSTER CSVs to windowed tensors [N,F]."""
    def __init__(self,
                 messages_csv: str,
                 orderbook_csv: str,
                 depth: int,
                 seq_len: int,
                 step: int,
                 row_start: Optional[int],
                 row_end: Optional[int],
                 scaler: Optional[ContinuousMinMax],
                 depth_levels: int = 0):
        # Read with names
        msgs = pd.read_csv(messages_csv, header=None, names=msg_names())
        ob   = pd.read_csv(orderbook_csv, header=None, names=ob_names(depth))
        if row_start is not None or row_end is not None:
            msgs = msgs.iloc[row_start:row_end].reset_index(drop=True)
            ob = ob.iloc[row_start:row_end].reset_index(drop=True)

        X, feat_idx = preprocess(msgs, ob, depth_levels=depth_levels)
        self.feature_index = feat_idx

        self.X = scaler.transform(X) if scaler is not None else X
        self.seq_len = int(seq_len)
        self.step = int(step)

        T = self.X.size(0)
        if T < self.seq_len:
            raise ValueError("sequence shorter than seq_len")
        self.W = (T - self.seq_len) // self.step + 1

    def __len__(self) -> int:
        return self.W

    def __getitem__(self, i: int) -> Tensor:
        s = i * self.step
        e = s + self.seq_len
        return self.X[s:e]


def _row_splits(n_rows: int, train_frac=0.7, val_frac=0.15) -> Tuple[int, int]:
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return train_end, val_end


def build_loaders(messages_csv: str,
                  orderbook_csv: str,
                  depth: int = 10,
                  seq_len: int = 200,
                  step: int = 50,
                  batch_size: int = 128,
                  shuffle_train: bool = True,
                  depth_levels: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader, ContinuousMinMax, Dict[str, int]]:
    # Count rows quickly from messages
    n_rows = len(pd.read_csv(messages_csv, header=None, usecols=[0]))
    train_end, val_end = _row_splits(n_rows)

    # Fit scaler on TRAIN rows only
    msgs_train = pd.read_csv(messages_csv, header=None, names=msg_names()).iloc[:train_end]
    ob_train = pd.read_csv(orderbook_csv, header=None, names=ob_names(depth)).iloc[:train_end]
    X_train, feat_idx = preprocess(msgs_train, ob_train, depth_levels=depth_levels)

    cont_keys = CONT_KEYS.copy()
    if depth_levels > 0:
        cont_keys += depth_size_keys(depth_levels)   # add depth sizes to scaling set

    cont_idx = [feat_idx[k] for k in cont_keys]
    scaler = ContinuousMinMax(cont_idx).fit(X_train)

    # Datasets
    ds_tr = LOBWindowDataset(messages_csv, orderbook_csv, depth, seq_len, step,
                             0, train_end, scaler, depth_levels=depth_levels)
    ds_va = LOBWindowDataset(messages_csv, orderbook_csv, depth, seq_len, step,
                             train_end, val_end, scaler, depth_levels=depth_levels)
    ds_te = LOBWindowDataset(messages_csv, orderbook_csv, depth, seq_len, step,
                             val_end, None, scaler, depth_levels=depth_levels)

    # Loaders
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False,       drop_last=True)
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False,       drop_last=False)

    return train_loader, val_loader, test_loader, scaler, feat_idx


# Helper for evaluation: extract [T, 2K] depth-size matrix from a window
def extract_depth_matrix(window: Tensor, feat_idx: Dict[str, int], K: int) -> Tensor:
    """window: [T,F] → [T, 2K] using [BidSize1..K, AskSize1..K]."""
    cols = depth_size_keys(K)
    idxs = [feat_idx[c] for c in cols if c in feat_idx]
    if len(idxs) != 2 * K:
        raise ValueError("Depth sizes not present in features; train with depth_levels=K.")
    return window[..., idxs]
