"""
lstm_ppm.py
===========
Shared building blocks for the LSTM-based PPM models (next-activity and
remaining-time). Loads the ragged sequence dataset produced by
make_sequence_dataset.py and materialises fixed-length, left-padded tensors so
that training is pure tensor indexing (fast on CPU).

Used by:
  - train_lstm_next_activity.py
  - train_lstm_remaining_time.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

N_EVENTS = 6          # ED_ARRIVAL ... ED_END (ids 1..6, 0 = PAD)
DEFAULT_MAX_LEN = 20  # keep the last 20 events of each prefix (covers ~99% of prefixes)


def build_tensors(npz_path, max_len=DEFAULT_MAX_LEN):
    """Vectorised construction of left-aligned, right-padded sequence tensors.

    Returns a dict of numpy arrays (one row per prefix), all aligned:
      ids[N, T]      event ids (0 = PAD), chronological, padded on the right
      delta[N, T]    raw minute gaps before each event
      lengths[N]     true (unpadded) length of each prefix window
      static[N, S]   per-case static context
      y_next[N]      next-activity label (0..5)
      y_rem_h[N]     remaining time in hours
      split[N]       0=train, 1=val, 2=test
    """
    d = np.load(npz_path, allow_pickle=True)
    event_ids = d["event_ids"].astype(np.int64)
    time_deltas = d["time_deltas"].astype(np.float32)
    remaining = d["remaining"].astype(np.float32)
    offsets = d["offsets"]
    static_all = d["static"].astype(np.float32)
    p_case = d["prefix_case_idx"]
    p_k = d["prefix_k"].astype(np.int64)
    p_split = d["prefix_split"].astype(np.int64)
    static_cols = list(d["static_cols"])

    n = len(p_k)
    T = max_len
    ends = offsets[p_case] + p_k             # exclusive end of prefix == index of next event
    tlen = np.minimum(p_k, T)                # window length actually used
    starts = ends - tlen                     # global start of the (last-T) window

    cols = np.arange(T)[None, :]             # [1, T]
    valid = cols < tlen[:, None]             # [N, T]
    pos = starts[:, None] + cols             # [N, T] global positions
    pos = np.where(valid, pos, 0)

    ids = np.where(valid, event_ids[pos], 0).astype(np.int64)
    delta = np.where(valid, time_deltas[pos], 0.0).astype(np.float32)

    y_next = (event_ids[ends] - 1).astype(np.int64)          # 0..5
    y_rem_h = (remaining[ends - 1] / 60.0).astype(np.float32)  # remaining at last observed event
    static = static_all[p_case]

    return {
        "ids": ids, "delta": delta, "lengths": tlen.astype(np.int64),
        "static": static, "y_next": y_next, "y_rem_h": y_rem_h,
        "split": p_split, "static_cols": static_cols,
    }


def normalise(data, train_mask):
    """Log-transform deltas and standardise (delta + static) with TRAIN stats.

    Returns float32 delta and static arrays plus the fitted stats (for the record).
    NaNs in static are imputed with the train mean before standardising.
    """
    # per-event time gap: log1p then standardise on valid (non-pad) train entries
    logd = np.log1p(np.maximum(data["delta"], 0.0))
    valid = data["delta"] != 0.0
    tr_valid = valid & train_mask[:, None]
    dm = float(logd[tr_valid].mean()) if tr_valid.any() else 0.0
    ds = float(logd[tr_valid].std()) if tr_valid.any() else 1.0
    ds = ds if ds > 1e-6 else 1.0
    delta_n = ((logd - dm) / ds).astype(np.float32)
    delta_n[~valid] = 0.0  # keep pad positions at 0

    # static: impute NaN with train mean, then standardise
    static = data["static"].copy()
    tr = static[train_mask]
    means = np.nanmean(tr, axis=0)
    inds = np.where(np.isnan(static))
    static[inds] = np.take(means, inds[1])
    tr = static[train_mask]
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    static_n = ((static - mu) / sd).astype(np.float32)

    return delta_n, static_n


class LSTMModel(nn.Module):
    """Embedding -> LSTM over the event sequence, fused with static context."""

    def __init__(self, n_events=N_EVENTS, emb_dim=32, hidden=128, layers=2,
                 dropout=0.3, n_static=13, out_dim=6):
        super().__init__()
        self.emb = nn.Embedding(n_events + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim + 1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.static_fc = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, ids, delta, lengths, static):
        x = torch.cat([self.emb(ids), delta.unsqueeze(-1)], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h_last = h[-1]                         # top-layer last hidden state [B, hidden]
        z = torch.cat([h_last, self.static_fc(static)], dim=-1)
        return self.head(z)


def iterate_batches(idx, batch_size, shuffle, seed=0):
    """Yield arrays of row indices for manual mini-batching (fast on CPU)."""
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = idx.copy()
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield idx[s:s + batch_size]
