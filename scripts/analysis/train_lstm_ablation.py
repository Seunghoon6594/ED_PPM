"""
train_lstm_ablation.py
======================
Input ablation for the next-activity LSTM: which inputs drive its gain?

Trains the same architecture four times under a fixed budget, zeroing inputs:
  full       - event sequence + inter-event time + static context
  no_static  - event sequence + time           (static context zeroed)
  no_time    - event sequence + static          (time gaps zeroed)
  seq_only   - event sequence only              (time + static zeroed)

All variants share the same case-level split, class weights, and epoch budget,
so differences reflect the contribution of each input group. Epoch budget is
deliberately small (relative comparison, not the headline number).

Output: results/ablation_next_activity.json
Run   : python scripts/analysis/train_lstm_ablation.py
Env   : PPM_ABLATION_EPOCHS (default 3)
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg
import lstm_ppm as L

EPOCHS = int(os.environ.get("PPM_ABLATION_EPOCHS", "3"))
BATCH = 512
VARIANTS = {
    "full":      {"static": True,  "time": True},
    "no_static": {"static": False, "time": True},
    "no_time":   {"static": True,  "time": False},
    "seq_only":  {"static": False, "time": False},
}


def train_variant(data, delta_n, static_n, tr_idx, va_idx, te_idx, y, class_weights, use):
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    n_static = static_n.shape[1]
    zeros_static = np.zeros_like(static_n)
    zeros_delta = np.zeros_like(delta_n)
    S = static_n if use["static"] else zeros_static
    D = delta_n if use["time"] else zeros_delta

    model = L.LSTMModel(emb_dim=32, hidden=cfg.LSTM_HIDDEN_DIM, layers=cfg.LSTM_NUM_LAYERS,
                        dropout=cfg.LSTM_DROPOUT, n_static=n_static, out_dim=6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    def predict(idx):
        model.eval(); outs = []
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                b = idx[s:s + 8192]
                outs.append(model(torch.from_numpy(data["ids"][b]).long(),
                                  torch.from_numpy(D[b]),
                                  torch.from_numpy(data["lengths"][b]),
                                  torch.from_numpy(S[b])).argmax(1).numpy())
        return np.concatenate(outs)

    best_f1, best_state = -1, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in L.iterate_batches(tr_idx, BATCH, shuffle=True, seed=cfg.RANDOM_SEED + epoch):
            opt.zero_grad()
            loss = crit(model(torch.from_numpy(data["ids"][b]).long(),
                              torch.from_numpy(D[b]),
                              torch.from_numpy(data["lengths"][b]),
                              torch.from_numpy(S[b])),
                        torch.from_numpy(y[b]))
            loss.backward(); opt.step()
        vp = predict(va_idx)
        f1 = f1_score(y[va_idx], vp, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    tp = predict(te_idx)
    return {"accuracy": round(accuracy_score(y[te_idx], tp), 4),
            "f1_weighted": round(f1_score(y[te_idx], tp, average="weighted"), 4),
            "f1_macro": round(f1_score(y[te_idx], tp, average="macro"), 4)}


def main():
    t0 = time.time()
    data = L.build_tensors(os.path.join(cfg.DATA_DIR, "lstm_sequences.npz"))
    delta_n, static_n = L.normalise(data, data["split"] == 0)
    tr_idx = np.where(data["split"] == 0)[0]
    va_idx = np.where(data["split"] == 1)[0]
    te_idx = np.where(data["split"] == 2)[0]
    y = data["y_next"]
    counts = np.bincount(y[tr_idx], minlength=6).astype(np.float64)
    class_weights = torch.tensor(counts.sum() / (6 * np.maximum(counts, 1)), dtype=torch.float32)

    results = {"epochs": EPOCHS}
    for name, use in VARIANTS.items():
        te = time.time()
        results[name] = train_variant(data, delta_n, static_n, tr_idx, va_idx, te_idx,
                                       y, class_weights, use)
        print(f"  [{name}] {results[name]} ({time.time()-te:.0f}s)", flush=True)

    path = os.path.join(cfg.RESULTS_DIR, "ablation_next_activity.json")
    json.dump(results, open(path, "w"), indent=2)
    print(f"Saved -> {path} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
