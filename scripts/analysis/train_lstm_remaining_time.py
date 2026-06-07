"""
train_lstm_remaining_time.py
============================
Task 2 (remaining-time prediction) with an LSTM over the raw event sequence.

Same case-level split and the same metrics (MAE / RMSE / MAPE in hours, MAPE with
eps = 1.0 h) as train_remaining_time.py, so the LSTM is directly comparable to
the mean baselines and the LightGBM regressor.

Input : data/lstm_sequences.npz  (from make_sequence_dataset.py)
Output: models/lstm_remaining_time.pt
        results/results_lstm_remaining_time.json
        results/plots/lstm_remaining_time_scatter.png
Run   : python scripts/analysis/train_lstm_remaining_time.py
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg
import lstm_ppm as L

SEQ_PATH = os.path.join(cfg.DATA_DIR, "lstm_sequences.npz")
MODEL_PATH = os.path.join(cfg.MODELS_DIR, "lstm_remaining_time.pt")
RESULTS_PATH = os.path.join(cfg.RESULTS_DIR, "results_lstm_remaining_time.json")
PLOT_PATH = os.path.join(cfg.RESULTS_DIR, "plots", "lstm_remaining_time_scatter.png")

BATCH = 512
MAX_EPOCHS = int(os.environ.get("PPM_LSTM_EPOCHS", "12"))
PATIENCE = 3
LR = 1e-3


def mape(y_true, y_pred, eps=1.0):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100


def predict(model, data, idx, delta_n, static_n):
    model.eval()
    outs = []
    with torch.no_grad():
        for b in L.iterate_batches(idx, 4096, shuffle=False):
            ids_b = torch.from_numpy(data["ids"][b]).long()
            d_b = torch.from_numpy(delta_n[b])
            s_b = torch.from_numpy(static_n[b])
            l_b = torch.from_numpy(data["lengths"][b])
            outs.append(model(ids_b, d_b, l_b, s_b).squeeze(-1).numpy())
    return np.concatenate(outs)


def evaluate(model, data, idx, delta_n, static_n):
    pred = predict(model, data, idx, delta_n, static_n)
    yt = data["y_rem_h"][idx]
    return {
        "mae_hours": float(mean_absolute_error(yt, pred)),
        "rmse_hours": float(np.sqrt(mean_squared_error(yt, pred))),
        "mape_pct": float(mape(yt, pred)),
    }, pred


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    t0 = time.time()

    print("Loading sequence tensors...")
    data = L.build_tensors(SEQ_PATH, max_len=L.DEFAULT_MAX_LEN)
    split = data["split"]
    delta_n, static_n = L.normalise(data, split == 0)
    n_static = static_n.shape[1]

    tr_idx = np.where(split == 0)[0]
    va_idx = np.where(split == 1)[0]
    te_idx = np.where(split == 2)[0]
    print(f"  train {len(tr_idx):,} | val {len(va_idx):,} | test {len(te_idx):,}")

    y = data["y_rem_h"]  # hours

    model = L.LSTMModel(emb_dim=32, hidden=cfg.LSTM_HIDDEN_DIM,
                        layers=cfg.LSTM_NUM_LAYERS, dropout=cfg.LSTM_DROPOUT,
                        n_static=n_static, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.SmoothL1Loss()  # Huber, robust to the long tail

    best_mae, best_state, bad = 1e9, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        te = time.time()
        running = 0.0
        for bi, b in enumerate(L.iterate_batches(tr_idx, BATCH, shuffle=True, seed=cfg.RANDOM_SEED + epoch)):
            ids_b = torch.from_numpy(data["ids"][b]).long()
            d_b = torch.from_numpy(delta_n[b])
            s_b = torch.from_numpy(static_n[b])
            l_b = torch.from_numpy(data["lengths"][b])
            y_b = torch.from_numpy(y[b])
            opt.zero_grad()
            loss = crit(model(ids_b, d_b, l_b, s_b).squeeze(-1), y_b)
            loss.backward()
            opt.step()
            running += loss.item()
            if bi % 1000 == 0:
                print(f"    epoch {epoch} batch {bi}/{len(tr_idx)//BATCH} loss {loss.item():.4f}", flush=True)

        val_metrics, _ = evaluate(model, data, va_idx, delta_n, static_n)
        print(f"  [epoch {epoch}] train_loss {running/(len(tr_idx)//BATCH):.4f} | "
              f"val_MAE {val_metrics['mae_hours']:.3f}h | val_RMSE {val_metrics['rmse_hours']:.3f}h | "
              f"{time.time()-te:.0f}s", flush=True)

        if val_metrics["mae_hours"] < best_mae:
            best_mae, bad = val_metrics["mae_hours"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    print("Evaluating on test...")
    test_metrics, test_pred = evaluate(model, data, te_idx, delta_n, static_n)
    print(f"  TEST  MAE {test_metrics['mae_hours']:.3f}h | RMSE {test_metrics['rmse_hours']:.3f}h | "
          f"MAPE {test_metrics['mape_pct']:.1f}%")

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    torch.save(best_state, MODEL_PATH)

    # scatter (sample 5000) to mirror the LightGBM plot
    yt = data["y_rem_h"][te_idx]
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    s = rng.choice(len(yt), size=min(5000, len(yt)), replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(yt[s], test_pred[s], s=4, alpha=0.3)
    lim = max(yt[s].max(), test_pred[s].max())
    plt.plot([0, lim], [0, lim], "r--", label="Perfect prediction")
    plt.xlabel("Actual remaining time (hours)")
    plt.ylabel("Predicted remaining time (hours)")
    plt.title("LSTM Remaining Time: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)

    results = {
        "model": "LSTM", "task": "remaining_time",
        "max_len": L.DEFAULT_MAX_LEN, "epochs_run": epoch,
        "best_val_mae_hours": round(best_mae, 4),
        "test": {k: round(v, 4) for k, v in test_metrics.items()},
        "hyperparams": {"hidden": cfg.LSTM_HIDDEN_DIM, "layers": cfg.LSTM_NUM_LAYERS,
                        "dropout": cfg.LSTM_DROPOUT, "lr": LR, "batch": BATCH,
                        "loss": "SmoothL1"},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved results -> {RESULTS_PATH}")
    print(f"Total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
