"""
train_lstm_next_activity.py
===========================
Task 1 (next-activity prediction) with an LSTM over the raw event sequence.

Same case-level split as the structured prefix dataset, so the LSTM is directly
comparable to the most-frequent / last-event baselines and the LightGBM model.

Input : data/lstm_sequences.npz  (from make_sequence_dataset.py)
Output: models/lstm_next_activity.pt
        results/results_lstm_next_activity.json
        results/confusion_matrix_lstm_next_activity.csv
Run   : python scripts/analysis/train_lstm_next_activity.py
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg
import lstm_ppm as L

SEQ_PATH = os.path.join(cfg.DATA_DIR, "lstm_sequences.npz")
MODEL_PATH = os.path.join(cfg.MODELS_DIR, "lstm_next_activity.pt")
RESULTS_PATH = os.path.join(cfg.RESULTS_DIR, "results_lstm_next_activity.json")
CM_PATH = os.path.join(cfg.RESULTS_DIR, "confusion_matrix_lstm_next_activity.csv")

BATCH = 512
MAX_EPOCHS = int(os.environ.get("PPM_LSTM_EPOCHS", "12"))
PATIENCE = 3
LR = 1e-3
EVENT_NAMES = ["ED_ARRIVAL", "TRIAGE", "VITAL_SIGN_REASSESSMENT",
               "MEDICATION_RECONCILIATION", "MEDICATION_DISPENSED", "ED_END"]


def predict_logits(model, data, idx, delta_n, static_n):
    model.eval()
    outs = []
    with torch.no_grad():
        for b in L.iterate_batches(idx, 4096, shuffle=False):
            ids_b = torch.from_numpy(data["ids"][b]).long()
            d_b = torch.from_numpy(delta_n[b])
            s_b = torch.from_numpy(static_n[b])
            l_b = torch.from_numpy(data["lengths"][b])
            outs.append(model(ids_b, d_b, l_b, s_b).numpy())
    return np.concatenate(outs)


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    t0 = time.time()

    print("Loading sequence tensors...")
    data = L.build_tensors(SEQ_PATH, max_len=L.DEFAULT_MAX_LEN)
    split = data["split"]
    train_mask = split == 0
    delta_n, static_n = L.normalise(data, train_mask)
    n_static = static_n.shape[1]

    tr_idx = np.where(split == 0)[0]
    va_idx = np.where(split == 1)[0]
    te_idx = np.where(split == 2)[0]
    print(f"  train {len(tr_idx):,} | val {len(va_idx):,} | test {len(te_idx):,} | static dims {n_static}")

    event_names = EVENT_NAMES

    # class weights (balanced) from train labels
    y = data["y_next"]
    counts = np.bincount(y[tr_idx], minlength=6).astype(np.float64)
    weights = counts.sum() / (6 * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print("  class counts (train):", dict(zip(event_names, counts.astype(int))))

    model = L.LSTMModel(emb_dim=32, hidden=cfg.LSTM_HIDDEN_DIM,
                        layers=cfg.LSTM_NUM_LAYERS, dropout=cfg.LSTM_DROPOUT,
                        n_static=n_static, out_dim=6)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, bad = -1.0, None, 0
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
            loss = crit(model(ids_b, d_b, l_b, s_b), y_b)
            loss.backward()
            opt.step()
            running += loss.item()
            if bi % 1000 == 0:
                print(f"    epoch {epoch} batch {bi}/{len(tr_idx)//BATCH} loss {loss.item():.4f}", flush=True)

        # validation
        val_logits = predict_logits(model, data, va_idx, delta_n, static_n)
        val_pred = val_logits.argmax(1)
        val_acc = accuracy_score(y[va_idx], val_pred)
        val_f1m = f1_score(y[va_idx], val_pred, average="macro")
        print(f"  [epoch {epoch}] train_loss {running/(len(tr_idx)//BATCH):.4f} | "
              f"val_acc {val_acc:.4f} | val_f1_macro {val_f1m:.4f} | {time.time()-te:.0f}s", flush=True)

        if val_f1m > best_f1:
            best_f1, bad = val_f1m, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # ---- test evaluation ----
    print("Evaluating on test...")
    test_logits = predict_logits(model, data, te_idx, delta_n, static_n)
    test_pred = test_logits.argmax(1)
    yt = y[te_idx]
    acc = accuracy_score(yt, test_pred)
    f1w = f1_score(yt, test_pred, average="weighted")
    f1m = f1_score(yt, test_pred, average="macro")
    print(f"  TEST  acc {acc:.4f} | f1_weighted {f1w:.4f} | f1_macro {f1m:.4f}")
    print(classification_report(yt, test_pred, target_names=event_names, zero_division=0))

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    torch.save(best_state, MODEL_PATH)

    cm = confusion_matrix(yt, test_pred, labels=list(range(6)))
    import pandas as pd
    pd.DataFrame(cm, index=event_names, columns=event_names).to_csv(CM_PATH)

    results = {
        "model": "LSTM",
        "task": "next_activity",
        "max_len": L.DEFAULT_MAX_LEN,
        "epochs_run": epoch,
        "best_val_f1_macro": round(best_f1, 4),
        "test": {"accuracy": round(acc, 4),
                 "f1_weighted": round(f1w, 4),
                 "f1_macro": round(f1m, 4)},
        "per_class": classification_report(yt, test_pred, target_names=event_names,
                                           output_dict=True, zero_division=0),
        "hyperparams": {"hidden": cfg.LSTM_HIDDEN_DIM, "layers": cfg.LSTM_NUM_LAYERS,
                        "dropout": cfg.LSTM_DROPOUT, "lr": LR, "batch": BATCH},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved results -> {RESULTS_PATH}")
    print(f"Total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
