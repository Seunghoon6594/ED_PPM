"""
fair_comparison.py
==================
Apples-to-apples next-activity comparison on ONE common test sample, so that the
retrieval-augmented TabPFN (our model) is compared to XGBoost, the LSTM and
vanilla TabPFN under identical evaluation conditions.

All features are built from data/lstm_sequences.npz so the test rows are
identical across models. Models:
  - XGBoost            : trained on the 50k reference pool (balanced)
  - Vanilla TabPFN     : training-free, random 256-row context from the pool
  - RA-TabPFN (ours)   : training-free, per-query 256-NN retrieved context
  - LSTM               : the full-data trained sequence model (models/lstm_next_activity.pt)

Note on fairness: the test set is identical for all. The TabPFN variants are
training-free over a 50k pool; the LSTM is trained on the full training split
(inherent to the method, stated openly).

Output: results/fair_comparison_next_activity.json, figures/fair_comparison.png
Run   : python scripts/analysis/fair_comparison.py
Env   : PPM_FAIR_TEST (1000), PPM_FAIR_POOL (50000), PPM_FAIR_CTX (256)
"""

import os
import sys
import json
import time

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg
import lstm_ppm as L
from train_processpfn import build_features, standardize

N_TEST = int(os.environ.get("PPM_FAIR_TEST", "1000"))
N_POOL = int(os.environ.get("PPM_FAIR_POOL", "50000"))
CTX = int(os.environ.get("PPM_FAIR_CTX", "256"))
N_EVENTS = 6
EVENT_NAMES = ["ED_ARRIVAL", "TRIAGE", "VITAL_SIGN_REASSESSMENT",
               "MEDICATION_RECONCILIATION", "MEDICATION_DISPENSED", "ED_END"]


def metrics(y, p):
    return {"accuracy": round(accuracy_score(y, p), 4),
            "f1_weighted": round(f1_score(y, p, average="weighted"), 4),
            "f1_macro": round(f1_score(y, p, average="macro"), 4)}


def main():
    from tabpfn import TabPFNClassifier
    t0 = time.time()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    print("Loading sequences + features...")
    data = L.build_tensors(os.path.join(cfg.DATA_DIR, "lstm_sequences.npz"))
    y_all = data["y_next"]
    tr = np.where(data["split"] == 0)[0]
    te = np.where(data["split"] == 2)[0]
    pool_idx = rng.choice(tr, size=min(N_POOL, len(tr)), replace=False)
    test_idx = []
    for c in range(N_EVENTS):
        cls = te[y_all[te] == c]
        if len(cls):
            test_idx.append(rng.choice(cls, size=min(max(1, N_TEST // N_EVENTS), len(cls)), replace=False))
    test_idx = np.concatenate(test_idx)
    yp, yt = y_all[pool_idx], y_all[test_idx]
    print(f"  pool {len(pool_idx):,} | common test {len(test_idx):,} | ctx {CTX}")

    base_pool, _ = build_features(data, pool_idx)
    base_te, _ = build_features(data, test_idx)
    Xp = standardize(base_pool, base_pool)
    Xt = standardize(base_te, base_pool)

    results = {"config": {"pool": int(len(pool_idx)), "test": int(len(test_idx)), "ctx": CTX}}

    # ---- XGBoost (trained on pool) ----
    t = time.time()
    counts = np.bincount(yp, minlength=N_EVENTS).astype(np.float64)
    cw = counts.sum() / (N_EVENTS * np.maximum(counts, 1))
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                        tree_method="hist", n_jobs=-1, random_state=cfg.RANDOM_SEED)
    le = LabelEncoder().fit(yp)  # pool may miss the ultra-rare ED_ARRIVAL class
    xgb.fit(Xp, le.transform(yp), sample_weight=cw[yp])
    results["XGBoost"] = metrics(yt, le.inverse_transform(xgb.predict(Xt)))
    print(f"  [XGBoost]       {results['XGBoost']} ({time.time()-t:.0f}s)", flush=True)

    # ---- LSTM (full-data trained model) ----
    t = time.time()
    delta_n, static_n = L.normalise(data, data["split"] == 0)
    m = L.LSTMModel(emb_dim=32, hidden=cfg.LSTM_HIDDEN_DIM, layers=cfg.LSTM_NUM_LAYERS,
                    dropout=cfg.LSTM_DROPOUT, n_static=static_n.shape[1], out_dim=6)
    m.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, "lstm_next_activity.pt")))
    m.eval()
    with torch.no_grad():
        pr = m(torch.from_numpy(data["ids"][test_idx]).long(),
               torch.from_numpy(delta_n[test_idx]),
               torch.from_numpy(data["lengths"][test_idx]),
               torch.from_numpy(static_n[test_idx])).argmax(1).numpy()
    results["LSTM"] = metrics(yt, pr)
    print(f"  [LSTM]          {results['LSTM']} ({time.time()-t:.0f}s)", flush=True)

    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, n_estimators=1)

    # ---- Vanilla TabPFN (random context) ----
    t = time.time()
    ctx = rng.choice(len(Xp), size=min(CTX, len(Xp)), replace=False)
    clf.fit(Xp[ctx], yp[ctx])
    pr = np.concatenate([clf.predict(Xt[s:s + 4096]) for s in range(0, len(Xt), 4096)])
    results["Vanilla_TabPFN"] = metrics(yt, pr)
    print(f"  [Vanilla TabPFN] {results['Vanilla_TabPFN']} ({time.time()-t:.0f}s)", flush=True)

    # ---- RA-TabPFN (ours): per-query kNN retrieval ----
    t = time.time()
    nn = NearestNeighbors(n_neighbors=CTX).fit(Xp)
    _, nbr = nn.kneighbors(Xt)
    pr = np.empty(len(Xt), np.int64)
    for i in range(len(Xt)):
        clf.fit(Xp[nbr[i]], yp[nbr[i]])
        pr[i] = clf.predict(Xt[i:i + 1])[0]
    results["RA_TabPFN"] = metrics(yt, pr)
    print(f"  [RA-TabPFN ours] {results['RA_TabPFN']} ({time.time()-t:.0f}s)", flush=True)

    json.dump(results, open(os.path.join(cfg.RESULTS_DIR, "fair_comparison_next_activity.json"), "w"), indent=2)

    # ---- figure ----
    order = ["Vanilla_TabPFN", "XGBoost", "LSTM", "RA_TabPFN"]
    labels = ["Vanilla\nTabPFN", "XGBoost", "LSTM", "RA-TabPFN\n(ours)"]
    f1m = [results[k]["f1_macro"] for k in order]
    acc = [results[k]["accuracy"] for k in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, f1m, w, label="F1-macro", color="#E07A3B")
    ax.bar(x + w / 2, acc, w, label="Accuracy", color="#2E6DA4")
    for i in range(len(order)):
        ax.text(x[i] - w / 2, f1m[i] + .005, f"{f1m[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, acc[i] + .005, f"{acc[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Next-activity on a common test sample (n={len(test_idx)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.FIGURES_DIR, "fair_comparison.png"), dpi=120)
    print(f"Saved results + figure | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
