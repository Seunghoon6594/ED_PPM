"""
fair_comparison.py
==================
Apples-to-apples comparison on ONE common test sample for BOTH tasks, under a
chosen split. Lets RA-TabPFN (our model) be compared to XGBoost, the LSTM and
vanilla TabPFN under identical evaluation conditions.

All features are built from data/lstm_sequences.npz so test rows are identical
across models.

Models:
  - XGBoost         : trained on the 50k reference pool (balanced / MAE objective)
  - Vanilla TabPFN  : training-free, random context from the pool
  - RA-TabPFN (ours): training-free, per-query kNN-retrieved context
  - LSTM            : the full-data trained sequence model (RANDOM split only;
                      under a temporal split it would need retraining, left as
                      future work, so it is omitted from the temporal table)

Split:
  PPM_SPLIT=random   -> the case-level split stored in the event log (default)
  PPM_SPLIT=temporal -> cases ordered by arrival time, first 70/15/15 by time
                        (the PPM-standard chronological split, for robustness)

Output: results/fair_comparison_<split>.json, figures/fair_comparison_<split>_*.png
Run   : python scripts/analysis/fair_comparison.py
Env   : PPM_SPLIT (random), PPM_FAIR_TEST (1000), PPM_FAIR_POOL (50000), PPM_FAIR_CTX (256)
"""

import os
import sys
import json
import time

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error)
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg
import lstm_ppm as L
from train_processpfn import build_features, standardize

SPLIT = os.environ.get("PPM_SPLIT", "random")
N_TEST = int(os.environ.get("PPM_FAIR_TEST", "1000"))
N_POOL = int(os.environ.get("PPM_FAIR_POOL", "50000"))
CTX = int(os.environ.get("PPM_FAIR_CTX", "256"))
N_EVENTS = 6


def temporal_split(data):
    """Per-prefix split (0/1/2) from a chronological 70/15/15 case ordering.

    Case index i in the npz corresponds to the i-th smallest case_id (the order
    used by make_sequence_dataset). We recover each case's arrival time, rank
    cases by it, and assign the earliest 70% to train, next 15% to val, last 15%
    to test -- the standard PPM temporal split.
    """
    el = pd.read_parquet(cfg.EVENT_LOG_PATH, columns=["case_id", "case_start"])
    first = el.groupby("case_id", sort=True)["case_start"].first()  # sorted by case_id asc
    start = pd.to_datetime(first.values)
    n = data["n_cases"]
    assert len(start) == n, f"case count mismatch {len(start)} vs {n}"
    order = np.argsort(start.astype("datetime64[ns]").astype(np.int64))
    case_split = np.empty(n, np.int64)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    case_split[order[:n_tr]] = 0
    case_split[order[n_tr:n_tr + n_va]] = 1
    case_split[order[n_tr + n_va:]] = 2
    return case_split[data["case_idx"]]


def mape(y, p, eps=1.0):
    return float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), eps))) * 100)


def cls_metrics(y, p):
    return {"accuracy": round(accuracy_score(y, p), 4),
            "f1_weighted": round(f1_score(y, p, average="weighted"), 4),
            "f1_macro": round(f1_score(y, p, average="macro"), 4)}


def reg_metrics(y, p):
    return {"mae_hours": round(float(mean_absolute_error(y, p)), 4),
            "rmse_hours": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            "mape_pct": round(mape(y, p), 4)}


def main():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    t0 = time.time()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    print(f"Loading sequences + features... (split={SPLIT})")
    data = L.build_tensors(os.path.join(cfg.DATA_DIR, "lstm_sequences.npz"))
    split = temporal_split(data) if SPLIT == "temporal" else data["split"]
    y_cls = data["y_next"]
    y_reg = data["y_rem_h"]
    tr = np.where(split == 0)[0]
    te = np.where(split == 2)[0]
    pool_idx = rng.choice(tr, size=min(N_POOL, len(tr)), replace=False)
    test_idx = []
    for c in range(N_EVENTS):
        cls = te[y_cls[te] == c]
        if len(cls):
            test_idx.append(rng.choice(cls, size=min(max(1, N_TEST // N_EVENTS), len(cls)), replace=False))
    test_idx = np.concatenate(test_idx)
    print(f"  pool {len(pool_idx):,} | common test {len(test_idx):,} | ctx {CTX}")

    base_pool, _ = build_features(data, pool_idx)
    base_te, _ = build_features(data, test_idx)
    Xp = standardize(base_pool, base_pool)
    Xt = standardize(base_te, base_pool)
    ycp, yct = y_cls[pool_idx], y_cls[test_idx]
    yrp, yrt = y_reg[pool_idx], y_reg[test_idx]

    res = {"config": {"split": SPLIT, "pool": int(len(pool_idx)),
                      "test": int(len(test_idx)), "ctx": CTX},
           "next_activity": {}, "remaining_time": {}}

    # ---------- XGBoost ----------
    t = time.time()
    counts = np.bincount(ycp, minlength=N_EVENTS).astype(np.float64)
    cw = counts.sum() / (N_EVENTS * np.maximum(counts, 1))
    le = LabelEncoder().fit(ycp)
    xc = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                       tree_method="hist", n_jobs=-1, random_state=cfg.RANDOM_SEED)
    xc.fit(Xp, le.transform(ycp), sample_weight=cw[ycp])
    res["next_activity"]["XGBoost"] = cls_metrics(yct, le.inverse_transform(xc.predict(Xt)))
    xr = XGBRegressor(objective="reg:absoluteerror", n_estimators=400, learning_rate=0.05,
                      max_depth=6, tree_method="hist", n_jobs=-1, random_state=cfg.RANDOM_SEED)
    xr.fit(Xp, yrp)
    res["remaining_time"]["XGBoost"] = reg_metrics(yrt, xr.predict(Xt))
    print(f"  [XGBoost] {res['next_activity']['XGBoost']} | {res['remaining_time']['XGBoost']} ({time.time()-t:.0f}s)", flush=True)

    # ---------- LSTM (random split only; trained on the random split) ----------
    if SPLIT == "random":
        t = time.time()
        delta_n, static_n = L.normalise(data, data["split"] == 0)
        ns = static_n.shape[1]
        def lstm_pred(model_file, out_dim):
            m = L.LSTMModel(emb_dim=32, hidden=cfg.LSTM_HIDDEN_DIM, layers=cfg.LSTM_NUM_LAYERS,
                            dropout=cfg.LSTM_DROPOUT, n_static=ns, out_dim=out_dim)
            m.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, model_file)))
            m.eval()
            with torch.no_grad():
                o = m(torch.from_numpy(data["ids"][test_idx]).long(),
                      torch.from_numpy(delta_n[test_idx]),
                      torch.from_numpy(data["lengths"][test_idx]),
                      torch.from_numpy(static_n[test_idx]))
            return o
        res["next_activity"]["LSTM"] = cls_metrics(yct, lstm_pred("lstm_next_activity.pt", 6).argmax(1).numpy())
        res["remaining_time"]["LSTM"] = reg_metrics(yrt, lstm_pred("lstm_remaining_time.pt", 1).squeeze(-1).numpy())
        print(f"  [LSTM] {res['next_activity']['LSTM']} | {res['remaining_time']['LSTM']} ({time.time()-t:.0f}s)", flush=True)

    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, n_estimators=1)
    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True, n_estimators=1)

    # ---------- Vanilla TabPFN (random context) ----------
    t = time.time()
    ctx = rng.choice(len(Xp), size=min(CTX, len(Xp)), replace=False)
    clf.fit(Xp[ctx], ycp[ctx])
    pc = np.concatenate([clf.predict(Xt[s:s + 4096]) for s in range(0, len(Xt), 4096)])
    res["next_activity"]["Vanilla_TabPFN"] = cls_metrics(yct, pc)
    reg.fit(Xp[ctx], yrp[ctx])
    pr = np.concatenate([reg.predict(Xt[s:s + 4096]) for s in range(0, len(Xt), 4096)])
    res["remaining_time"]["Vanilla_TabPFN"] = reg_metrics(yrt, pr)
    print(f"  [Vanilla TabPFN] {res['next_activity']['Vanilla_TabPFN']} | {res['remaining_time']['Vanilla_TabPFN']} ({time.time()-t:.0f}s)", flush=True)

    # ---------- RA-TabPFN (ours): per-query kNN retrieval ----------
    t = time.time()
    nn = NearestNeighbors(n_neighbors=CTX).fit(Xp)
    _, nbr = nn.kneighbors(Xt)
    pc = np.empty(len(Xt), np.int64); pr = np.empty(len(Xt), np.float64)
    for i in range(len(Xt)):
        c = nbr[i]
        clf.fit(Xp[c], ycp[c]); pc[i] = clf.predict(Xt[i:i + 1])[0]
        reg.fit(Xp[c], yrp[c]); pr[i] = reg.predict(Xt[i:i + 1])[0]
    res["next_activity"]["RA_TabPFN"] = cls_metrics(yct, pc)
    res["remaining_time"]["RA_TabPFN"] = reg_metrics(yrt, pr)
    print(f"  [RA-TabPFN ours] {res['next_activity']['RA_TabPFN']} | {res['remaining_time']['RA_TabPFN']} ({time.time()-t:.0f}s)", flush=True)

    out = os.path.join(cfg.RESULTS_DIR, f"fair_comparison_{SPLIT}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"Saved -> {out} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
