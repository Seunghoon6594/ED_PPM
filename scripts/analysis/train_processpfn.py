"""
train_processpfn.py
===================
ProcessPFN: a retrieval-augmented, sequence-aware adaptation of the TabPFN
tabular foundation model for predictive process monitoring (next-activity).

Motivation (from our own results): a vanilla tabular foundation model matches a
full-data GBDT but loses to the LSTM because it ignores event ORDER. We test
whether two training-free adaptations close that gap:
  (i)  sequence-aware features  (recency-decayed event histogram, last-4/5,
       bigram/trigram transition codes, event-distribution entropy) -- in the
       spirit of TabPFN-TS (reframing sequential signal as tabular features);
  (ii) retrieval context  -- per query, the in-context set is the k nearest
       training prefixes instead of a random sample -- in the spirit of
       LoCalPFN / TabDPT.

This yields a 2x2 ablation (features x context selection):
  base + random      = Vanilla TabPFN
  base + kNN          = TabPFN-kNN
  seq  + random       = Seq-TabPFN
  seq  + kNN          = ProcessPFN (ours)

All four are evaluated training-free on the SAME stratified test sample with the
SAME context size, so differences isolate each component. Local + DUA-safe
(no data leaves the machine).

Input : data/lstm_sequences.npz
Output: results/results_processpfn.json (ablation + headline ProcessPFN)
Run   : python scripts/analysis/train_processpfn.py
Env   : PPM_POOL (50000), PPM_TEST (3000), PPM_CTX (512), PPM_DECAY (0.6)
"""

import os
import sys
import json
import time

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg
import lstm_ppm as L

N_POOL = int(os.environ.get("PPM_POOL", "50000"))
N_TEST = int(os.environ.get("PPM_TEST", "1000"))
CTX = int(os.environ.get("PPM_CTX", "256"))
DECAY = float(os.environ.get("PPM_DECAY", "0.6"))
# TabPFN internal ensemble size; 1 keeps per-query retrieval inference tractable on CPU
N_EST = int(os.environ.get("PPM_NEST", "1"))
N_EVENTS = 6
EVENT_NAMES = ["ED_ARRIVAL", "TRIAGE", "VITAL_SIGN_REASSESSMENT",
               "MEDICATION_RECONCILIATION", "MEDICATION_DISPENSED", "ED_END"]


def build_features(data, idx):
    """Construct base and sequence-aware feature matrices for rows `idx`.

    Returns (X_base, X_seq) where X_seq is X_base augmented with order features.
    """
    ids = data["ids"][idx]          # [n, T] left-aligned, 0 = pad
    delta = data["delta"][idx]      # [n, T] minute gaps
    lengths = data["lengths"][idx]  # [n]
    static = data["static"][idx]    # [n, S]
    n, T = ids.shape
    rows = np.arange(n)
    valid = ids > 0                 # [n, T]

    # last-j event id (1..6), 0 if absent
    def last_j(j):
        pos = lengths - j
        ok = pos >= 0
        out = np.zeros(n, dtype=np.int64)
        out[ok] = ids[rows[ok], pos[ok]]
        return out
    l1, l2, l3, l4, l5 = (last_j(j) for j in (1, 2, 3, 4, 5))

    def onehot(v):
        oh = np.zeros((n, N_EVENTS), np.float32)
        m = v > 0
        oh[rows[m], v[m] - 1] = 1.0
        return oh

    # frequency counts per event type
    freq = np.stack([(ids == e).sum(1) for e in range(1, N_EVENTS + 1)], 1).astype(np.float32)

    # duration features
    dur_sum = (delta * valid).sum(1)
    last_dur = delta[rows, np.maximum(lengths - 1, 0)]
    avg_dur = dur_sum / np.maximum(lengths, 1)
    std_dur = np.sqrt(((delta - avg_dur[:, None]) ** 2 * valid).sum(1) / np.maximum(lengths, 1))

    base = np.column_stack([
        freq, onehot(l1), onehot(l2), onehot(l3),
        dur_sum, last_dur, avg_dur, std_dur,
        lengths.astype(np.float32), static,
    ]).astype(np.float32)

    # ---- sequence-aware extras ----
    # recency-decayed event histogram: weight = DECAY^(distance from last event)
    col = np.arange(T)[None, :]
    dist = (lengths[:, None] - 1 - col)            # 0 at the last event
    w = np.where(valid, DECAY ** np.clip(dist, 0, None), 0.0).astype(np.float32)
    decay_hist = np.stack([(w * (ids == e)).sum(1) for e in range(1, N_EVENTS + 1)], 1)

    bigram = (l2 * (N_EVENTS + 1) + l1).astype(np.float32)
    trigram = (l3 * (N_EVENTS + 1) ** 2 + l2 * (N_EVENTS + 1) + l1).astype(np.float32)

    p = freq / np.maximum(freq.sum(1, keepdims=True), 1)
    entropy = -(np.where(p > 0, p * np.log(p + 1e-9), 0.0)).sum(1)

    seq_extra = np.column_stack([
        decay_hist, onehot(l4), onehot(l5), bigram, trigram, entropy.astype(np.float32),
    ]).astype(np.float32)

    return base, np.column_stack([base, seq_extra]).astype(np.float32)


def standardize(X, ref):
    # impute NaN (e.g., missing acuity/vitals) with reference (pool) column means
    imp = np.nanmean(ref, axis=0)
    imp = np.where(np.isnan(imp), 0.0, imp)
    Xf = np.where(np.isnan(X), imp, X)
    reff = np.where(np.isnan(ref), imp, ref)
    mu, sd = reff.mean(0), reff.std(0)
    sd[sd < 1e-6] = 1.0
    out = (Xf - mu) / sd
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def eval_random(clf, Xp, yp, Xt, yt, rng):
    ctx = rng.choice(len(Xp), size=min(CTX, len(Xp)), replace=False)
    clf.fit(Xp[ctx], yp[ctx])
    pred = np.concatenate([clf.predict(Xt[s:s + 4096]) for s in range(0, len(Xt), 4096)])
    return pred


def eval_knn(clf, Xp, yp, Xt, yt, nn):
    _, nbr = nn.kneighbors(Xt)        # [n_test, CTX] indices into pool
    pred = np.empty(len(Xt), np.int64)
    for i in range(len(Xt)):
        c = nbr[i]
        clf.fit(Xp[c], yp[c])
        pred[i] = clf.predict(Xt[i:i + 1])[0]
    return pred


def metrics(y, p):
    return {"accuracy": round(accuracy_score(y, p), 4),
            "f1_weighted": round(f1_score(y, p, average="weighted"), 4),
            "f1_macro": round(f1_score(y, p, average="macro"), 4)}


def main():
    from tabpfn import TabPFNClassifier
    t0 = time.time()
    np.random.seed(cfg.RANDOM_SEED)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    print("Loading sequences + building features...")
    data = L.build_tensors(os.path.join(cfg.DATA_DIR, "lstm_sequences.npz"))
    y_all = data["y_next"]
    tr = np.where(data["split"] == 0)[0]
    te = np.where(data["split"] == 2)[0]
    pool_idx = rng.choice(tr, size=min(N_POOL, len(tr)), replace=False)
    # stratified test sample
    test_idx = []
    for c in range(N_EVENTS):
        cls = te[y_all[te] == c]
        if len(cls):
            test_idx.append(rng.choice(cls, size=min(max(1, N_TEST // N_EVENTS), len(cls)), replace=False))
    test_idx = np.concatenate(test_idx)
    print(f"  pool {len(pool_idx):,} | test {len(test_idx):,} | ctx {CTX}")

    base_pool, seq_pool = build_features(data, pool_idx)
    base_te, seq_te = build_features(data, test_idx)
    yp, yt = y_all[pool_idx], y_all[test_idx]

    # standardize each feature set with pool stats
    base_pool_s = standardize(base_pool, base_pool); base_te_s = standardize(base_te, base_pool)
    seq_pool_s = standardize(seq_pool, seq_pool); seq_te_s = standardize(seq_te, seq_pool)

    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, n_estimators=N_EST)
    nn_base = NearestNeighbors(n_neighbors=CTX).fit(base_pool_s)
    nn_seq = NearestNeighbors(n_neighbors=CTX).fit(seq_pool_s)

    results = {"config": {"pool": int(len(pool_idx)), "test": int(len(test_idx)),
                          "ctx": CTX, "decay": DECAY, "n_estimators": N_EST}}

    runs = [
        ("vanilla_tabpfn", "base", "random"),
        ("tabpfn_knn", "base", "knn"),
        ("seq_tabpfn", "seq", "random"),
        ("processpfn", "seq", "knn"),
    ]
    for name, feat, ctxsel in runs:
        t = time.time()
        Xp = base_pool_s if feat == "base" else seq_pool_s
        Xt = base_te_s if feat == "base" else seq_te_s
        if ctxsel == "random":
            pred = eval_random(clf, Xp, yp, Xt, yt, rng)
        else:
            nn = nn_base if feat == "base" else nn_seq
            pred = eval_knn(clf, Xp, yp, Xt, yt, nn)
        results[name] = metrics(yt, pred)
        print(f"  [{name:16s}] {results[name]} ({time.time()-t:.0f}s)", flush=True)

    path = os.path.join(cfg.RESULTS_DIR, "results_processpfn.json")
    json.dump(results, open(path, "w"), indent=2)
    print(f"Saved -> {path} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
