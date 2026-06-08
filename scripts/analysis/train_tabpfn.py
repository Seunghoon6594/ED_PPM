"""
train_tabpfn.py
===============
TabPFN baseline for both PPM tasks on the structured prefix features.

TabPFN is a prior-data-fitted transformer designed for SMALL tabular datasets:
it performs in-context learning over the training set at inference, so it cannot
ingest our 2.25M training prefixes. Following standard practice we fit it on a
random training subsample (default 10,000 rows, its recommended regime) using the
SAME 44 structured features as LightGBM, and evaluate on the SAME test split.
This is a no-tuning reference point: how far does an off-the-shelf foundation
model for tables get from a tiny sample, versus models trained on the full data.

Input : data/prefix_dataset_*.parquet
Output: results/results_tabpfn_next_activity.json
        results/results_tabpfn_remaining_time.json
Run   : python scripts/analysis/train_tabpfn.py
Env   : PPM_TABPFN_TRAIN (default 10000), PPM_TABPFN_TEST (default 100000)
"""

import os
import sys
import json
import time

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

NON_FEATURE_COLS = {"case_id", "split", "prefix_length", "last_event",
                    "next_event", "remaining_time_min"}
# TabPFN is a small-data, in-context learner; on CPU its inference is costly, so
# we fit on a small training subsample and evaluate on a large test subsample.
TRAIN_SUB = int(os.environ.get("PPM_TABPFN_TRAIN", "3000"))
TEST_SUB = int(os.environ.get("PPM_TABPFN_TEST", "20000"))  # 0 = full test
EVENT_NAMES = ["ED_ARRIVAL", "TRIAGE", "VITAL_SIGN_REASSESSMENT",
               "MEDICATION_RECONCILIATION", "MEDICATION_DISPENSED", "ED_END"]


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def chunked_predict(predict_fn, X, chunk=4096):
    out = []
    for s in range(0, len(X), chunk):
        out.append(predict_fn(X.iloc[s:s + chunk]))
    return np.concatenate(out)


def mape(y_true, y_pred, eps=1.0):
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def run_next_activity():
    from tabpfn import TabPFNClassifier
    print("\n=== TabPFN: Next-activity ===")
    df = pd.read_parquet(cfg.PREFIX_NEXT_ACT_PATH)
    feats = feature_cols(df)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    le = LabelEncoder().fit(df["next_event"])
    sub = train.sample(n=min(TRAIN_SUB, len(train)), random_state=cfg.RANDOM_SEED)
    X_tr = sub[feats].fillna(0)
    y_tr = le.transform(sub["next_event"])

    if TEST_SUB and TEST_SUB < len(test):
        test = test.sample(n=TEST_SUB, random_state=cfg.RANDOM_SEED)
    X_te = test[feats].fillna(0)
    y_te = le.transform(test["next_event"])
    print(f"  train subsample {len(X_tr):,} | test {len(X_te):,} | classes {len(np.unique(y_tr))}")

    t = time.time()
    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    clf.fit(X_tr, y_tr)
    pred = chunked_predict(clf.predict, X_te)
    print(f"  fit+predict {time.time()-t:.0f}s")

    acc = accuracy_score(y_te, pred)
    f1w = f1_score(y_te, pred, average="weighted")
    f1m = f1_score(y_te, pred, average="macro")
    print(f"  TEST acc {acc:.4f} | f1_weighted {f1w:.4f} | f1_macro {f1m:.4f}")

    res = {"model": "TabPFN", "task": "next_activity",
           "train_subsample": int(len(X_tr)), "test_n": int(len(X_te)),
           "test": {"accuracy": round(acc, 4),
                    "f1_weighted": round(f1w, 4),
                    "f1_macro": round(f1m, 4)}}
    path = os.path.join(cfg.RESULTS_DIR, "results_tabpfn_next_activity.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"  saved -> {path}")


def run_remaining_time():
    from tabpfn import TabPFNRegressor
    print("\n=== TabPFN: Remaining-time ===")
    df = pd.read_parquet(cfg.PREFIX_REMAINING_PATH)
    feats = feature_cols(df)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    sub = train.sample(n=min(TRAIN_SUB, len(train)), random_state=cfg.RANDOM_SEED)
    X_tr = sub[feats].fillna(0)
    y_tr = (sub["remaining_time_min"].values / 60.0).astype(np.float32)

    if TEST_SUB and TEST_SUB < len(test):
        test = test.sample(n=TEST_SUB, random_state=cfg.RANDOM_SEED)
    X_te = test[feats].fillna(0)
    y_te = (test["remaining_time_min"].values / 60.0).astype(np.float32)
    print(f"  train subsample {len(X_tr):,} | test {len(X_te):,}")

    t = time.time()
    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
    reg.fit(X_tr, y_tr)
    pred = chunked_predict(reg.predict, X_te)
    print(f"  fit+predict {time.time()-t:.0f}s")

    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mp = mape(y_te, pred)
    print(f"  TEST MAE {mae:.3f}h | RMSE {rmse:.3f}h | MAPE {mp:.1f}%")

    res = {"model": "TabPFN", "task": "remaining_time",
           "train_subsample": int(len(X_tr)), "test_n": int(len(X_te)),
           "test": {"mae_hours": round(mae, 4),
                    "rmse_hours": round(rmse, 4),
                    "mape_pct": round(mp, 4)}}
    path = os.path.join(cfg.RESULTS_DIR, "results_tabpfn_remaining_time.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"  saved -> {path}")


if __name__ == "__main__":
    t0 = time.time()
    run_next_activity()
    run_remaining_time()
    print(f"\nTotal {time.time()-t0:.0f}s")
