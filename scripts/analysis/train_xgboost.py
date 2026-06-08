"""
train_xgboost.py
================
XGBoost baseline for both PPM tasks - a second gradient-boosted-tree model to
check that the tabular result is robust across GBDT implementations (vs LightGBM).

Same 44 structured features, same case-level split, and matched settings
(500 trees, lr 0.05, depth ~6, balanced class weights, early stopping on val).

Input : data/prefix_dataset_*.parquet
Output: results/results_xgboost_next_activity.json
        results/results_xgboost_remaining_time.json
Run   : python scripts/analysis/train_xgboost.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error)
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

NON_FEATURE_COLS = {"case_id", "split", "prefix_length", "last_event",
                    "next_event", "remaining_time_min"}
COMMON = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
              tree_method="hist", n_jobs=-1, random_state=cfg.RANDOM_SEED,
              early_stopping_rounds=50)


def mape(y_true, y_pred, eps=1.0):
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def run_next_activity():
    print("\n=== XGBoost: Next-activity ===")
    df = pd.read_parquet(cfg.PREFIX_NEXT_ACT_PATH)
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    le = LabelEncoder().fit(df["next_event"])
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ytr = le.transform(tr["next_event"]); yva = le.transform(va["next_event"]); yte = le.transform(te["next_event"])

    # balanced sample weights (mirrors LightGBM class_weight='balanced')
    counts = np.bincount(ytr, minlength=len(le.classes_)).astype(np.float64)
    cw = counts.sum() / (len(le.classes_) * np.maximum(counts, 1))
    w = cw[ytr]

    t = time.time()
    clf = XGBClassifier(objective="multi:softprob", num_class=len(le.classes_),
                        eval_metric="mlogloss", **COMMON)
    clf.fit(tr[feats].fillna(0), ytr, sample_weight=w,
            eval_set=[(va[feats].fillna(0), yva)], verbose=False)
    clf.save_model(os.path.join(cfg.MODELS_DIR, "xgb_next_activity.json"))
    pred = clf.predict(te[feats].fillna(0))
    print(f"  trained in {time.time()-t:.0f}s")
    acc = accuracy_score(yte, pred)
    f1w = f1_score(yte, pred, average="weighted")
    f1m = f1_score(yte, pred, average="macro")
    print(f"  TEST acc {acc:.4f} | f1_weighted {f1w:.4f} | f1_macro {f1m:.4f}")
    json.dump({"model": "XGBoost", "task": "next_activity",
               "test": {"accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
                        "f1_macro": round(f1m, 4)}},
              open(os.path.join(cfg.RESULTS_DIR, "results_xgboost_next_activity.json"), "w"), indent=2)


def run_remaining_time():
    print("\n=== XGBoost: Remaining-time ===")
    df = pd.read_parquet(cfg.PREFIX_REMAINING_PATH)
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ytr = tr["remaining_time_min"].values / 60.0
    yva = va["remaining_time_min"].values / 60.0
    yte = te["remaining_time_min"].values / 60.0

    t = time.time()
    reg = XGBRegressor(objective="reg:absoluteerror", eval_metric="mae", **COMMON)
    reg.fit(tr[feats].fillna(0), ytr, eval_set=[(va[feats].fillna(0), yva)], verbose=False)
    reg.save_model(os.path.join(cfg.MODELS_DIR, "xgb_remaining_time.json"))
    pred = reg.predict(te[feats].fillna(0))
    print(f"  trained in {time.time()-t:.0f}s")
    mae = float(mean_absolute_error(yte, pred))
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    mp = mape(yte, pred)
    print(f"  TEST MAE {mae:.3f}h | RMSE {rmse:.3f}h | MAPE {mp:.1f}%")
    json.dump({"model": "XGBoost", "task": "remaining_time",
               "test": {"mae_hours": round(mae, 4), "rmse_hours": round(rmse, 4),
                        "mape_pct": round(mp, 4)}},
              open(os.path.join(cfg.RESULTS_DIR, "results_xgboost_remaining_time.json"), "w"), indent=2)


if __name__ == "__main__":
    t0 = time.time()
    run_next_activity()
    run_remaining_time()
    print(f"\nTotal {time.time()-t0:.0f}s")
