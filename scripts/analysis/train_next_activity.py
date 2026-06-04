"""
train_next_activity.py
======================
Task 1: Next activity prediction

Baseline 모델:
  1. Most-frequent baseline (항상 가장 빈번한 이벤트 예측)
  2. Last-event baseline (마지막 이벤트 기반 빈도 조건부 예측)
  3. LightGBM (structured features)

평가: Accuracy, Macro F1, Weighted F1, Confusion Matrix
"""

import sys
import os
import logging
import time
import json

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

os.makedirs(cfg.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.LOGS_DIR, "train_next_activity.log"),
                            mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# feature에서 제외할 컬럼
NON_FEATURE_COLS = {"case_id", "split", "prefix_length", "last_event",
                    "next_event", "remaining_time_min"}


# ==============================================================================
# 데이터 로딩
# ==============================================================================

def load_data():
    log.info("Loading prefix dataset (next activity)...")
    df = pd.read_parquet(cfg.PREFIX_NEXT_ACT_PATH)
    log.info(f"  Total: {len(df):,} records")

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]
    log.info(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    return train, val, test


def prepare_xy(df: pd.DataFrame, label_encoder: LabelEncoder = None):
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].fillna(0)
    y_raw = df["next_event"].values

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)

    return X, y, label_encoder


# ==============================================================================
# Baseline 1: Most-frequent
# ==============================================================================

def baseline_most_frequent(train, val, test):
    log.info("Baseline 1: Most-frequent event...")
    most_freq = train["next_event"].value_counts().idxmax()
    log.info(f"  Most frequent next event: {most_freq}")

    results = {}
    for split_name, df in [("val", val), ("test", test)]:
        y_true = df["next_event"].values
        y_pred = np.array([most_freq] * len(df))
        acc  = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results[split_name] = {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}
        log.info(f"  [{split_name}] acc={acc:.4f} | f1_w={f1_w:.4f} | f1_m={f1_m:.4f}")

    return results


# ==============================================================================
# Baseline 2: Last-event conditioned
# ==============================================================================

def baseline_last_event(train, val, test):
    log.info("Baseline 2: Last-event conditioned...")
    # 각 last_event별 가장 빈번한 next_event
    transition = (
        train.groupby("last_event")["next_event"]
        .apply(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    global_most_freq = train["next_event"].value_counts().idxmax()

    results = {}
    for split_name, df in [("val", val), ("test", test)]:
        y_true = df["next_event"].values
        y_pred = df["last_event"].map(transition).fillna(global_most_freq).values
        acc  = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results[split_name] = {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}
        log.info(f"  [{split_name}] acc={acc:.4f} | f1_w={f1_w:.4f} | f1_m={f1_m:.4f}")

    return results, transition


# ==============================================================================
# Baseline 3: LightGBM
# ==============================================================================

def train_lightgbm(train, val, test):
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm not installed. Skipping LightGBM baseline.")
        return None, None

    log.info("Baseline 3: LightGBM...")

    le = None
    X_train, y_train, le = prepare_xy(train)
    X_val,   y_val,   _  = prepare_xy(val,  le)
    X_test,  y_test,  _  = prepare_xy(test, le)

    params = {**cfg.LGBM_PARAMS_CLASSIFICATION,
              "num_class": len(le.classes_),
              "objective": "multiclass",
              "metric": "multi_logloss",
              "verbose": -1}

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(cfg.MODELS_DIR, "lgbm_next_activity.txt")
    model.save_model(model_path)
    log.info(f"  Model saved: {model_path}")

    results = {}
    for split_name, X, y_true_enc in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred_proba = model.predict(X)
        y_pred_enc   = y_pred_proba.argmax(axis=1)
        y_true_str   = le.inverse_transform(y_true_enc)
        y_pred_str   = le.inverse_transform(y_pred_enc)

        acc  = accuracy_score(y_true_str, y_pred_str)
        f1_w = f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0)
        f1_m = f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
        results[split_name] = {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}
        log.info(f"  [{split_name}] acc={acc:.4f} | f1_w={f1_w:.4f} | f1_m={f1_m:.4f}")

        if split_name == "test":
            log.info(f"\nClassification Report (test):\n"
                     f"{classification_report(y_true_str, y_pred_str, zero_division=0)}")

            cm = confusion_matrix(y_true_str, y_pred_str, labels=le.classes_)
            cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
            cm_path = os.path.join(cfg.RESULTS_DIR, "confusion_matrix_next_activity.csv")
            os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
            cm_df.to_csv(cm_path)
            log.info(f"  Confusion matrix saved: {cm_path}")

            # Feature importance
            fi = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importance(importance_type="gain"),
            }).sort_values("importance", ascending=False)
            fi_path = os.path.join(cfg.RESULTS_DIR, "feature_importance_next_activity.csv")
            fi.to_csv(fi_path, index=False)
            log.info(f"  Feature importance saved: {fi_path}")
            log.info(f"  Top-10 features:\n{fi.head(10).to_string()}")

    return results, model


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("TRAIN NEXT ACTIVITY PREDICTION")
    log.info("=" * 60)

    train, val, test = load_data()

    # Label 분포 확인
    log.info("Label distribution (train):")
    log.info(f"\n{train['next_event'].value_counts().to_string()}")

    # Baselines
    res_mf   = baseline_most_frequent(train, val, test)
    res_le, transition = baseline_last_event(train, val, test)
    res_lgbm, model = train_lightgbm(train, val, test)

    # 결과 정리
    summary = {
        "most_frequent_baseline": res_mf,
        "last_event_baseline":    res_le,
    }
    if res_lgbm:
        summary["lightgbm"] = res_lgbm

    result_path = os.path.join(cfg.RESULTS_DIR, "results_next_activity.json")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results saved: {result_path}")

    log.info(f"Total time: {time.time() - t0:.1f}s")

    # 결과 요약 출력
    print("\n" + "=" * 60)
    print("NEXT ACTIVITY PREDICTION - RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Test Acc':>10} {'Test F1-W':>10} {'Test F1-M':>10}")
    print("-" * 60)
    for model_name, res in summary.items():
        test_res = res.get("test", {})
        print(f"{model_name:<30} {test_res.get('accuracy', 0):>10.4f} "
              f"{test_res.get('f1_weighted', 0):>10.4f} "
              f"{test_res.get('f1_macro', 0):>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
