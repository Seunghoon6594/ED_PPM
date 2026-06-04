"""
train_remaining_time.py
=======================
Task 2: Remaining time prediction

Baseline 모델:
  1. Global mean baseline (전체 평균 remaining time)
  2. Last-event conditioned mean baseline
  3. LightGBM regression

평가: MAE, RMSE, MAPE (시간 단위로 변환)
"""

import sys
import os
import logging
import time
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

os.makedirs(cfg.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.LOGS_DIR, "train_remaining_time.log"),
                            mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

NON_FEATURE_COLS = {"case_id", "split", "prefix_length", "last_event",
                    "next_event", "remaining_time_min"}


def mape(y_true, y_pred, eps=1.0):
    """Mean Absolute Percentage Error (분모 최소값 eps로 0 방지)."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100


# ==============================================================================
# 데이터 로딩
# ==============================================================================

def load_data():
    log.info("Loading prefix dataset (remaining time)...")
    df = pd.read_parquet(cfg.PREFIX_REMAINING_PATH)
    log.info(f"  Total: {len(df):,} records")

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]
    log.info(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    log.info(f"  Remaining time (train, hours): {(train['remaining_time_min']/60).describe().to_string()}")

    return train, val, test


def prepare_xy(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].fillna(0)
    y = df["remaining_time_min"].values
    return X, y


def eval_metrics(y_true, y_pred, split_name):
    """분 단위 예측을 시간 단위로 변환하여 평가."""
    y_true_h = y_true / 60
    y_pred_h = y_pred / 60

    mae  = mean_absolute_error(y_true_h, y_pred_h)
    rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    mape_val = mape(y_true_h, y_pred_h)
    metrics = {"mae_hours": mae, "rmse_hours": rmse, "mape_pct": mape_val}
    log.info(f"  [{split_name}] MAE={mae:.3f}h | RMSE={rmse:.3f}h | MAPE={mape_val:.1f}%")
    return metrics


# ==============================================================================
# Baseline 1: Global mean
# ==============================================================================

def baseline_global_mean(train, val, test):
    log.info("Baseline 1: Global mean remaining time...")
    global_mean = train["remaining_time_min"].mean()
    log.info(f"  Global mean: {global_mean/60:.2f} hours")

    results = {}
    for split_name, df in [("val", val), ("test", test)]:
        y_true = df["remaining_time_min"].values
        y_pred = np.full(len(df), global_mean)
        results[split_name] = eval_metrics(y_true, y_pred, split_name)

    return results


# ==============================================================================
# Baseline 2: Last-event conditioned mean
# ==============================================================================

def baseline_last_event_mean(train, val, test):
    log.info("Baseline 2: Last-event conditioned mean...")
    cond_mean = train.groupby("last_event")["remaining_time_min"].mean().to_dict()
    global_mean = train["remaining_time_min"].mean()

    results = {}
    for split_name, df in [("val", val), ("test", test)]:
        y_true = df["remaining_time_min"].values
        y_pred = df["last_event"].map(cond_mean).fillna(global_mean).values
        results[split_name] = eval_metrics(y_true, y_pred, split_name)

    return results


# ==============================================================================
# Baseline 3: LightGBM regression
# ==============================================================================

def train_lightgbm(train, val, test):
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm not installed. Skipping.")
        return None

    log.info("Baseline 3: LightGBM regression...")

    X_train, y_train = prepare_xy(train)
    X_val,   y_val   = prepare_xy(val)
    X_test,  y_test  = prepare_xy(test)

    params = {**cfg.LGBM_PARAMS_REGRESSION,
              "objective": "regression",
              "metric": "mae",
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
    model_path = os.path.join(cfg.MODELS_DIR, "lgbm_remaining_time.txt")
    model.save_model(model_path)
    log.info(f"  Model saved: {model_path}")

    results = {}
    for split_name, X, y_true in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_pred = np.clip(y_pred, 0, None)   # 음수 예측 방지
        results[split_name] = eval_metrics(y_true, y_pred, split_name)

    # Feature importance
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    fi_path = os.path.join(cfg.RESULTS_DIR, "feature_importance_remaining_time.csv")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    fi.to_csv(fi_path, index=False)
    log.info(f"  Feature importance saved: {fi_path}")
    log.info(f"  Top-10 features:\n{fi.head(10).to_string()}")

    # Scatter plot: actual vs predicted
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import platform
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    y_test_h = y_test / 60
    y_pred_h = model.predict(X_test) / 60
    y_pred_h = np.clip(y_pred_h, 0, None)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test_h[:5000], y_pred_h[:5000], alpha=0.2, s=5, color="#4C72B0")
    max_val = max(y_test_h.max(), y_pred_h.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual remaining time (hours)")
    ax.set_ylabel("Predicted remaining time (hours)")
    ax.set_title("Remaining Time Prediction: Actual vs Predicted")
    ax.legend()
    plot_path = os.path.join(cfg.RESULTS_DIR, "plots", "remaining_time_scatter.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Scatter plot saved: {plot_path}")

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("TRAIN REMAINING TIME PREDICTION")
    log.info("=" * 60)

    train, val, test = load_data()

    res_gm  = baseline_global_mean(train, val, test)
    res_lem = baseline_last_event_mean(train, val, test)
    res_lgbm = train_lightgbm(train, val, test)

    summary = {
        "global_mean_baseline":     res_gm,
        "last_event_mean_baseline": res_lem,
    }
    if res_lgbm:
        summary["lightgbm"] = res_lgbm

    result_path = os.path.join(cfg.RESULTS_DIR, "results_remaining_time.json")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results saved: {result_path}")

    log.info(f"Total time: {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("REMAINING TIME PREDICTION - RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Test MAE (h)':>12} {'Test RMSE (h)':>13} {'Test MAPE (%)':>13}")
    print("-" * 70)
    for model_name, res in summary.items():
        test_res = res.get("test", {})
        print(f"{model_name:<30} {test_res.get('mae_hours', 0):>12.3f} "
              f"{test_res.get('rmse_hours', 0):>13.3f} "
              f"{test_res.get('mape_pct', 0):>13.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
