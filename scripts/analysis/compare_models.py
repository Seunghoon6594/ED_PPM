"""
compare_models.py
=================
Aggregate every model's test-set results for both tasks into one comparison
table (markdown) and one figure. Pulls from:
  results/results_next_activity.json       (baselines + LightGBM, Task 1)
  results/results_lstm_next_activity.json  (LSTM, Task 1)
  results/results_remaining_time.json      (baselines + LightGBM, Task 2)
  results/results_lstm_remaining_time.json (LSTM, Task 2)

Output: results/model_comparison.md, figures/model_comparison.png
Run   : python scripts/analysis/compare_models.py
"""

import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    r1 = load(os.path.join(cfg.RESULTS_DIR, "results_next_activity.json")) or {}
    r1_xgb = load(os.path.join(cfg.RESULTS_DIR, "results_xgboost_next_activity.json"))
    r1_lstm = load(os.path.join(cfg.RESULTS_DIR, "results_lstm_next_activity.json"))
    r1_tab = load(os.path.join(cfg.RESULTS_DIR, "results_tabpfn_next_activity.json"))
    r2 = load(os.path.join(cfg.RESULTS_DIR, "results_remaining_time.json")) or {}
    r2_xgb = load(os.path.join(cfg.RESULTS_DIR, "results_xgboost_remaining_time.json"))
    r2_lstm = load(os.path.join(cfg.RESULTS_DIR, "results_lstm_remaining_time.json"))
    r2_tab = load(os.path.join(cfg.RESULTS_DIR, "results_tabpfn_remaining_time.json"))

    # ---- Task 1 rows ----
    t1 = []
    for key, name in [("most_frequent_baseline", "Most-frequent baseline"),
                      ("last_event_baseline", "Last-event baseline")]:
        if key in r1:
            t = r1[key]["test"]
            t1.append((name, t["accuracy"], t["f1_weighted"], t["f1_macro"]))
    if r1_xgb:
        t = r1_xgb["test"]
        t1.append(("XGBoost", t["accuracy"], t["f1_weighted"], t["f1_macro"]))
    if r1_lstm:
        t = r1_lstm["test"]
        t1.append(("LSTM", t["accuracy"], t["f1_weighted"], t["f1_macro"]))
    if r1_tab:
        t = r1_tab["test"]
        lbl = f"TabPFN ({r1_tab['train_subsample']//1000}k/{r1_tab['test_n']//1000}k)*"
        t1.append((lbl, t["accuracy"], t["f1_weighted"], t["f1_macro"]))

    # ---- Task 2 rows ----
    t2 = []
    for key, name in [("global_mean_baseline", "Global-mean baseline"),
                      ("last_event_mean_baseline", "Last-event-mean baseline")]:
        if key in r2:
            t = r2[key]["test"]
            t2.append((name, t["mae_hours"], t["rmse_hours"], t["mape_pct"]))
    if r2_xgb:
        t = r2_xgb["test"]
        t2.append(("XGBoost", t["mae_hours"], t["rmse_hours"], t["mape_pct"]))
    if r2_lstm:
        t = r2_lstm["test"]
        t2.append(("LSTM", t["mae_hours"], t["rmse_hours"], t["mape_pct"]))
    if r2_tab:
        t = r2_tab["test"]
        lbl = f"TabPFN ({r2_tab['train_subsample']//1000}k/{r2_tab['test_n']//1000}k)*"
        t2.append((lbl, t["mae_hours"], t["rmse_hours"], t["mape_pct"]))

    # ---- markdown ----
    md = ["# Model Comparison (test set)", "",
          "## Task 1 - Next-activity prediction", "",
          "| Model | Accuracy | F1-weighted | F1-macro |",
          "|-------|----------|-------------|----------|"]
    for name, acc, fw, fm in t1:
        md.append(f"| {name} | {acc:.3f} | {fw:.3f} | {fm:.3f} |")
    md += ["", "## Task 2 - Remaining-time prediction", "",
           "| Model | MAE (h) | RMSE (h) | MAPE (%) |",
           "|-------|---------|----------|----------|"]
    for name, mae, rmse, mp in t2:
        md.append(f"| {name} | {mae:.3f} | {rmse:.3f} | {mp:.1f} |")
    if r1_tab or r2_tab:
        md += ["", "*TabPFN is fit on a small training subsample and evaluated on a "
               "test subsample (CPU in-context inference cost); all other models use "
               "the full training data and full test set, so TabPFN is a no-tuning "
               "reference point rather than a like-for-like row."]
    md_path = os.path.join(cfg.RESULTS_DIR, "model_comparison.md")
    open(md_path, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved -> {md_path}")

    # ---- figure ----
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if t1:
        names = [r[0].replace(" baseline", "") for r in t1]
        x = np.arange(len(names))
        w = 0.38
        acc = [r[1] for r in t1]
        f1m = [r[3] for r in t1]
        ax1.bar(x - w / 2, acc, w, label="Accuracy", color="#2E6DA4")
        ax1.bar(x + w / 2, f1m, w, label="F1-macro", color="#E07A3B")
        ax1.set_title("Task 1: Next-activity (Accuracy vs F1-macro)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=8, rotation=12)
        ax1.legend()
        for i in range(len(names)):
            ax1.text(x[i] - w / 2, acc[i] + 0.005, f"{acc[i]:.2f}", ha="center", fontsize=8)
            ax1.text(x[i] + w / 2, f1m[i] + 0.005, f"{f1m[i]:.2f}", ha="center", fontsize=8)
    if t2:
        names = [r[0] for r in t2]
        ax2.bar(names, [r[1] for r in t2], color="#128C7D")
        ax2.set_title("Task 2: Remaining-time MAE (lower is better)")
        ax2.set_ylabel("MAE (hours)")
        ax2.tick_params(axis="x", labelsize=8, rotation=15)
        for i, r in enumerate(t2):
            ax2.text(i, r[1] + 0.03, f"{r[1]:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(cfg.FIGURES_DIR, "model_comparison.png")
    plt.savefig(fig_path, dpi=120)
    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
