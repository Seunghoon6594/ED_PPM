"""
make_process_map.py
===================
Directly-follows process map of the abstracted event log (a standard
process-mining artefact). Computes how often each activity is directly followed
by each other activity, and renders the row-normalised transition matrix as an
annotated heatmap.

Input : data/event_log_master.parquet
Output: results/directly_follows_matrix.csv, figures/process_map.png
Run   : python scripts/analysis/make_process_map.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

ORDER = ["ED_ARRIVAL", "TRIAGE", "VITAL_SIGN_REASSESSMENT",
         "MEDICATION_RECONCILIATION", "MEDICATION_DISPENSED", "ED_END"]
SHORT = ["ARRIVAL", "TRIAGE", "VITAL", "MED_REC", "MED_DISP", "END"]


def main():
    el = pd.read_parquet(cfg.EVENT_LOG_PATH, columns=["case_id", "event_name", "event_order"])
    el = el.sort_values(["case_id", "event_order"], kind="stable")
    el["next_event"] = el.groupby("case_id", sort=False)["event_name"].shift(-1)
    df = el.dropna(subset=["next_event"])

    counts = (df.groupby(["event_name", "next_event"]).size()
              .unstack(fill_value=0).reindex(index=ORDER, columns=ORDER, fill_value=0))
    counts.to_csv(os.path.join(cfg.RESULTS_DIR, "directly_follows_matrix.csv"))

    # row-normalised transition probabilities
    row_sums = counts.sum(axis=1).replace(0, 1)
    prob = counts.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(prob.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT, rotation=30, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(SHORT)
    ax.set_xlabel("Next activity"); ax.set_ylabel("Current activity")
    ax.set_title("Directly-follows transition probabilities\n(row-normalised; ED event log)")
    for i in range(6):
        for j in range(6):
            p = prob.values[i, j]
            if p > 0:
                ax.text(j, i, f"{p:.2f}", ha="center", va="center",
                        color="white" if p > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(next | current)")
    plt.tight_layout()
    out = os.path.join(cfg.FIGURES_DIR, "process_map.png")
    plt.savefig(out, dpi=120)
    print("Saved:", out)
    print("\nTop transitions (P >= 0.15):")
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if prob.values[i, j] >= 0.15:
                print(f"  {SHORT[i]:9s} -> {SHORT[j]:9s}  {prob.values[i, j]:.2f}  (n={int(counts.values[i, j]):,})")


if __name__ == "__main__":
    main()
