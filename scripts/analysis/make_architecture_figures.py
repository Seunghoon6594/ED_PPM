"""
make_architecture_figures.py
============================
Schematic figures for the paper (no data needed):
  1. figures/pipeline_overview.png   - end-to-end methodology
  2. figures/ra_tabpfn_schematic.png - the RA-TabPFN model

Run: python scripts/analysis/make_architecture_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

NAVY = "#1B3255"
BLUE = "#2E6DA4"
TEAL = "#128C7D"
ORANGE = "#E07A3B"
LIGHT = "#EFF3F8"
GRAY = "#4A4A4A"


def box(ax, cx, cy, w, h, lines, fc=LIGHT, ec=BLUE, title=None, title_fc=BLUE,
        fs=10, tfs=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.01,rounding_size=0.04",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    y = cy
    if title:
        ax.text(cx, cy + h / 2 - 0.16, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=title_fc, zorder=3)
        y = cy - 0.12
    ax.text(cx, y, "\n".join(lines) if isinstance(lines, list) else lines,
            ha="center", va="center", fontsize=fs, color=GRAY, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=lw, color=color, zorder=1))


# ---------------------------------------------------------------- Figure 1
def pipeline_overview():
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 3.6); ax.axis("off")
    cy = 1.9
    box(ax, 1.5, cy, 2.4, 2.2,
        ["edstays", "triage", "vitalsign", "medrecon", "pyxis",
         "(diagnosis: excluded)"],
        ec=GRAY, title="MIMIC-IV-ED tables", title_fc=NAVY, fs=9)
    box(ax, 4.6, cy, 2.6, 2.2,
        ["6 activities", "same-timestamp", "aggregation", "LOS filter,",
         "leakage controls"],
        ec=TEAL, title="Event-log", title_fc=TEAL, fs=9)
    box(ax, 7.7, cy, 2.5, 2.2,
        ["length-k prefixes", "case-level split", "70 / 15 / 15",
         "structured + seq"],
        ec=BLUE, title="Prefix dataset", title_fc=BLUE, fs=9)
    box(ax, 10.7, cy, 2.5, 2.2,
        ["rules, XGBoost,", "LSTM, TabPFN,", "RA-TabPFN (ours)"],
        ec=ORANGE, title="Models", title_fc=ORANGE, fs=9.5)
    box(ax, 13.0, cy, 1.7, 2.2,
        ["next-", "activity", "+", "remaining", "time"],
        ec=NAVY, title="Tasks", title_fc=NAVY, fs=9)
    for x1, x2 in [(2.7, 3.3), (5.9, 6.45), (8.95, 9.45), (11.95, 12.15)]:
        arrow(ax, x1, cy, x2, cy)
    ax.set_title("MIMIC-IV-ED Predictive Process Monitoring pipeline",
                 fontsize=13, fontweight="bold", color=NAVY)
    plt.tight_layout()
    p = os.path.join(cfg.FIGURES_DIR, "pipeline_overview.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", p)


# ---------------------------------------------------------------- Figure 2
def ra_tabpfn_schematic():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")

    box(ax, 1.7, 3.4, 2.8, 1.3,
        ["events so far in", "the ED visit", "-> feature vector"],
        ec=BLUE, title="Query prefix", title_fc=BLUE, fs=9.5)
    box(ax, 1.7, 1.1, 2.8, 1.3,
        ["50k past prefixes", "(reference pool)"],
        ec=GRAY, title="Training pool", title_fc=NAVY, fs=9.5)
    box(ax, 5.4, 2.2, 2.4, 1.3,
        ["k nearest", "prefixes by", "feature similarity"],
        ec=TEAL, title="kNN retrieval", title_fc=TEAL, fs=9.5)
    box(ax, 8.7, 2.2, 2.6, 1.6,
        ["frozen, in-context", "inference over the", "retrieved context",
         "(NO training,", "NO tuning)"],
        ec=ORANGE, title="TabPFN", title_fc=ORANGE, fs=9.5)
    box(ax, 11.0, 2.2, 1.6, 1.1,
        ["next", "activity"],
        ec=NAVY, title="Predict", title_fc=NAVY, fs=9)

    arrow(ax, 3.1, 3.2, 8.7, 2.95)              # query -> tabpfn (context query)
    arrow(ax, 3.1, 1.2, 4.2, 1.9)               # pool -> retrieval
    arrow(ax, 6.6, 2.2, 7.4, 2.2)               # retrieval -> tabpfn
    arrow(ax, 10.0, 2.2, 10.2, 2.2)             # tabpfn -> predict
    ax.text(5.9, 3.35, "query", fontsize=8, color=GRAY, style="italic")
    ax.set_title("RA-TabPFN: retrieval-augmented, training-free tabular foundation model",
                 fontsize=12.5, fontweight="bold", color=NAVY)
    plt.tight_layout()
    p = os.path.join(cfg.FIGURES_DIR, "ra_tabpfn_schematic.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", p)


if __name__ == "__main__":
    pipeline_overview()
    ra_tabpfn_schematic()
