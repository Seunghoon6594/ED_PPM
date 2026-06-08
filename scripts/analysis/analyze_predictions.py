"""
analyze_predictions.py
======================
Post-hoc analyses on the trained LightGBM and LSTM models (no retraining):

  A1  Earliness     - metric vs prefix length k (how early are predictions useful)
  A2  Acuity        - performance for high-acuity (1-2) vs low-acuity (3-5) visits
  B5  Bootstrap CI  - 95% confidence intervals on the headline test metrics

Reuses the saved models:
  models/lgbm_next_activity.txt, models/lgbm_remaining_time.txt
  models/lstm_next_activity.pt,  models/lstm_remaining_time.pt

Output: figures/earliness_next_activity.png, figures/earliness_remaining_time.png,
        figures/acuity_subgroup.png, results/extended_analysis.md (+ csvs)
Run   : python scripts/analysis/analyze_predictions.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
sys.path.insert(0, os.path.dirname(__file__))
import pipeline_config as cfg
import lstm_ppm as L

NON_FEATURE_COLS = {"case_id", "split", "prefix_length", "last_event",
                    "next_event", "remaining_time_min"}
RESULTS = cfg.RESULTS_DIR
FIGS = cfg.FIGURES_DIR


# --------------------------------------------------------------- predictions
def xgb_predictions():
    """Return dicts of test predictions + metadata for both tasks (XGBoost)."""
    out = {}
    # next-activity
    df = pd.read_parquet(cfg.PREFIX_NEXT_ACT_PATH)
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    le = LabelEncoder().fit(df["next_event"])
    test = df[df["split"] == "test"]
    clf = XGBClassifier()
    clf.load_model(os.path.join(cfg.MODELS_DIR, "xgb_next_activity.json"))
    pred = clf.predict(test[feats].fillna(0))
    out["next"] = {"y": le.transform(test["next_event"]), "pred": pred,
                   "k": test["prefix_length"].to_numpy(), "acuity": test["acuity"].to_numpy()}
    # remaining-time
    dfr = pd.read_parquet(cfg.PREFIX_REMAINING_PATH)
    featsr = [c for c in dfr.columns if c not in NON_FEATURE_COLS]
    testr = dfr[dfr["split"] == "test"]
    reg = XGBRegressor()
    reg.load_model(os.path.join(cfg.MODELS_DIR, "xgb_remaining_time.json"))
    pred_h = reg.predict(testr[featsr].fillna(0))
    out["rem"] = {"y": testr["remaining_time_min"].to_numpy() / 60.0, "pred": pred_h,
                  "k": testr["prefix_length"].to_numpy(), "acuity": testr["acuity"].to_numpy()}
    return out


def lstm_predict(model, data, idx, delta_n, static_n, regression=False):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), 8192):
            b = idx[s:s + 8192]
            ids_b = torch.from_numpy(data["ids"][b]).long()
            d_b = torch.from_numpy(delta_n[b])
            st_b = torch.from_numpy(static_n[b])
            l_b = torch.from_numpy(data["lengths"][b])
            o = model(ids_b, d_b, l_b, st_b)
            outs.append(o.squeeze(-1).numpy() if regression else o.argmax(1).numpy())
    return np.concatenate(outs)


def lstm_predictions():
    data = L.build_tensors(os.path.join(cfg.DATA_DIR, "lstm_sequences.npz"))
    delta_n, static_n = L.normalise(data, data["split"] == 0)
    n_static = static_n.shape[1]
    te = np.where(data["split"] == 2)[0]
    acuity = data["static"][:, 0]  # raw acuity is first static column
    out = {}
    # next-activity
    m = L.LSTMModel(emb_dim=32, hidden=cfg.LSTM_HIDDEN_DIM, layers=cfg.LSTM_NUM_LAYERS,
                    dropout=cfg.LSTM_DROPOUT, n_static=n_static, out_dim=6)
    m.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, "lstm_next_activity.pt")))
    out["next"] = {"y": data["y_next"][te], "pred": lstm_predict(m, data, te, delta_n, static_n),
                   "k": data["prefix_len"][te], "acuity": acuity[te]}
    # remaining-time
    mr = L.LSTMModel(emb_dim=32, hidden=cfg.LSTM_HIDDEN_DIM, layers=cfg.LSTM_NUM_LAYERS,
                     dropout=cfg.LSTM_DROPOUT, n_static=n_static, out_dim=1)
    mr.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, "lstm_remaining_time.pt")))
    out["rem"] = {"y": data["y_rem_h"][te], "pred": lstm_predict(mr, data, te, delta_n, static_n, regression=True),
                  "k": data["prefix_len"][te], "acuity": acuity[te]}
    return out


# --------------------------------------------------------------- analyses
K_BUCKETS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 200)]
K_LABELS = ["2", "3", "4", "5", "6-7", "8-10", "11+"]


def bucket_mask(k, lo, hi):
    return (k >= lo) & (k <= hi)


def earliness_next(models):
    rows = []
    for lo, hi in K_BUCKETS:
        row = {"bucket": f"{lo}-{hi}"}
        for name, d in models.items():
            mk = bucket_mask(d["next"]["k"], lo, hi)
            if mk.sum() == 0:
                row[f"{name}_acc"] = np.nan; row[f"{name}_f1m"] = np.nan; continue
            row[f"{name}_acc"] = accuracy_score(d["next"]["y"][mk], d["next"]["pred"][mk])
            row[f"{name}_f1m"] = f1_score(d["next"]["y"][mk], d["next"]["pred"][mk], average="macro")
            row["n"] = int(mk.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def earliness_rem(models):
    rows = []
    for lo, hi in K_BUCKETS:
        row = {"bucket": f"{lo}-{hi}"}
        for name, d in models.items():
            mk = bucket_mask(d["rem"]["k"], lo, hi)
            row[f"{name}_mae"] = (mean_absolute_error(d["rem"]["y"][mk], d["rem"]["pred"][mk])
                                  if mk.sum() else np.nan)
            row["n"] = int(mk.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def acuity_table(models):
    rows = []
    groups = {"high (1-2)": lambda a: a <= 2, "low (3-5)": lambda a: a >= 3}
    for gname, fn in groups.items():
        for name, d in models.items():
            an = d["next"]["acuity"]; ar = d["rem"]["acuity"]
            mn = fn(an) & ~np.isnan(an); mr = fn(ar) & ~np.isnan(ar)
            rows.append({
                "group": gname, "model": name, "n_next": int(mn.sum()),
                "acc": accuracy_score(d["next"]["y"][mn], d["next"]["pred"][mn]),
                "f1_macro": f1_score(d["next"]["y"][mn], d["next"]["pred"][mn], average="macro"),
                "mae_h": mean_absolute_error(d["rem"]["y"][mr], d["rem"]["pred"][mr]),
            })
    return pd.DataFrame(rows)


def bootstrap_ci(y, pred, fn, B=400, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(B):
        s = rng.integers(0, n, n)
        stats.append(fn(y[s], pred[s]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(fn(y, pred)), float(lo), float(hi)


def main():
    print("Computing XGBoost predictions...")
    xg = xgb_predictions()
    print("Computing LSTM predictions...")
    ls = lstm_predictions()
    models = {"XGBoost": xg, "LSTM": ls}

    # ---- A1 earliness ----
    en = earliness_next(models)
    er = earliness_rem(models)
    en.to_csv(os.path.join(RESULTS, "earliness_next_activity.csv"), index=False)
    er.to_csv(os.path.join(RESULTS, "earliness_remaining_time.csv"), index=False)

    x = np.arange(len(K_LABELS))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(x, en["XGBoost_f1m"], "o-", label="XGBoost")
    a1.plot(x, en["LSTM_f1m"], "s-", label="LSTM")
    a1.set_xticks(x); a1.set_xticklabels(K_LABELS)
    a1.set_xlabel("Prefix length k"); a1.set_ylabel("F1-macro")
    a1.set_title("Next-activity F1-macro vs prefix length"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(x, en["XGBoost_acc"], "o-", label="XGBoost")
    a2.plot(x, en["LSTM_acc"], "s-", label="LSTM")
    a2.set_xticks(x); a2.set_xticklabels(K_LABELS)
    a2.set_xlabel("Prefix length k"); a2.set_ylabel("Accuracy")
    a2.set_title("Next-activity Accuracy vs prefix length"); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "earliness_next_activity.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(x, er["XGBoost_mae"], "o-", label="XGBoost")
    plt.plot(x, er["LSTM_mae"], "s-", label="LSTM")
    plt.xticks(x, K_LABELS); plt.xlabel("Prefix length k"); plt.ylabel("MAE (hours)")
    plt.title("Remaining-time MAE vs prefix length"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "earliness_remaining_time.png"), dpi=120)
    plt.close()

    # ---- A2 acuity ----
    at = acuity_table(models)
    at.to_csv(os.path.join(RESULTS, "acuity_subgroup.csv"), index=False)

    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 5))
    grp = ["high (1-2)", "low (3-5)"]
    w = 0.35; xi = np.arange(2)
    for j, name in enumerate(["XGBoost", "LSTM"]):
        f1 = [at[(at.group == g) & (at.model == name)]["f1_macro"].values[0] for g in grp]
        mae = [at[(at.group == g) & (at.model == name)]["mae_h"].values[0] for g in grp]
        b1.bar(xi + (j - .5) * w, f1, w, label=name)
        b2.bar(xi + (j - .5) * w, mae, w, label=name)
    b1.set_xticks(xi); b1.set_xticklabels(grp); b1.set_title("Next-activity F1-macro by acuity"); b1.legend()
    b2.set_xticks(xi); b2.set_xticklabels(grp); b2.set_title("Remaining-time MAE by acuity"); b2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "acuity_subgroup.png"), dpi=120)
    plt.close()

    # ---- B5 bootstrap CI ----
    ci = {}
    for name, d in models.items():
        acc, alo, ahi = bootstrap_ci(d["next"]["y"], d["next"]["pred"], accuracy_score)
        f1, flo, fhi = bootstrap_ci(d["next"]["y"], d["next"]["pred"],
                                    lambda a, b: f1_score(a, b, average="macro"))
        mae, mlo, mhi = bootstrap_ci(d["rem"]["y"], d["rem"]["pred"], mean_absolute_error)
        ci[name] = {"accuracy": [round(acc, 4), round(alo, 4), round(ahi, 4)],
                    "f1_macro": [round(f1, 4), round(flo, 4), round(fhi, 4)],
                    "mae_hours": [round(mae, 4), round(mlo, 4), round(mhi, 4)]}
    json.dump(ci, open(os.path.join(RESULTS, "bootstrap_ci.json"), "w"), indent=2)

    # ---- markdown summary ----
    md = ["# Extended analysis", "",
          "## A1. Earliness - next-activity (Accuracy & F1-macro by prefix length k)", "",
          "```", en.round(4).to_string(index=False), "```", "",
          "## A1. Earliness - remaining-time (MAE by k)", "",
          "```", er.round(4).to_string(index=False), "```", "",
          "## A2. Acuity subgroup", "",
          "```", at.round(4).to_string(index=False), "```", "",
          "## B5. Bootstrap 95% CI (test, [point, lo, hi])", "",
          "```json", json.dumps(ci, indent=2), "```"]
    open(os.path.join(RESULTS, "extended_analysis.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:1]))
    print("Saved figures + results/extended_analysis.md")


if __name__ == "__main__":
    main()
