"""
make_sequence_dataset.py
========================
Event log -> sequence dataset for recurrent models (LSTM).

The structured prefix datasets (make_prefix_dataset.py) flatten each prefix into
44 hand-crafted features. An LSTM instead consumes the ordered event sequence
directly, so here we export a compact, ragged representation of every case:

  - per-event:  event-id sequence and the time gap before each event
  - per-event:  remaining time at each event (used as the regression label)
  - per-case:   static context (acuity, gender, triage vitals, arrival time,
                arrival transport one-hot)
  - per-case:   train / val / test split (reused verbatim from
                prefix_dataset_next_activity.parquet so the LSTM, LightGBM and
                baselines are all evaluated on exactly the same case split)

Prefixes are NOT materialised here (that would blow up memory). Instead the
training Dataset enumerates (case, k) pairs and slices the ragged arrays on the
fly: input = events[:k], next-activity label = events[k], remaining-time label =
remaining[k-1]. This reproduces the same 3,216,660 prefixes as the structured
datasets.

Output: data/lstm_sequences.npz
Run:    python scripts/preprocessing/make_sequence_dataset.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

# Fixed event ordering. id 0 is reserved for PAD.
EVENT_ORDER = [
    "ED_ARRIVAL",
    "TRIAGE",
    "VITAL_SIGN_REASSESSMENT",
    "MEDICATION_RECONCILIATION",
    "MEDICATION_DISPENSED",
    "ED_END",
]
EVENT_TO_ID = {name: i + 1 for i, name in enumerate(EVENT_ORDER)}  # 1..6

STATIC_CONT = ["acuity", "gender", "triage_hr", "triage_sbp", "triage_o2sat",
               "arrival_hour", "arrival_dow", "arrival_is_night"]
TRANSPORT_CATS = ["AMBULANCE", "HELICOPTER", "OTHER", "UNKNOWN", "WALK IN"]
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}

OUT_PATH = os.path.join(cfg.DATA_DIR, "lstm_sequences.npz")


def main():
    t0 = time.time()
    print("Loading event log (selected columns)...")
    cols = ["case_id", "event_name", "event_order", "time_since_prev_min",
            "remaining_time_min", "case_start", "acuity", "gender",
            "triage_hr", "triage_sbp", "triage_o2sat", "arrival_transport"]
    el = pd.read_parquet(cfg.EVENT_LOG_PATH, columns=cols)
    print(f"  {len(el):,} events")

    # Reuse the exact case -> split assignment from the structured prefix dataset.
    print("Loading case -> split map from prefix dataset...")
    split_map = (pd.read_parquet(cfg.PREFIX_NEXT_ACT_PATH, columns=["case_id", "split"])
                 .drop_duplicates("case_id"))
    split_map["split_id"] = split_map["split"].map(SPLIT_TO_ID).astype(np.int8)
    print(f"  {len(split_map):,} cases")

    # Order events within each case.
    el = el.sort_values(["case_id", "event_order"], kind="stable").reset_index(drop=True)
    el["ev_id"] = el["event_name"].map(EVENT_TO_ID).astype(np.int8)
    el["gender"] = (el["gender"].astype(str) == "M").astype(np.float32)  # M=1, F=0

    # Derive arrival-time features from case_start.
    case_start = pd.to_datetime(el["case_start"])
    el["arrival_hour"] = case_start.dt.hour.astype(np.float32)
    el["arrival_dow"] = case_start.dt.dayofweek.astype(np.float32)
    el["arrival_is_night"] = ((case_start.dt.hour >= 22) | (case_start.dt.hour < 6)).astype(np.float32)

    # ---- ragged per-event arrays via CSR-style offsets ----
    counts = el.groupby("case_id", sort=False).size().values
    case_ids = el["case_id"].drop_duplicates().values
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    event_ids = el["ev_id"].to_numpy(np.int8)
    time_deltas = el["time_since_prev_min"].fillna(0.0).to_numpy(np.float32)
    remaining = el["remaining_time_min"].fillna(0.0).to_numpy(np.float32)

    # ---- per-case static features (first row of each case) ----
    first = el.groupby("case_id", sort=False).first()
    first = first.loc[case_ids]  # align order

    static_parts = []
    for c in STATIC_CONT:
        static_parts.append(first[c].to_numpy(np.float32))
    transport = first["arrival_transport"].fillna("UNKNOWN").astype(str)
    for cat in TRANSPORT_CATS:
        static_parts.append((transport == cat).to_numpy(np.float32))
    static = np.column_stack(static_parts).astype(np.float32)
    static_cols = STATIC_CONT + [f"transport_{c}" for c in TRANSPORT_CATS]

    # ---- align split to case order ----
    sp = split_map.set_index("case_id")["split_id"].reindex(case_ids)
    if sp.isna().any():
        raise ValueError(f"{int(sp.isna().sum())} cases missing a split assignment")
    split_id = sp.to_numpy(np.int8)

    # ---- build the (case_idx, k) prefix index, k = 2 .. len-1 ----
    case_idx_list, k_list = [], []
    lengths = counts
    for ci, L in enumerate(lengths):
        if L < 3:
            continue
        ks = np.arange(2, L)  # 2 .. L-1
        k_list.append(ks)
        case_idx_list.append(np.full(len(ks), ci, dtype=np.int64))
    prefix_case_idx = np.concatenate(case_idx_list)
    prefix_k = np.concatenate(k_list).astype(np.int32)
    prefix_split = split_id[prefix_case_idx]

    print(f"  prefixes generated: {len(prefix_k):,}")

    print(f"Saving -> {OUT_PATH}")
    np.savez_compressed(
        OUT_PATH,
        event_ids=event_ids,
        time_deltas=time_deltas,
        remaining=remaining,
        offsets=offsets,
        static=static,
        split_id=split_id,
        prefix_case_idx=prefix_case_idx,
        prefix_k=prefix_k,
        prefix_split=prefix_split,
        event_vocab=np.array(EVENT_ORDER),
        static_cols=np.array(static_cols),
    )
    by = {s: int((prefix_split == i).sum()) for s, i in SPLIT_TO_ID.items()}
    print(f"  prefixes by split: {by}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
