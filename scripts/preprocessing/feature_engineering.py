"""
feature_engineering.py
=======================
Prefix dataset에 추가 feature를 보강하고,
textual abstraction과 feature dictionary를 생성한다.

make_prefix_dataset.py에서 이미 생성된 structured features에 추가로:
  - 텍스트 기반 prefix 표현 (LLM 확장용)
  - feature dictionary (MD 문서)

출력:
  - data/textual_prefixes.parquet
  - data/feature_dictionary.md
"""

import sys
import os
import logging
import time

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

EVENT_DESCRIPTIONS = {
    cfg.EVENT_NAMES["arrival"]:   "arrived at the emergency department",
    cfg.EVENT_NAMES["triage"]:    "underwent triage assessment",
    cfg.EVENT_NAMES["vitalsign"]: "had vital signs measured",
    cfg.EVENT_NAMES["medrecon"]:  "had medication history reconciled",
    cfg.EVENT_NAMES["pyxis"]:     "received medication",
    cfg.EVENT_NAMES["end"]:       "left the emergency department",
}

ACUITY_MAP = {1: "immediately life-threatening", 2: "emergent", 3: "urgent",
              4: "less urgent", 5: "non-urgent"}


def build_textual_prefix_from_row(row) -> str:
    """
    Prefix dataset의 한 행에서 자연어 텍스트를 생성.
    prefix_events 리스트가 없으므로 event frequency counts와 last_event를 활용.
    """
    gender = int(row.get("gender", -1))
    acuity = row.get("acuity")
    prefix_length = int(row.get("prefix_length", 1))
    last_event = row.get("last_event", "UNKNOWN")
    elapsed = row.get("prefix_duration_min", 0)

    gender_str = "The patient"
    if gender == 1:
        gender_str = "The male patient"
    elif gender == 0:
        gender_str = "The female patient"

    acuity_str = ""
    if pd.notna(acuity):
        acuity_str = f" (acuity: {ACUITY_MAP.get(int(acuity), str(int(acuity)))})"

    last_desc = EVENT_DESCRIPTIONS.get(last_event, last_event)
    text = (
        f"{gender_str}{acuity_str} had {prefix_length} event(s) recorded over "
        f"{elapsed:.0f} minutes, most recently {last_desc}."
    )
    return text


def build_textual_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"  Building textual prefixes for {len(df):,} records...")
    texts = df.apply(build_textual_prefix_from_row, axis=1)
    result = pd.DataFrame({
        "case_id":       df["case_id"].values,
        "split":         df["split"].values,
        "prefix_length": df["prefix_length"].values,
        "text":          texts.values,
        "next_event":    df["next_event"].values if "next_event" in df.columns else None,
    })
    return result


def write_feature_dictionary(df: pd.DataFrame):
    """Structured feature의 컬럼 목록과 설명을 MD로 저장."""
    path = os.path.join(cfg.DATA_DIR, "feature_dictionary.md")

    exclude_cols = {"case_id", "split", "prefix_length", "last_event",
                    "next_event", "remaining_time_min"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    lines = ["# Feature Dictionary\n\n",
             "## Structured Features\n\n",
             "| Feature | Description | Type |\n|---------|-------------|------|\n"]

    for col in sorted(feature_cols):
        if col.startswith("freq_"):
            desc = f"Cumulative count of [{col[5:].upper()}] events in prefix"
            ftype = "int"
        elif col.startswith("last1_"):
            desc = f"1-hot: most recent event = {col[6:].upper()}"
            ftype = "binary"
        elif col.startswith("last2_"):
            desc = f"1-hot: 2nd most recent event = {col[6:].upper()}"
            ftype = "binary"
        elif col.startswith("last3_"):
            desc = f"1-hot: 3rd most recent event = {col[6:].upper()}"
            ftype = "binary"
        elif col.startswith("transport_"):
            desc = f"1-hot: arrival transport = {col[10:].upper()}"
            ftype = "binary"
        elif col == "prefix_duration_min":
            desc = "Elapsed time from case start to prefix end (min)"
            ftype = "float"
        elif col == "last_duration_min":
            desc = "Time since previous event (min)"
            ftype = "float"
        elif col == "avg_duration_min":
            desc = "Average inter-event time in prefix (min)"
            ftype = "float"
        elif col == "std_duration_min":
            desc = "Std of inter-event time in prefix (min)"
            ftype = "float"
        elif col == "arrival_hour":
            desc = "Hour of ED arrival (0-23)"
            ftype = "int"
        elif col == "arrival_dow":
            desc = "Day of week of ED arrival (0=Mon)"
            ftype = "int"
        elif col == "arrival_is_night":
            desc = "1 if arrival between 22:00-06:00"
            ftype = "binary"
        elif col == "acuity":
            desc = "Triage acuity level (1=critical, 5=non-urgent)"
            ftype = "float"
        elif col == "gender":
            desc = "Patient gender (1=M, 0=F, -1=unknown)"
            ftype = "int"
        elif col.startswith("triage_"):
            desc = f"Triage initial vital: {col[7:].upper()}"
            ftype = "float"
        else:
            desc = col
            ftype = "?"
        lines.append(f"| `{col}` | {desc} | {ftype} |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info(f"  Saved: {path}")
    return len(feature_cols)


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("FEATURE ENGINEERING")
    log.info("=" * 60)

    log.info("Loading prefix dataset (next activity)...")
    df_next = pd.read_parquet(cfg.PREFIX_NEXT_ACT_PATH)
    log.info(f"  {len(df_next):,} records, {len(df_next.columns)} columns")

    log.info("Building textual prefixes...")
    df_text = build_textual_prefixes(df_next)
    df_text.to_parquet(cfg.TEXTUAL_PREFIX_PATH, index=False)
    log.info(f"  Saved: {cfg.TEXTUAL_PREFIX_PATH}")

    log.info("Writing feature dictionary...")
    n_feats = write_feature_dictionary(df_next)

    log.info(f"Done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 50)
    print(f"  Structured features: {n_feats} columns")
    print(f"  Textual prefixes: {len(df_text):,}")
    print()
    print("Sample textual prefix:")
    print(df_text["text"].iloc[10])
    print("=" * 50)


if __name__ == "__main__":
    main()
