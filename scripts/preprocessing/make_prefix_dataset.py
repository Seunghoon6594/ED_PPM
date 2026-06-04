"""
make_prefix_dataset.py
======================
Event log -> Prefix dataset (벡터화 고속 구현).

핵심 아이디어:
  event log의 각 행은 잠재적인 "next event"를 나타냄.
  event_order >= MIN_PREFIX_LENGTH인 행만 추출하면
  해당 행의 case 내 이전 이벤트들이 prefix임.

  prefix 특성은 cumulative operations (cumsum, shift)으로 빠르게 계산.
  -> Python 루프 없이 전체 처리 가능.

두 가지 출력:
  Task 1: prefix -> next_event (next activity prediction)
  Task 2: prefix -> remaining_time_min (remaining time prediction)

저장 형식:
  compact dataset  : case_id, prefix_end_order, next_event, remaining_time
  structured feats : 모든 feature를 펼친 wide-format DataFrame
"""

import sys
import os
import logging
import time

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ==============================================================================
# Case-level train/val/test split
# ==============================================================================

def split_cases(case_ids: np.ndarray, seed: int = cfg.RANDOM_SEED) -> pd.Series:
    """Case ID -> split 레이블 매핑 반환."""
    np.random.seed(seed)
    shuffled = np.random.permutation(case_ids)
    n = len(shuffled)
    n_train = int(n * cfg.TRAIN_RATIO)
    n_val   = int(n * cfg.VAL_RATIO)

    split_map = {}
    for cid in shuffled[:n_train]:
        split_map[cid] = "train"
    for cid in shuffled[n_train:n_train + n_val]:
        split_map[cid] = "val"
    for cid in shuffled[n_train + n_val:]:
        split_map[cid] = "test"

    return split_map


# ==============================================================================
# Cumulative sequence features (벡터화)
# ==============================================================================

def compute_cumulative_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 event_order 위치까지의 event 누적 횟수 계산 (prefix frequency counts).

    방법:
      1. event_name을 one-hot
      2. case 내 cumsum -> 현재 행까지의 누적 count
      3. 현재 행 자체는 미래 정보이므로 shift(1) (직전 시점 기준)
    """
    log.info("  Computing cumulative event frequency counts...")
    # one-hot
    dummies = pd.get_dummies(df["event_name"], prefix="freq")
    dummies.index = df.index

    # case별 cumsum (현재 행 포함), 그런 다음 shift(1) -> prefix 기준
    result = dummies.groupby(df["case_id"]).cumsum().groupby(df["case_id"]).shift(1).fillna(0)

    return result.astype(int)


def compute_last_k_events(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """마지막 k개 이벤트 (prefix 기준 = 현재 행의 이전 k개)."""
    log.info(f"  Computing last-{k} events...")
    all_events = list(cfg.EVENT_NAMES.values()) + ["NONE"]
    frames = []

    for i in range(1, k + 1):
        shifted = df.groupby("case_id")["event_name"].shift(i).fillna("NONE")
        for evt in all_events:
            col_name = f"last{i}_{evt.lower()}"
            frames.append(pd.Series((shifted == evt).astype(int), name=col_name))

    return pd.concat(frames, axis=1)


# ==============================================================================
# Main prefix generation
# ==============================================================================

def build_prefix_datasets(df: pd.DataFrame, split_map: dict) -> tuple:
    """
    Event log -> prefix dataset (벡터화).

    각 행 i (event_order >= MIN_PREFIX_LENGTH)를 하나의 prefix record로 변환:
      - prefix = case의 0 ~ i-1 번째 이벤트
      - next_event = df['event_name'].iloc[i]
      - remaining_time = df['remaining_time_min'].iloc[i-1] (직전 행 기준)
    """
    log.info("Building prefix datasets (vectorized)...")

    # -- 1. Cumulative features 계산 ------------------------------------------
    cum_counts = compute_cumulative_event_counts(df)
    last_k     = compute_last_k_events(df, k=3)

    # -- 2. Temporal features -------------------------------------------------
    log.info("  Computing temporal features...")
    # prefix_duration = 직전 행의 elapsed_time (현재 행은 미래)
    df["_prefix_duration"] = df.groupby("case_id")["elapsed_time_min"].shift(1)
    # time_since_prev는 이미 df에 있음

    # 이동 평균/표준편차 계산을 위해 expanding window (prefix 기준 shift(1))
    df["_avg_dur"] = (
        df.groupby("case_id")["time_since_prev_min"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    df["_std_dur"] = (
        df.groupby("case_id")["time_since_prev_min"]
        .transform(lambda x: x.expanding().std().shift(1).fillna(0))
    )

    # -- 3. Case-level context ------------------------------------------------
    # 이미 event_log에 포함된 case-level columns 활용
    case_cols = ["acuity", "arrival_transport", "gender",
                 "triage_hr", "triage_sbp", "triage_o2sat",
                 "los_hours", "case_start"]

    # -- 4. Prefix records 필터링: event_order >= MIN_PREFIX_LENGTH -----------
    log.info(f"  Filtering rows with event_order >= {cfg.MIN_PREFIX_LENGTH}...")
    # "next event"를 구하기 위해 shift(-1)
    df["_next_event"]    = df.groupby("case_id")["event_name"].shift(-1)
    df["_next_elapsed"]  = df.groupby("case_id")["elapsed_time_min"].shift(-1)

    # 유효한 prefix row: event_order >= MIN_PREFIX_LENGTH - 1
    # (event_order k-1 행 = 길이 k prefix의 마지막 이벤트)
    # 그리고 next_event가 존재해야 함 (마지막 이벤트인 ED_END 제외)
    mask = (
        (df["event_order"] >= cfg.MIN_PREFIX_LENGTH - 1) &
        df["_next_event"].notna()
    )
    prefix_rows = df[mask].copy()
    log.info(f"  Valid prefix rows: {len(prefix_rows):,}")

    # prefix_length = event_order + 1 (0-indexed -> 1-indexed count)
    prefix_rows["prefix_length"] = prefix_rows["event_order"] + 1

    # -- 5. Split 할당 ---------------------------------------------------------
    prefix_rows["split"] = prefix_rows["case_id"].map(split_map).fillna("train")

    # -- 6. Remaining time = 직전 행(prefix 마지막)에서의 remaining_time -------
    # prefix 마지막 이벤트의 remaining_time = 현재 행의 remaining_time + inter-event time
    # 실제로는: remaining_time at (event_order i) = remaining_time_min already computed
    # 즉, prefix_rows['remaining_time_min'] = case_end - event_time at current row
    # 하지만 우리가 원하는 것은 prefix 마지막 이벤트(event_order = prefix_length-1)에서의 remaining time
    # 현재 prefix_rows는 이미 그 행임 (event_order = prefix_length - 1)
    # remaining_time_min은 이미 이 행의 remaining_time임

    # -- 7. Feature 조립 -------------------------------------------------------
    log.info("  Assembling feature matrix...")

    # cum_counts, last_k 인덱스 맞추기
    cum_part  = cum_counts.loc[prefix_rows.index]
    lastk_part = last_k.loc[prefix_rows.index]

    # arrival_hour, dow 계산
    if "case_start" in prefix_rows.columns:
        case_start = pd.to_datetime(prefix_rows["case_start"])
        prefix_rows["arrival_hour"]     = case_start.dt.hour
        prefix_rows["arrival_dow"]      = case_start.dt.dayofweek
        prefix_rows["arrival_is_night"] = ((case_start.dt.hour >= 22) | (case_start.dt.hour < 6)).astype(int)
    else:
        prefix_rows["arrival_hour"]     = -1
        prefix_rows["arrival_dow"]      = -1
        prefix_rows["arrival_is_night"] = -1

    # arrival_transport one-hot
    transport_dummies = pd.get_dummies(
        prefix_rows["arrival_transport"].fillna("UNKNOWN"),
        prefix="transport"
    )

    # gender 인코딩
    prefix_rows["gender_enc"] = prefix_rows["gender"].map({"M": 1, "F": 0}).fillna(-1)

    # 기본 컬럼 선택
    base_cols = prefix_rows[[
        "case_id", "split", "prefix_length", "event_name",
        "remaining_time_min",
        "_next_event",
        "acuity", "gender_enc",
        "triage_hr", "triage_sbp", "triage_o2sat",
        "arrival_hour", "arrival_dow", "arrival_is_night",
        "_prefix_duration", "time_since_prev_min", "_avg_dur", "_std_dur",
    ]].rename(columns={
        "event_name": "last_event",
        "_next_event": "next_event",
        "_prefix_duration": "prefix_duration_min",
        "time_since_prev_min": "last_duration_min",
        "_avg_dur": "avg_duration_min",
        "_std_dur": "std_duration_min",
        "gender_enc": "gender",
    })

    # 모두 합치기
    feat_df = pd.concat(
        [base_cols.reset_index(drop=True),
         cum_part.reset_index(drop=True),
         lastk_part.reset_index(drop=True),
         transport_dummies.reset_index(drop=True)],
        axis=1
    )

    # -- 8. Task별 분리 ---------------------------------------------------------
    # Task 1: next activity prediction
    df_next_act = feat_df.copy()

    # Task 2: remaining time prediction
    df_rt = feat_df.drop(columns=["next_event"]).copy()

    # 불필요 임시 컬럼 정리
    df.drop(columns=["_prefix_duration", "_avg_dur", "_std_dur", "_next_event", "_next_elapsed"],
            inplace=True, errors="ignore")

    return df_next_act, df_rt


# ==============================================================================
# Main
# ==============================================================================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("MAKE PREFIX DATASET (vectorized)")
    log.info("=" * 60)

    # 로딩
    log.info("Loading event log...")
    df = pd.read_parquet(cfg.EVENT_LOG_PATH)
    df["event_time"]  = pd.to_datetime(df["event_time"])
    df["case_start"]  = pd.to_datetime(df["case_start"])
    log.info(f"  {len(df):,} events, {df['case_id'].nunique():,} cases")

    # Case split
    case_ids = df["case_id"].unique()
    split_map = split_cases(case_ids)
    log.info(
        f"  Train: {sum(v=='train' for v in split_map.values()):,} | "
        f"Val: {sum(v=='val' for v in split_map.values()):,} | "
        f"Test: {sum(v=='test' for v in split_map.values()):,}"
    )

    # Prefix 생성
    df_next_act, df_rt = build_prefix_datasets(df, split_map)

    # 유효성 체크 및 저장
    log.info(f"  Next activity dataset: {len(df_next_act):,} records")
    log.info(f"    next_event distribution:\n{df_next_act['next_event'].value_counts().to_string()}")
    log.info(f"  Remaining time dataset: {len(df_rt):,} records")
    log.info(f"    remaining_time_min:\n{df_rt['remaining_time_min'].describe().to_string()}")

    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    df_next_act.to_parquet(cfg.PREFIX_NEXT_ACT_PATH, index=False)
    df_rt.to_parquet(cfg.PREFIX_REMAINING_PATH, index=False)
    log.info(f"  Saved: {cfg.PREFIX_NEXT_ACT_PATH}")
    log.info(f"  Saved: {cfg.PREFIX_REMAINING_PATH}")

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.1f}s")

    print("\n" + "=" * 50)
    print("PREFIX DATASET SUMMARY")
    print("=" * 50)
    for split in ["train", "val", "test"]:
        n = (df_next_act["split"] == split).sum()
        print(f"  {split}: {n:,} prefix records")
    print()
    print("Next event distribution:")
    print(df_next_act["next_event"].value_counts().to_string())
    print()
    print("Remaining time (minutes):")
    print(df_rt["remaining_time_min"].describe().to_string())
    print("=" * 50)


if __name__ == "__main__":
    main()
