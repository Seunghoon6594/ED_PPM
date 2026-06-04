"""
build_event_log.py
==================
MIMIC-IV-ED 원시 데이터 -> Process Mining용 Event Log 변환 파이프라인.

Event abstraction 규칙:
  1. ED_ARRIVAL          : edstays.intime (모든 케이스의 첫 이벤트)
  2. TRIAGE              : edstays.intime (triage 전용 timestamp 없음)
  3. VITAL_SIGN_REASSESSMENT : vitalsign.charttime (동일 시점 묶음)
  4. MEDICATION_RECONCILIATION : medrecon.charttime (동일 시점 묶음)
  5. MEDICATION_DISPENSED     : pyxis.charttime (동일 시점 묶음)
  6. ED_END              : edstays.outtime

출력: analysis_ppm/data/event_log_master.parquet
"""

import sys
import os
import logging
import time

import pandas as pd
import numpy as np

# configs 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

# -- 로깅 설정 ------------------------------------------------------------------
os.makedirs(cfg.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.LOGS_DIR, "build_event_log.log"), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ==============================================================================
# 1. 데이터 로딩
# ==============================================================================

def load_edstays() -> pd.DataFrame:
    """edstays 테이블 로딩 및 기본 전처리."""
    log.info("Loading edstays...")
    df = pd.read_csv(
        cfg.RAW_FILES["edstays"],
        parse_dates=["intime", "outtime"],
    )
    log.info(f"  edstays raw: {len(df):,} rows")

    # LOS 계산
    df["los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600

    # 이상 케이스 필터링
    n_before = len(df)
    df = df[
        (df["los_hours"] >= cfg.MIN_LOS_HOURS) &
        (df["los_hours"] <= cfg.MAX_LOS_HOURS) &
        df["intime"].notna() &
        df["outtime"].notna()
    ].copy()
    log.info(f"  After LOS filter [{cfg.MIN_LOS_HOURS}h, {cfg.MAX_LOS_HOURS}h]: {len(df):,} rows (removed {n_before - len(df):,})")

    return df


def load_triage() -> pd.DataFrame:
    """triage 테이블 로딩."""
    log.info("Loading triage...")
    df = pd.read_csv(cfg.RAW_FILES["triage"])
    log.info(f"  triage raw: {len(df):,} rows")
    return df


def load_vitalsign() -> pd.DataFrame:
    """vitalsign 테이블 로딩."""
    log.info("Loading vitalsign...")
    df = pd.read_csv(cfg.RAW_FILES["vitalsign"], parse_dates=["charttime"])
    log.info(f"  vitalsign raw: {len(df):,} rows")
    return df


def load_medrecon() -> pd.DataFrame:
    """medrecon 테이블 로딩."""
    log.info("Loading medrecon...")
    df = pd.read_csv(cfg.RAW_FILES["medrecon"], parse_dates=["charttime"])
    log.info(f"  medrecon raw: {len(df):,} rows")
    return df


def load_pyxis() -> pd.DataFrame:
    """pyxis 테이블 로딩."""
    log.info("Loading pyxis...")
    df = pd.read_csv(cfg.RAW_FILES["pyxis"], parse_dates=["charttime"])
    log.info(f"  pyxis raw: {len(df):,} rows")
    return df


# ==============================================================================
# 2. Event 추출 함수
# ==============================================================================

def extract_arrival_events(edstays: pd.DataFrame) -> pd.DataFrame:
    """
    ED_ARRIVAL 이벤트 추출.
    - 소스: edstays.intime
    - 1 stay = 1 event
    - 보조 정보: arrival_transport, gender, race
    """
    events = pd.DataFrame({
        "case_id":           edstays["stay_id"],
        "subject_id":        edstays["subject_id"],
        "event_name":        cfg.EVENT_NAMES["arrival"],
        "event_time":        edstays["intime"],
        "event_source_table": "edstays",
        "attr_arrival_transport": edstays["arrival_transport"],
        "attr_gender":       edstays["gender"],
        "attr_race":         edstays["race"],
    })
    log.info(f"  ED_ARRIVAL events: {len(events):,}")
    return events


def extract_triage_events(edstays: pd.DataFrame, triage: pd.DataFrame) -> pd.DataFrame:
    """
    TRIAGE 이벤트 추출.
    - triage 전용 timestamp 없음 -> edstays.intime 사용
    - 보조 정보: acuity, chiefcomplaint, initial vitals
    """
    merged = edstays[["stay_id", "intime"]].merge(triage, on="stay_id", how="inner")

    events = pd.DataFrame({
        "case_id":            merged["stay_id"],
        "subject_id":         merged["subject_id"],
        "event_name":         cfg.EVENT_NAMES["triage"],
        "event_time":         merged["intime"],   # intime을 triage 시점으로 대리 사용
        "event_source_table": "triage",
        "attr_acuity":        merged["acuity"],
        "attr_chiefcomplaint": merged["chiefcomplaint"],
        "attr_triage_hr":     merged["heartrate"],
        "attr_triage_sbp":    merged["sbp"],
        "attr_triage_o2sat":  merged["o2sat"],
        "attr_triage_temp":   merged["temperature"],
        "attr_triage_pain":   merged["pain"],
    })
    log.info(f"  TRIAGE events: {len(events):,}")
    return events


def extract_vitalsign_events(vitalsign: pd.DataFrame, edstays: pd.DataFrame) -> pd.DataFrame:
    """
    VITAL_SIGN_REASSESSMENT 이벤트 추출.
    - 동일 stay의 동일 charttime -> 1 event로 묶음 (집계)
    - outtime 이후 기록 제거
    - intime보다 60분 이상 이전 기록 제거
    """
    # edstays의 intime, outtime 조인
    df = vitalsign.merge(
        edstays[["stay_id", "intime", "outtime"]],
        on="stay_id", how="inner"
    )

    # offset 계산
    df["offset_min"] = (df["charttime"] - df["intime"]).dt.total_seconds() / 60

    # 필터: outtime 이후 제거
    if cfg.FILTER_VITALS_AFTER_OUTTIME:
        n_before = len(df)
        df = df[df["charttime"] <= df["outtime"]]
        log.info(f"  vitalsign: removed {n_before - len(df):,} records after outtime")

    # 필터: 너무 이른 기록 제거
    n_before = len(df)
    df = df[df["offset_min"] >= cfg.MAX_NEGATIVE_OFFSET_MINUTES]
    log.info(f"  vitalsign: removed {n_before - len(df):,} records with offset < {cfg.MAX_NEGATIVE_OFFSET_MINUTES} min")

    # stay_id + charttime 단위로 묶음
    grp = df.groupby(["stay_id", "charttime"], as_index=False).agg(
        subject_id=("subject_id", "first"),
        attr_hr=("heartrate", "mean"),
        attr_sbp=("sbp", "mean"),
        attr_dbp=("dbp", "mean"),
        attr_o2sat=("o2sat", "mean"),
        attr_temp=("temperature", "mean"),
        attr_resprate=("resprate", "mean"),
        attr_pain=("pain", "first"),
    )

    events = pd.DataFrame({
        "case_id":            grp["stay_id"],
        "subject_id":         grp["subject_id"],
        "event_name":         cfg.EVENT_NAMES["vitalsign"],
        "event_time":         grp["charttime"],
        "event_source_table": "vitalsign",
        "attr_hr":            grp["attr_hr"],
        "attr_sbp":           grp["attr_sbp"],
        "attr_dbp":           grp["attr_dbp"],
        "attr_o2sat":         grp["attr_o2sat"],
        "attr_temp":          grp["attr_temp"],
        "attr_resprate":      grp["attr_resprate"],
        "attr_pain":          grp["attr_pain"],
    })
    log.info(f"  VITAL_SIGN_REASSESSMENT events: {len(events):,} (from {len(vitalsign):,} raw rows)")
    return events


def extract_medrecon_events(medrecon: pd.DataFrame, edstays: pd.DataFrame) -> pd.DataFrame:
    """
    MEDICATION_RECONCILIATION 이벤트 추출.
    - 환자의 기존 복용약 확인 이벤트
    - 동일 stay의 동일 charttime -> 1 event로 묶음
    - 약물 목록을 문자열로 요약
    """
    df = medrecon.merge(
        edstays[["stay_id", "intime", "outtime"]],
        on="stay_id", how="inner"
    )

    # offset 필터
    df["offset_min"] = (df["charttime"] - df["intime"]).dt.total_seconds() / 60
    n_before = len(df)
    df = df[df["offset_min"] >= cfg.MAX_NEGATIVE_OFFSET_MINUTES]
    log.info(f"  medrecon: removed {n_before - len(df):,} records with offset < {cfg.MAX_NEGATIVE_OFFSET_MINUTES} min")

    # outtime 이후 기록 제거 (pyxis와 동일하게 처리)
    n_before = len(df)
    df = df[df["charttime"] <= df["outtime"]]
    log.info(f"  medrecon: removed {n_before - len(df):,} records after outtime")

    # stay_id + charttime 단위로 묶음 (lambda 최소화로 속도 개선)
    grp_base = df.groupby(["stay_id", "charttime"], as_index=False).agg(
        subject_id=("subject_id", "first"),
        attr_med_count=("name", "count"),
    )
    # 약물명/분류 요약: join 전에 별도 처리
    name_agg = (
        df.dropna(subset=["name"])
        .groupby(["stay_id", "charttime"])["name"]
        .apply(lambda x: "|".join(x.unique()[:10]))
        .reset_index(name="attr_med_names")
    )
    class_agg = (
        df.dropna(subset=["etcdescription"])
        .groupby(["stay_id", "charttime"])["etcdescription"]
        .apply(lambda x: "|".join(x.unique()[:5]))
        .reset_index(name="attr_med_classes")
    )
    grp = grp_base.merge(name_agg, on=["stay_id", "charttime"], how="left")
    grp = grp.merge(class_agg, on=["stay_id", "charttime"], how="left")

    events = pd.DataFrame({
        "case_id":            grp["stay_id"],
        "subject_id":         grp["subject_id"],
        "event_name":         cfg.EVENT_NAMES["medrecon"],
        "event_time":         grp["charttime"],
        "event_source_table": "medrecon",
        "attr_med_count":     grp["attr_med_count"],
        "attr_med_names":     grp["attr_med_names"],
        "attr_med_classes":   grp["attr_med_classes"],
    })
    log.info(f"  MEDICATION_RECONCILIATION events: {len(events):,} (from {len(medrecon):,} raw rows)")
    return events


def extract_pyxis_events(pyxis: pd.DataFrame, edstays: pd.DataFrame) -> pd.DataFrame:
    """
    MEDICATION_DISPENSED 이벤트 추출.
    - 응급실 내 실제 투약 (Pyxis 자동 분배기)
    - 동일 stay의 동일 charttime -> 1 event로 묶음
    """
    df = pyxis.merge(
        edstays[["stay_id", "intime", "outtime"]],
        on="stay_id", how="inner"
    )

    # offset 필터
    df["offset_min"] = (df["charttime"] - df["intime"]).dt.total_seconds() / 60
    n_before = len(df)
    df = df[df["offset_min"] >= cfg.MAX_NEGATIVE_OFFSET_MINUTES]
    log.info(f"  pyxis: removed {n_before - len(df):,} records with offset < {cfg.MAX_NEGATIVE_OFFSET_MINUTES} min")

    # outtime 이후 기록 제거
    n_before = len(df)
    df = df[df["charttime"] <= df["outtime"]]
    log.info(f"  pyxis: removed {n_before - len(df):,} records after outtime")

    # stay_id + charttime 단위로 묶음
    grp_base = df.groupby(["stay_id", "charttime"], as_index=False).agg(
        subject_id=("subject_id", "first"),
        attr_med_count=("name", "count"),
    )
    name_agg = (
        df.dropna(subset=["name"])
        .groupby(["stay_id", "charttime"])["name"]
        .apply(lambda x: "|".join(x.unique()[:10]))
        .reset_index(name="attr_med_names")
    )
    grp = grp_base.merge(name_agg, on=["stay_id", "charttime"], how="left")

    events = pd.DataFrame({
        "case_id":            grp["stay_id"],
        "subject_id":         grp["subject_id"],
        "event_name":         cfg.EVENT_NAMES["pyxis"],
        "event_time":         grp["charttime"],
        "event_source_table": "pyxis",
        "attr_med_count":     grp["attr_med_count"],
        "attr_med_names":     grp["attr_med_names"],
    })
    log.info(f"  MEDICATION_DISPENSED events: {len(events):,} (from {len(pyxis):,} raw rows)")
    return events


def extract_end_events(edstays: pd.DataFrame) -> pd.DataFrame:
    """
    ED_END 이벤트 추출.
    - 소스: edstays.outtime
    - disposition은 target/label 후보 -> 별도 컬럼으로 저장
    """
    events = pd.DataFrame({
        "case_id":            edstays["stay_id"],
        "subject_id":         edstays["subject_id"],
        "event_name":         cfg.EVENT_NAMES["end"],
        "event_time":         edstays["outtime"],
        "event_source_table": "edstays",
        "attr_disposition":   edstays["disposition"],
    })
    log.info(f"  ED_END events: {len(events):,}")
    return events


# ==============================================================================
# 3. Event log 조립 및 파생 컬럼 생성
# ==============================================================================

def assemble_event_log(event_dfs: list, edstays: pd.DataFrame) -> pd.DataFrame:
    """
    개별 event DataFrame들을 하나의 event log로 조립.

    추가 컬럼:
      - event_order          : case 내 시간순 순서 (0-indexed)
      - elapsed_time_min     : case 시작(intime)부터 경과 분
      - time_since_prev_min  : 직전 이벤트로부터 경과 분
      - case_los_hours       : case 전체 LOS
      - case_n_events        : case 내 총 이벤트 수
    """
    log.info("Assembling event log...")

    # 전체 concat
    log.info("  Concatenating event DataFrames...")
    all_events = pd.concat(event_dfs, ignore_index=True, sort=False)
    log.info(f"  Total events before dedup/sort: {len(all_events):,}")

    # case_id가 유효한 stay_id인 것만 유지
    valid_stays = set(edstays["stay_id"])
    all_events = all_events[all_events["case_id"].isin(valid_stays)].copy()
    log.info(f"  After filtering to valid stay_ids: {len(all_events):,}")

    # event_time 기준 정렬 (같은 시각이면 event_name 알파벳 순 -> 재현 가능)
    # ARRIVAL < TRIAGE < 나머지 (같은 intime에서 ARRIVAL이 먼저 오도록)
    event_order_map = {
        cfg.EVENT_NAMES["arrival"]:   0,
        cfg.EVENT_NAMES["triage"]:    1,
        cfg.EVENT_NAMES["vitalsign"]: 2,
        cfg.EVENT_NAMES["medrecon"]:  2,
        cfg.EVENT_NAMES["pyxis"]:     2,
        cfg.EVENT_NAMES["end"]:       99,
    }
    all_events["_sort_priority"] = all_events["event_name"].map(event_order_map).fillna(2)
    all_events.sort_values(
        ["case_id", "event_time", "_sort_priority", "event_name"],
        inplace=True,
        na_position="last"
    )
    all_events.drop(columns=["_sort_priority"], inplace=True)
    all_events.reset_index(drop=True, inplace=True)

    # case_id별 파생 컬럼 계산
    log.info("  Computing derived columns (event_order, elapsed_time, time_since_prev)...")
    all_events["event_order"] = all_events.groupby("case_id").cumcount()

    # edstays에서 intime, outtime, los 조인
    case_info = edstays[["stay_id", "intime", "outtime", "los_hours"]].rename(
        columns={"stay_id": "case_id", "intime": "case_start", "outtime": "case_end"}
    )
    all_events = all_events.merge(case_info, on="case_id", how="left")

    # elapsed_time_min: case 시작(intime)부터 현재 이벤트까지
    all_events["elapsed_time_min"] = (
        all_events["event_time"] - all_events["case_start"]
    ).dt.total_seconds() / 60

    # time_since_prev_min: 직전 이벤트로부터 경과 시간
    all_events["time_since_prev_min"] = (
        all_events.groupby("case_id")["elapsed_time_min"].diff().fillna(0)
    )

    # case 내 총 이벤트 수
    case_event_counts = all_events.groupby("case_id").size().rename("case_n_events")
    all_events = all_events.merge(case_event_counts, on="case_id", how="left")

    # remaining_time_min: 현재 이벤트부터 case 종료까지 (레이블 후보)
    all_events["remaining_time_min"] = (
        all_events["case_end"] - all_events["event_time"]
    ).dt.total_seconds() / 60

    log.info(f"  Final event log: {len(all_events):,} events, {all_events['case_id'].nunique():,} cases")
    return all_events


def filter_short_cases(event_log: pd.DataFrame) -> pd.DataFrame:
    """최소 이벤트 수 미만의 케이스 제거."""
    n_before = event_log["case_id"].nunique()
    valid_cases = (
        event_log.groupby("case_id")["event_order"].max() + 1 >= cfg.MIN_EVENTS_PER_CASE
    )
    valid_case_ids = valid_cases[valid_cases].index
    filtered = event_log[event_log["case_id"].isin(valid_case_ids)].copy()
    log.info(
        f"  After min_events filter ({cfg.MIN_EVENTS_PER_CASE}): "
        f"{filtered['case_id'].nunique():,} cases (removed {n_before - filtered['case_id'].nunique():,})"
    )
    return filtered


# ==============================================================================
# 4. Case-level context 추가
# ==============================================================================

def add_case_context(event_log: pd.DataFrame, edstays: pd.DataFrame, triage: pd.DataFrame) -> pd.DataFrame:
    """
    case_id 단위의 맥락 정보 추가.
    이 정보들은 예측 feature로 사용 가능하며, leakage가 없는 것들만 포함.
      - arrival_transport, gender, race (edstays)
      - acuity, chiefcomplaint, initial vitals (triage)
    disposition, hadm_id 등 결과 관련 정보는 여기서 추가하지 않음.
    """
    log.info("Adding case-level context...")

    case_ctx = edstays[["stay_id", "arrival_transport", "gender", "race"]].rename(
        columns={"stay_id": "case_id"}
    )
    triage_ctx = triage[["stay_id", "acuity", "chiefcomplaint",
                          "heartrate", "sbp", "dbp", "o2sat", "temperature", "resprate", "pain"]].rename(
        columns={
            "stay_id": "case_id",
            "heartrate": "triage_hr",
            "sbp": "triage_sbp",
            "dbp": "triage_dbp",
            "o2sat": "triage_o2sat",
            "temperature": "triage_temp",
            "resprate": "triage_resprate",
            "pain": "triage_pain",
        }
    )

    # stay 단위 de-duplicate 후 join (1:1 보장)
    ctx = case_ctx.drop_duplicates("case_id").merge(
        triage_ctx.drop_duplicates("case_id"), on="case_id", how="left"
    )

    # 이미 event_log에 있는 컬럼은 제외 (case_id 제외)
    existing_cols = set(event_log.columns) - {"case_id"}
    new_cols = [c for c in ctx.columns if c not in existing_cols]
    ctx = ctx[new_cols]  # case_id 포함

    event_log = event_log.merge(ctx, on="case_id", how="left")

    added = [c for c in new_cols if c != "case_id"]
    log.info(f"  Case context added. New columns: {added}")
    return event_log


# ==============================================================================
# 5. 저장
# ==============================================================================

def save_event_log(event_log: pd.DataFrame) -> None:
    """Event log를 parquet 형식으로 저장."""
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    event_log.to_parquet(cfg.EVENT_LOG_PATH, index=False)
    size_mb = os.path.getsize(cfg.EVENT_LOG_PATH) / 1024 / 1024
    log.info(f"  Saved: {cfg.EVENT_LOG_PATH} ({size_mb:.1f} MB)")


# ==============================================================================
# 6. 메인 실행
# ==============================================================================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("BUILD EVENT LOG - MIMIC-IV-ED")
    log.info("=" * 60)

    # 1. 로딩
    edstays   = load_edstays()
    triage    = load_triage()
    vitalsign = load_vitalsign()
    medrecon  = load_medrecon()
    pyxis     = load_pyxis()

    log.info("-" * 40)

    # 2. Event 추출
    log.info("Extracting events...")
    ev_arrival  = extract_arrival_events(edstays)
    ev_triage   = extract_triage_events(edstays, triage)
    ev_vitals   = extract_vitalsign_events(vitalsign, edstays)
    ev_medrecon = extract_medrecon_events(medrecon, edstays)
    ev_pyxis    = extract_pyxis_events(pyxis, edstays)
    ev_end      = extract_end_events(edstays)

    log.info("-" * 40)

    # 3. 조립
    event_log = assemble_event_log(
        [ev_arrival, ev_triage, ev_vitals, ev_medrecon, ev_pyxis, ev_end],
        edstays
    )

    # 4. 짧은 케이스 제거
    event_log = filter_short_cases(event_log)

    # 5. Case context 추가
    event_log = add_case_context(event_log, edstays, triage)

    # 6. 저장
    log.info("-" * 40)
    log.info("Saving event log...")
    save_event_log(event_log)

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.1f}s")
    log.info("=" * 60)

    # 간단한 요약 출력
    print("\n" + "=" * 50)
    print("EVENT LOG SUMMARY")
    print("=" * 50)
    print(f"Total cases     : {event_log['case_id'].nunique():,}")
    print(f"Total events    : {len(event_log):,}")
    print(f"Event types     : {event_log['event_name'].nunique()}")
    print()
    print("Event frequency:")
    print(event_log["event_name"].value_counts().to_string())
    print()
    print(f"Avg events/case : {len(event_log) / event_log['case_id'].nunique():.2f}")
    print(f"Median case LOS : {event_log.groupby('case_id')['los_hours'].first().median():.2f} hours")
    print("=" * 50)

    return event_log


if __name__ == "__main__":
    main()
