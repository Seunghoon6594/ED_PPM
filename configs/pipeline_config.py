"""
Pipeline configuration for MIMIC-IV-ED Predictive Process Monitoring.
All major parameters are centralized here.
"""

import os

# -- Paths ----------------------------------------------------------------------
# Repository root (the directory that contains configs/, scripts/, data/, ...).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MIMIC-IV-ED raw .csv.gz files. These are NOT included in this repository because
# the dataset requires PhysioNet credentialed access. Place the six .csv.gz files in
# a ./dataset/ folder at the repository root, or set the MIMIC_ED_DIR environment
# variable to point to wherever they are stored.
DATASET_DIR = os.environ.get("MIMIC_ED_DIR")
if not DATASET_DIR:
    _local = os.path.join(REPO_ROOT, "dataset")
    _parent = os.path.join(REPO_ROOT, "..", "dataset")
    DATASET_DIR = _local if os.path.isdir(_local) else _parent

DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

# Raw data files
RAW_FILES = {
    "edstays":   os.path.join(DATASET_DIR, "edstays.csv.gz"),
    "triage":    os.path.join(DATASET_DIR, "triage.csv.gz"),
    "vitalsign": os.path.join(DATASET_DIR, "vitalsign.csv.gz"),
    "medrecon":  os.path.join(DATASET_DIR, "medrecon.csv.gz"),
    "pyxis":     os.path.join(DATASET_DIR, "pyxis.csv.gz"),
    "diagnosis": os.path.join(DATASET_DIR, "diagnosis.csv.gz"),
}

# Processed / output files
EVENT_LOG_PATH          = os.path.join(DATA_DIR, "event_log_master.parquet")
PREFIX_NEXT_ACT_PATH    = os.path.join(DATA_DIR, "prefix_dataset_next_activity.parquet")
PREFIX_REMAINING_PATH   = os.path.join(DATA_DIR, "prefix_dataset_remaining_time.parquet")
STRUCTURED_FEAT_PATH    = os.path.join(DATA_DIR, "structured_features.parquet")
TEXTUAL_PREFIX_PATH     = os.path.join(DATA_DIR, "textual_prefixes.parquet")

# -- Reproducibility ------------------------------------------------------------
RANDOM_SEED = 42

# -- Case filtering -------------------------------------------------------------
MIN_LOS_HOURS = 0.1          # 최소 체류시간 (시간) - 0 또는 음수 LOS 제거
MAX_LOS_HOURS = 72.0         # 최대 체류시간 - 72h 초과는 비정형 케이스
MIN_EVENTS_PER_CASE = 2      # prefix 생성을 위한 최소 이벤트 수

# -- Event abstraction ----------------------------------------------------------
# triage의 고유 timestamp가 없으므로 ED_ARRIVAL과 TRIAGE 모두 intime을 사용.
# event_order 컬럼으로 순서를 구분한다 (ARRIVAL=0, TRIAGE=1).
TRIAGE_OFFSET_MINUTES = 0    # TRIAGE event를 intime + N분으로 설정

# vitalsign 이벤트: outtime 이후 기록 제거
FILTER_VITALS_AFTER_OUTTIME = True

# medrecon/pyxis: 같은 charttime의 여러 약물 -> 1 event로 묶음
GROUP_MEDS_BY_CHARTTIME = True

# 이상치 threshold: intime 이전 기록 허용 범위 (음수 offset)
MAX_NEGATIVE_OFFSET_MINUTES = -60   # 60분 이전까지는 허용 (기록 오류 범위)

# -- Event naming ---------------------------------------------------------------
EVENT_NAMES = {
    "arrival":    "ED_ARRIVAL",
    "triage":     "TRIAGE",
    "vitalsign":  "VITAL_SIGN_REASSESSMENT",
    "medrecon":   "MEDICATION_RECONCILIATION",
    "pyxis":      "MEDICATION_DISPENSED",
    "end":        "ED_END",
}

# -- Prefix generation ----------------------------------------------------------
MIN_PREFIX_LENGTH = 2        # 최소 prefix 길이 (이 길이 이상의 prefix만 생성)
MAX_PREFIX_LENGTH = None     # None = 제한 없음 (case 길이 - 1)

# -- Train / validation / test split (case-level) ------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# -- Model hyperparameters (baseline) ------------------------------------------
LGBM_PARAMS_CLASSIFICATION = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

LGBM_PARAMS_REGRESSION = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LSTM_HIDDEN_DIM   = 128
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.3
LSTM_BATCH_SIZE   = 256
LSTM_MAX_EPOCHS   = 50
LSTM_LEARNING_RATE = 1e-3
LSTM_PATIENCE     = 5        # early stopping

# -- Rare event handling --------------------------------------------------------
MIN_EVENT_FREQ = 100         # 이 빈도 미만의 event는 "OTHER"로 처리 여부 검토
