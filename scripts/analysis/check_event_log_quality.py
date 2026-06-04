"""
check_event_log_quality.py
==========================
Event log 품질 점검 스크립트.

점검 항목:
  1. 기본 통계 (case 수, event 수, event 종류)
  2. Event frequency 분포
  3. Case 길이 분포
  4. 비정상 케이스 (음수 elapsed, 시간 역전, 너무 짧은/긴 case)
  5. Timestamp 역전 여부
  6. Sample trace 20개
  7. Event sequence 패턴 분석

출력:
  - analysis_ppm/data/event_log_statistics.csv
  - analysis_ppm/figures/event_frequency.png
  - analysis_ppm/figures/case_length_distribution.png
  - analysis_ppm/figures/event_timeline_samples.png
  - analysis_ppm/logs/event_log_quality_report.md
"""

import sys
import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import pipeline_config as cfg

# -- 한글 폰트 설정 -------------------------------------------------------------
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# -- 로깅 -----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def load_event_log() -> pd.DataFrame:
    log.info(f"Loading event log from {cfg.EVENT_LOG_PATH}")
    df = pd.read_parquet(cfg.EVENT_LOG_PATH)
    df["event_time"] = pd.to_datetime(df["event_time"])
    log.info(f"  Loaded: {len(df):,} events, {df['case_id'].nunique():,} cases")
    return df


# ==============================================================================
# 1. 기본 통계
# ==============================================================================

def basic_stats(df: pd.DataFrame) -> dict:
    case_lengths = df.groupby("case_id").size()
    los = df.groupby("case_id")["los_hours"].first()

    stats = {
        "total_cases": int(df["case_id"].nunique()),
        "total_events": int(len(df)),
        "event_types": int(df["event_name"].nunique()),
        "avg_events_per_case": float(case_lengths.mean()),
        "median_events_per_case": float(case_lengths.median()),
        "min_events_per_case": int(case_lengths.min()),
        "max_events_per_case": int(case_lengths.max()),
        "avg_los_hours": float(los.mean()),
        "median_los_hours": float(los.median()),
    }
    return stats


# ==============================================================================
# 2. 이상 케이스 탐지
# ==============================================================================

def detect_anomalies(df: pd.DataFrame) -> dict:
    issues = {}

    # 음수 elapsed_time
    neg_elapsed = df[df["elapsed_time_min"] < 0]
    issues["negative_elapsed_time_events"] = len(neg_elapsed)

    # 시간 역전 (time_since_prev < 0)
    time_reversal = df[df["time_since_prev_min"] < 0]
    issues["time_reversal_events"] = len(time_reversal)
    issues["time_reversal_cases"] = int(time_reversal["case_id"].nunique())

    # 음수 remaining_time
    neg_remaining = df[df["remaining_time_min"] < 0]
    issues["negative_remaining_time_events"] = len(neg_remaining)

    # 첫 이벤트가 ED_ARRIVAL이 아닌 케이스
    first_events = df[df["event_order"] == 0]
    not_arrival = first_events[first_events["event_name"] != cfg.EVENT_NAMES["arrival"]]
    issues["cases_not_starting_with_arrival"] = int(not_arrival["case_id"].nunique())

    # 마지막 이벤트가 ED_END가 아닌 케이스
    last_events = df.loc[df.groupby("case_id")["event_order"].idxmax()]
    not_end = last_events[last_events["event_name"] != cfg.EVENT_NAMES["end"]]
    issues["cases_not_ending_with_ed_end"] = int(not_end["case_id"].nunique())

    return issues


# ==============================================================================
# 3. 시각화
# ==============================================================================

def plot_event_frequency(df: pd.DataFrame):
    freq = df["event_name"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(freq.index[::-1], freq.values[::-1], color="#4C72B0", edgecolor="white")
    for bar, val in zip(bars, freq.values[::-1]):
        ax.text(bar.get_width() + freq.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlabel("Event Count")
    ax.set_title("Event Frequency Distribution")
    ax.set_xlim(0, freq.max() * 1.15)
    plt.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, "event_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


def plot_case_length_distribution(df: pd.DataFrame):
    case_lengths = df.groupby("case_id").size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: case 길이 히스토그램
    axes[0].hist(case_lengths, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(case_lengths.mean(), color="red", linestyle="--", label=f"Mean={case_lengths.mean():.1f}")
    axes[0].axvline(case_lengths.median(), color="orange", linestyle="--", label=f"Median={case_lengths.median():.1f}")
    axes[0].set_xlabel("Number of Events per Case")
    axes[0].set_ylabel("Case Count")
    axes[0].set_title("Case Length Distribution")
    axes[0].legend()

    # 오른쪽: LOS 분포
    los = df.groupby("case_id")["los_hours"].first()
    axes[1].hist(los.clip(0, 24), bins=50, color="#DD8452", edgecolor="white", alpha=0.85)
    axes[1].axvline(los.mean(), color="red", linestyle="--", label=f"Mean={los.mean():.1f}h")
    axes[1].axvline(los.median(), color="orange", linestyle="--", label=f"Median={los.median():.1f}h")
    axes[1].set_xlabel("LOS (hours, clipped at 24h)")
    axes[1].set_ylabel("Case Count")
    axes[1].set_title("ED Length of Stay Distribution")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, "case_length_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


def plot_event_position_heatmap(df: pd.DataFrame):
    """각 event_order 위치별 event 분포 히트맵."""
    # 최대 10 위치만 표시
    sub = df[df["event_order"] < 10].copy()
    pivot = sub.groupby(["event_order", "event_name"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.T.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels([f"pos {i}" for i in pivot.index])
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns, fontsize=9)
    ax.set_title("Event Distribution by Position (first 10 events)")
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, "event_position_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


# ==============================================================================
# 4. Sample traces
# ==============================================================================

def sample_traces(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """다양한 길이의 케이스에서 샘플 trace 추출."""
    np.random.seed(cfg.RANDOM_SEED)

    case_lengths = df.groupby("case_id").size()
    # 짧은/중간/긴 케이스 균등 샘플
    short_ids = case_lengths[case_lengths <= 4].sample(min(7, len(case_lengths[case_lengths <= 4]))).index
    mid_ids   = case_lengths[(case_lengths > 4) & (case_lengths <= 8)].sample(min(7, len(case_lengths[(case_lengths > 4) & (case_lengths <= 8)]))).index
    long_ids  = case_lengths[case_lengths > 8].sample(min(6, len(case_lengths[case_lengths > 8]))).index

    sample_ids = list(short_ids) + list(mid_ids) + list(long_ids)
    sample_ids = sample_ids[:n]

    traces = df[df["case_id"].isin(sample_ids)][
        ["case_id", "event_order", "event_name", "event_time", "elapsed_time_min", "time_since_prev_min"]
    ].sort_values(["case_id", "event_order"])

    return traces


# ==============================================================================
# 5. 통계 저장
# ==============================================================================

def save_statistics(df: pd.DataFrame, stats: dict, anomalies: dict):
    # event frequency CSV
    freq = df["event_name"].value_counts().reset_index()
    freq.columns = ["event_name", "count"]
    freq["percentage"] = (freq["count"] / len(df) * 100).round(2)
    freq["cases_with_event"] = freq["event_name"].map(
        df.groupby("event_name")["case_id"].nunique()
    )
    freq["case_coverage_pct"] = (freq["cases_with_event"] / df["case_id"].nunique() * 100).round(2)
    freq_path = os.path.join(cfg.DATA_DIR, "event_log_statistics.csv")
    freq.to_csv(freq_path, index=False)
    log.info(f"  Saved: {freq_path}")
    return freq


def write_quality_report(stats: dict, anomalies: dict, freq: pd.DataFrame, traces: pd.DataFrame):
    report_path = os.path.join(cfg.LOGS_DIR, "event_log_quality_report.md")

    lines = []
    lines.append("# Event Log Quality Report\n")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    lines.append("## 1. Basic Statistics\n")
    lines.append(f"| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| Total cases | {stats['total_cases']:,} |\n")
    lines.append(f"| Total events | {stats['total_events']:,} |\n")
    lines.append(f"| Event types | {stats['event_types']} |\n")
    lines.append(f"| Avg events/case | {stats['avg_events_per_case']:.2f} |\n")
    lines.append(f"| Median events/case | {stats['median_events_per_case']:.1f} |\n")
    lines.append(f"| Min events/case | {stats['min_events_per_case']} |\n")
    lines.append(f"| Max events/case | {stats['max_events_per_case']} |\n")
    lines.append(f"| Avg LOS (hours) | {stats['avg_los_hours']:.2f} |\n")
    lines.append(f"| Median LOS (hours) | {stats['median_los_hours']:.2f} |\n\n")

    lines.append("## 2. Event Frequency\n")
    lines.append("| Event Name | Count | % of Events | Case Coverage (%) |\n")
    lines.append("|------------|-------|-------------|-------------------|\n")
    for _, row in freq.iterrows():
        lines.append(f"| {row['event_name']} | {row['count']:,} | {row['percentage']}% | {row['case_coverage_pct']}% |\n")
    lines.append("\n")

    lines.append("## 3. Anomaly Detection\n")
    lines.append(f"| Issue | Count |\n|-------|-------|\n")
    for k, v in anomalies.items():
        lines.append(f"| {k} | {v:,} |\n")
    lines.append("\n")

    lines.append("## 4. Visualizations\n")
    lines.append("![Event Frequency](../figures/event_frequency.png)\n")
    lines.append("*그림 1: Event 종류별 발생 빈도*\n- 생성 스크립트: `scripts/analysis/check_event_log_quality.py`\n\n")
    lines.append("![Case Length Distribution](../figures/case_length_distribution.png)\n")
    lines.append("*그림 2: Case 길이(이벤트 수) 및 LOS 분포*\n\n")
    lines.append("![Event Position Heatmap](../figures/event_position_heatmap.png)\n")
    lines.append("*그림 3: 위치별 Event 분포 히트맵*\n\n")

    lines.append("## 5. Sample Traces (20 cases)\n\n")
    lines.append("```\n")
    for case_id, grp in traces.groupby("case_id"):
        lines.append(f"[Case {case_id}]\n")
        for _, row in grp.iterrows():
            lines.append(
                f"  [{row['elapsed_time_min']:6.0f}min] {row['event_name']:<35} "
                f"(+{row['time_since_prev_min']:.0f}min)\n"
            )
        lines.append("\n")
    lines.append("```\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info(f"  Saved: {report_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR, exist_ok=True)

    df = load_event_log()

    log.info("Computing basic statistics...")
    stats = basic_stats(df)
    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    log.info("Detecting anomalies...")
    anomalies = detect_anomalies(df)
    for k, v in anomalies.items():
        log.info(f"  {k}: {v}")

    log.info("Generating plots...")
    plot_event_frequency(df)
    plot_case_length_distribution(df)
    plot_event_position_heatmap(df)

    log.info("Sampling traces...")
    traces = sample_traces(df, n=20)

    log.info("Saving statistics and report...")
    freq = save_statistics(df, stats, anomalies)
    write_quality_report(stats, anomalies, freq, traces)

    print("\n" + "=" * 50)
    print("QUALITY CHECK COMPLETE")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nAnomalies:")
    for k, v in anomalies.items():
        flag = " [!]" if v > 0 else " [ok]"
        print(f"  {k}: {v}{flag}")
    print("=" * 50)


if __name__ == "__main__":
    main()
