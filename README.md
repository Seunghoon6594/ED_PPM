# MIMIC-IV-ED Predictive Process Monitoring (PPM)

Predictive Process Monitoring on the MIMIC-IV Emergency Department dataset.
Each ED visit is treated as one process case. The pipeline abstracts the raw
clinical tables into an event log and trains models for two standard PPM tasks:

- Task 1 - Next-activity prediction: given the events observed so far in a
  visit, predict the next clinical event (6-class classification).
- Task 2 - Remaining-time prediction: given the events so far, predict the
  time remaining until the patient leaves the ED (regression).

Each task is solved with rule-based baselines, gradient-boosted trees (XGBoost)
and TabPFN on hand-crafted prefix features, and an LSTM over the raw event
sequence.

The core of the project is the event-log abstraction: turning five raw
clinical tables into a clean, leakage-safe activity sequence per visit.

## Repository structure

```
.
├── configs/
│   └── pipeline_config.py        # all paths, filters, and hyperparameters
├── scripts/
│   ├── preprocessing/
│   │   ├── build_event_log.py     # raw tables -> event_log_master.parquet
│   │   ├── make_prefix_dataset.py # event log  -> structured prefix datasets
│   │   ├── make_sequence_dataset.py # event log -> LSTM sequence dataset
│   │   └── feature_engineering.py # prefixes   -> textual features
│   └── analysis/
│       ├── check_event_log_quality.py
│       ├── train_next_activity.py       # Task 1: rule-based baselines
│       ├── train_remaining_time.py      # Task 2: rule-based baselines
│       ├── train_xgboost.py             # XGBoost (both tasks)
│       ├── train_tabpfn.py              # TabPFN (both tasks, subsampled)
│       ├── lstm_ppm.py                  # shared LSTM model / data loader
│       ├── train_lstm_next_activity.py  # Task 1: LSTM
│       ├── train_lstm_remaining_time.py # Task 2: LSTM
│       ├── train_lstm_ablation.py       # LSTM input ablation (Task 1)
│       ├── analyze_predictions.py       # earliness, acuity subgroups, bootstrap CIs
│       ├── make_process_map.py          # directly-follows transition heatmap
│       └── compare_models.py            # combined comparison table + figure
├── data/                         # event_log_statistics.csv (parquet outputs are regenerated)
├── figures/                      # event-log diagnostics, comparison + analysis plots
├── models/                       # trained XGBoost and LSTM models
├── results/                      # metrics, comparison tables, analysis outputs
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended. Key dependencies: pandas, numpy, scikit-learn,
xgboost, tabpfn, torch, matplotlib, pyarrow.

## Data

The raw data is not included in this repository: MIMIC-IV-ED requires
credentialed access through PhysioNet and may not be redistributed.

1. Obtain MIMIC-IV-ED from PhysioNet: https://physionet.org/content/mimic-iv-ed/
2. Place the six `.csv.gz` files in a `dataset/` folder at the repository root:

   ```
   dataset/
   ├── edstays.csv.gz
   ├── triage.csv.gz
   ├── vitalsign.csv.gz
   ├── medrecon.csv.gz
   ├── pyxis.csv.gz
   └── diagnosis.csv.gz
   ```

   Alternatively, set the `MIMIC_ED_DIR` environment variable to the folder that
   contains them.

## Usage

Run the pipeline from the repository root, in order:

```bash
# 1. Build the event log
python scripts/preprocessing/build_event_log.py

# 2. (optional) Event-log quality checks and figures
python scripts/analysis/check_event_log_quality.py

# 3. Build prefix datasets for both tasks
python scripts/preprocessing/make_prefix_dataset.py

# 4. Feature engineering (textual prefixes)
python scripts/preprocessing/feature_engineering.py

# 5. Rule-based baselines (both tasks)
python scripts/analysis/train_next_activity.py
python scripts/analysis/train_remaining_time.py

# 6. Tabular models on the structured prefix features
python scripts/analysis/train_xgboost.py
python scripts/analysis/train_tabpfn.py        # subsampled (CPU in-context cost)

# 7. LSTM over the raw event sequence
python scripts/preprocessing/make_sequence_dataset.py
python scripts/analysis/train_lstm_next_activity.py
python scripts/analysis/train_lstm_remaining_time.py

# 8. Comparison + post-hoc analyses + process map
python scripts/analysis/compare_models.py
python scripts/analysis/analyze_predictions.py   # earliness, acuity, bootstrap CIs
python scripts/analysis/train_lstm_ablation.py   # LSTM input ablation
python scripts/analysis/make_process_map.py
```

Each script reads its inputs and writes its outputs to the locations defined in
`configs/pipeline_config.py`.

## Event abstraction

Each ED visit (`stay_id`) is one case, starting at `intime` and ending at
`outtime`. Rows that share the same `(stay_id, timestamp)` are aggregated into a
single event. Six event types are derived:

| Event | Source table | Meaning |
|-------|--------------|---------|
| `ED_ARRIVAL` | edstays | Visit start (arrival/registration) |
| `TRIAGE` | triage | Initial assessment (acuity, vitals) |
| `VITAL_SIGN_REASSESSMENT` | vitalsign | Repeated vital-sign monitoring |
| `MEDICATION_RECONCILIATION` | medrecon | Prior home-medication review |
| `MEDICATION_DISPENSED` | pyxis | In-ED medication dispensing |
| `ED_END` | edstays | Visit end (disposition) |

`diagnosis` is excluded because ICD codes are recorded at discharge and would
leak future information. Cases are kept only when length of stay is within
`[0.1h, 72h]`.

Leakage control: case-level train/validation/test split (70/15/15), no future
events inside a prefix, and `disposition` / `hadm_id` / `diagnosis` are never
used as features.

## Results (test set)

Task 1 - Next-activity prediction

| Model | Accuracy | F1-weighted | F1-macro |
|-------|----------|-------------|----------|
| Most-frequent baseline | 0.442 | 0.271 | 0.102 |
| Last-event baseline | 0.458 | 0.340 | 0.338 |
| XGBoost | 0.395 | 0.368 | 0.570 |
| TabPFN (3k/20k subsample) | 0.480 | 0.384 | 0.342 |
| LSTM | 0.458 | 0.445 | 0.633 |

The LSTM leads on both accuracy and F1-macro: it beats the balanced XGBoost on
both axes and reaches recall ~0.77 on the clinically important `ED_END`
(visit-end) class. Its advantage holds at every prefix length and is larger for
high-acuity (severe) patients (see `figures/earliness_next_activity.png`,
`figures/acuity_subgroup.png`).

Task 2 - Remaining-time prediction

| Model | MAE (h) | RMSE (h) | MAPE (%) |
|-------|---------|----------|----------|
| Global-mean baseline | 4.293 | 6.505 | 155.3 |
| Last-event-mean baseline | 4.270 | 6.483 | 151.4 |
| XGBoost | 3.610 | 6.504 | 81.7 |
| TabPFN (3k/20k subsample) | 3.923 | 6.315 | 123.1 |
| LSTM | 3.580 | 6.460 | 80.8 |

For remaining time the LSTM and the MAE-objective XGBoost are practically
comparable (~3.6 h MAE, ~81% MAPE), both ~10% better than the best baseline:
sequence modelling helps Task 1 substantially but Task 2 only marginally.

TabPFN is fit on a 3,000-row training subsample and evaluated on a 20,000-row
test subsample (CPU in-context inference cost), so it is a no-tuning reference
point rather than a like-for-like row.

## Configuration

All parameters (paths, LOS filters, split ratios, model hyperparameters,
random seed) are centralized in `configs/pipeline_config.py`. The random seed is
fixed at 42 for reproducibility.
