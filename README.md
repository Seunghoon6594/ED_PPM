# MIMIC-IV-ED Predictive Process Monitoring (PPM)

Predictive Process Monitoring on the MIMIC-IV Emergency Department dataset.
Each ED visit is treated as one process case. The pipeline abstracts the raw
clinical tables into an event log and trains models for two standard PPM tasks:

- Task 1 - Next-activity prediction: given the events observed so far in a
  visit, predict the next clinical event (6-class classification).
- Task 2 - Remaining-time prediction: given the events so far, predict the
  time remaining until the patient leaves the ED (regression).

Each task is solved with rule-based baselines, a LightGBM model on hand-crafted
prefix features, and an LSTM over the raw event sequence.

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
│       ├── train_next_activity.py       # Task 1: baselines + LightGBM
│       ├── train_remaining_time.py      # Task 2: baselines + LightGBM
│       ├── lstm_ppm.py                  # shared LSTM model / data loader
│       ├── train_lstm_next_activity.py  # Task 1: LSTM
│       ├── train_lstm_remaining_time.py # Task 2: LSTM
│       └── compare_models.py            # combined comparison table + figure
├── data/                         # event_log_statistics.csv (parquet outputs are regenerated)
├── figures/                      # event-log diagnostics
├── models/                       # trained LightGBM models
├── results/                      # metrics, confusion matrix, feature importance, plots
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended. Key dependencies: pandas, numpy, scikit-learn,
lightgbm, matplotlib, pyarrow.

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

# 5. Task 1 - next-activity prediction (baselines + LightGBM)
python scripts/analysis/train_next_activity.py

# 6. Task 2 - remaining-time prediction (baselines + LightGBM)
python scripts/analysis/train_remaining_time.py

# 7. LSTM models (build sequence dataset, then train both tasks)
python scripts/preprocessing/make_sequence_dataset.py
python scripts/analysis/train_lstm_next_activity.py
python scripts/analysis/train_lstm_remaining_time.py

# 8. Combined comparison table + figure
python scripts/analysis/compare_models.py
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
| LightGBM | 0.486 | 0.383 | 0.328 |
| LSTM | 0.458 | 0.445 | 0.633 |

The LSTM roughly doubles F1-macro (0.633 vs 0.328) and raises the recall of the
clinically important `ED_END` (visit-end) class from 0.05 to 0.77, while accuracy
stays level with the last-event baseline (the dominant vital-sign class caps raw
accuracy).

Task 2 - Remaining-time prediction

| Model | MAE (h) | RMSE (h) | MAPE (%) |
|-------|---------|----------|----------|
| Global-mean baseline | 4.293 | 6.505 | 155.3 |
| Last-event-mean baseline | 4.270 | 6.483 | 151.4 |
| LightGBM | 3.965 | 6.206 | 130.4 |
| LSTM | 3.580 | 6.460 | 80.8 |

The LSTM lowers MAE by ~10% over LightGBM and roughly halves MAPE (it is far more
accurate near discharge, where remaining time is small); its RMSE is marginally
higher because of larger residuals on the extreme long-stay tail.

## Configuration

All parameters (paths, LOS filters, split ratios, LightGBM hyperparameters,
random seed) are centralized in `configs/pipeline_config.py`. The random seed is
fixed at 42 for reproducibility.
