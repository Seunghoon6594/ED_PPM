# Model Comparison (test set)

## Task 1 - Next-activity prediction

| Model | Accuracy | F1-weighted | F1-macro |
|-------|----------|-------------|----------|
| Most-frequent baseline | 0.442 | 0.271 | 0.102 |
| Last-event baseline | 0.458 | 0.340 | 0.338 |
| XGBoost | 0.395 | 0.368 | 0.570 |
| LSTM | 0.458 | 0.445 | 0.633 |
| TabPFN (3k/20k)* | 0.480 | 0.384 | 0.342 |

## Task 2 - Remaining-time prediction

| Model | MAE (h) | RMSE (h) | MAPE (%) |
|-------|---------|----------|----------|
| Global-mean baseline | 4.293 | 6.505 | 155.3 |
| Last-event-mean baseline | 4.270 | 6.483 | 151.4 |
| XGBoost | 3.610 | 6.504 | 81.7 |
| LSTM | 3.580 | 6.460 | 80.8 |
| TabPFN (3k/20k)* | 3.923 | 6.315 | 123.1 |

*TabPFN is fit on a small training subsample and evaluated on a test subsample (CPU in-context inference cost); all other models use the full training data and full test set, so TabPFN is a no-tuning reference point rather than a like-for-like row.
