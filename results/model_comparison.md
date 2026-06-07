# Model Comparison (test set)

## Task 1 - Next-activity prediction

| Model | Accuracy | F1-weighted | F1-macro |
|-------|----------|-------------|----------|
| Most-frequent baseline | 0.442 | 0.271 | 0.102 |
| Last-event baseline | 0.458 | 0.340 | 0.338 |
| LightGBM | 0.486 | 0.383 | 0.328 |
| LSTM | 0.458 | 0.445 | 0.633 |

## Task 2 - Remaining-time prediction

| Model | MAE (h) | RMSE (h) | MAPE (%) |
|-------|---------|----------|----------|
| Global-mean baseline | 4.293 | 6.505 | 155.3 |
| Last-event-mean baseline | 4.270 | 6.483 | 151.4 |
| LightGBM | 3.965 | 6.206 | 130.4 |
| LSTM | 3.580 | 6.460 | 80.8 |
