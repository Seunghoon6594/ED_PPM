# Extended analysis

## A1. Earliness - next-activity (Accuracy & F1-macro by prefix length k)

```
bucket  XGBoost_acc  XGBoost_f1m     n  LSTM_acc  LSTM_f1m
   2-2       0.5244       0.5579 63664    0.6107    0.5545
   3-3       0.4137       0.4315 62537    0.4240    0.4608
   4-4       0.3308       0.2914 59208    0.3760    0.3646
   5-5       0.3576       0.3471 53212    0.4391    0.4388
   6-7       0.3708       0.3785 81478    0.4634    0.4683
  8-10       0.3677       0.4019 73032    0.4558    0.4808
11-200       0.3959       0.4518 88724    0.4365    0.4869
```

## A1. Earliness - remaining-time (MAE by k)

```
bucket  XGBoost_mae     n  LSTM_mae
   2-2       3.4409 63664    3.4560
   3-3       3.3816 62537    3.3861
   4-4       3.3456 59208    3.3398
   5-5       3.4103 53212    3.3799
   6-7       3.5925 81478    3.5446
  8-10       3.8426 73032    3.7726
11-200       4.0151 88724    3.9622
```

## A2. Acuity subgroup

```
     group   model  n_next    acc  f1_macro  mae_h
high (1-2) XGBoost  227122 0.4117    0.4341 3.8818
high (1-2)    LSTM  227122 0.4559    0.6292 3.8537
 low (3-5) XGBoost  247612 0.3758    0.4940 3.3954
 low (3-5)    LSTM  247612 0.4605    0.5604 3.3577
```

## B5. Bootstrap 95% CI (test, [point, lo, hi])

```json
{
  "XGBoost": {
    "accuracy": [
      0.3945,
      0.3931,
      0.3959
    ],
    "f1_macro": [
      0.5703,
      0.4271,
      0.5961
    ],
    "mae_hours": [
      3.6104,
      3.597,
      3.6262
    ]
  },
  "LSTM": {
    "accuracy": [
      0.4582,
      0.4569,
      0.4597
    ],
    "f1_macro": [
      0.6334,
      0.6321,
      0.6343
    ],
    "mae_hours": [
      3.5804,
      3.5674,
      3.5958
    ]
  }
}
```