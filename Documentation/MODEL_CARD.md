# Final Model Card

## Model

**Tuned XGBoost classifier**

Artifact:

```text
models/xgboost_final.json
```

## Objective

Predict the probability that a loan belongs to the bad-loan class.

Target:

```text
risk_label
```

```text
0 = Good Loan
1 = Bad Loan
```

## Training

Representative training sample:

```text
300,000 rows
```

The full train set contained:

```text
1,045,109 rows
```

The final XGBoost model was trained on a representative sample to keep training practical on local hardware.

## Features

The final model used 162 features after removing the 42 grade/sub-grade one-hot features.

Important features included:

```text
interest_rate
term
home_ownership_MORTGAGE
home_ownership_RENT
acc_open_past_24mths
verification_status_Not Verified
mort_acc
open_rv_24m
loan_to_income
dti
avg_cur_bal
all_util
installment_to_income
num_actv_rev_tl
high_dti
long_employment
```

## Hyperparameters

```text
max_depth = 6
learning_rate = 0.05
min_child_weight = 5
subsample = 0.7
colsample_bytree = 0.7
```

## Test performance

```text
ROC-AUC:  0.7299
Accuracy: 0.6678
Precision: 0.3352
Recall:   0.6643
F1:       0.4455
```

## Confusion matrix

```text
                 Predicted
               Good     Bad

Actual Good   139611   69176
Actual Bad     17619   34872
```

## Risk tiers

| Tier | Actual bad rate |
|---|---:|
| Low Risk | 6.17% |
| Medium Risk | 18.02% |
| High Risk | 39.90% |

## Interpretation

The model has useful discriminatory ability and produces risk scores that separate the test population into groups with materially different observed bad-loan rates.

However, the model is not perfect and should be treated as a decision-support component.

## Limitations

- Model performance is based on the available historical dataset.
- The model is not a guarantee of repayment behavior.
- Thresholds should ideally be selected using business costs and validation data.
- Financial decisions should not rely on this model alone.
- Fairness and regulatory review would be required for a real lending deployment.
- Production deployment would require monitoring for data drift and model degradation.
