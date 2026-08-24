# Micro-Lending Credit Risk Model

## Dataset

- Original dataset: 2,260,668 loans
- Training rows: 1,045,109
- Test rows: 261,278
- Final training sample: 300,000

## Model Comparison

| model                 |   roc_auc |   accuracy |   precision |   recall |     f1 |
|:----------------------|----------:|-----------:|------------:|---------:|-------:|
| Logistic Regression   |    0.6415 |     0.6317 |      0.2833 |   0.5448 | 0.3728 |
| XGBoost               |    0.7266 |     0.6764 |      0.3386 |   0.6408 | 0.4431 |
| XGBoost without Grade |    0.7298 |     0.6678 |      0.3349 |   0.6631 | 0.4451 |
| Tuned XGBoost         |    0.7299 |     0.6678 |      0.3352 |   0.6643 | 0.4455 |

## Final Model

The final model is an XGBoost binary classification model using 162 engineered and preprocessed features. LendingClub grade and sub-grade features were excluded from the final model.

### Final Performance

- ROC-AUC: **0.7299**
- Accuracy: **0.6678**
- Precision: **0.3352**
- Recall: **0.6643**
- F1 Score: **0.4455**

## Risk Tier Analysis

| risk_tier   |   applicants |   actual_bad_loans |   average_predicted_risk |   applicant_percentage |   actual_bad_rate |
|:------------|-------------:|-------------------:|-------------------------:|-----------------------:|------------------:|
| Low Risk    |        68880 |               4252 |                 0.196912 |                26.3627 |           6.17305 |
| Medium Risk |       130373 |              23489 |                 0.448819 |                49.8982 |          18.0168  |
| High Risk   |        62025 |              24750 |                 0.703458 |                23.7391 |          39.9033  |

## Business Interpretation

The model produces a probability representing the estimated risk of a loan being classified as a bad loan. These probabilities can be grouped into risk tiers to support lending decisions and additional manual review.

The observed bad-loan rate increases substantially across the Low, Medium, and High Risk groups, indicating that the model provides useful risk segmentation.

## Important Features

- interest_rate
- term
- home_ownership_MORTGAGE
- home_ownership_RENT
- acc_open_past_24mths
- verification_status_Not Verified
- mort_acc
- open_rv_24m
- loan_to_income
- dti
- avg_cur_bal
- addr_state_OR
- all_util
- purpose_small_business
- num_actv_rev_tl
- high_dti
- addr_state_WA
- addr_state_CO
- installment_to_income
- long_employment

## Conclusion

The final XGBoost model provides a meaningful baseline for automated credit-risk assessment. Its probability outputs can be used to segment loan applications into risk tiers and support risk-aware lending decisions.
