# Micro-Lending Platform — Detailed Learning Walkthrough

This document is the "why did I do this?" companion to the README.

Use it when revising the project before interviews.

---

## 1. Start With the Business Problem

A lender wants to answer:

> Given an applicant and their loan/credit information, how likely is this loan to become a bad loan?

That immediately gives us:

### Input

Applicant + loan + credit history.

### Output

A probability:

```text
P(bad loan)
```

### Business interpretation

```text
Low probability  → lower risk
Medium probability → review
High probability → higher risk
```

---

# 2. Why MySQL First?

The requirement was an end-to-end data platform, not simply a CSV-to-model notebook.

MySQL gives us:

- structured storage
- SQL querying
- business reporting
- relational access
- a realistic data-engineering step

Python then becomes the analytics/ML layer.

Think:

```text
MySQL = data storage + SQL analytics

Python = data science + ML

FastAPI = model service

Streamlit = user interface
```

---

# 3. Why Not Load Everything Into Pandas?

The source contained about 2.26 million rows.

A naive approach:

```python
df = pd.read_sql(...)
```

caused a memory allocation error.

The important lesson is:

> Dataset size is not just about the number of rows. Number of columns and dtype conversions can make Pandas consume substantially more RAM than expected.

So the project moved toward:

```text
chunksize = 50,000
```

This means at most one chunk is being transformed in memory at a time.

---

# 4. Missing Values

Missing values are not automatically "bad data".

There are different reasons for missingness:

- field not applicable
- applicant did not provide it
- feature was only collected for certain loan types
- field was not available
- sparse secondary-applicant information

For example, hardship and settlement fields were missing for more than 98% of records.

Keeping every sparse column can introduce noise and unnecessary complexity.

---

# 5. Target Variable

Machine learning needs a target.

We converted loan status into:

```text
0 → Good Loan
1 → Bad Loan
```

Now each row becomes:

```text
X = applicant/loan information

y = risk_label
```

The learning problem becomes:

```text
X → y
```

For risk scoring:

```text
X → probability(y = 1)
```

---

# 6. Feature Engineering

Raw fields are not always the most useful representation.

Suppose:

```text
annual_income = $60,000
loan_amount = $30,000
```

The raw values tell us something.

But:

```text
loan_to_income = 30,000 / 60,000
               = 0.50
```

is much easier for a model to interpret as a financial burden.

Similarly:

```text
annual income
        ↓
monthly income
        ↓
installment / monthly income
```

creates a repayment-burden feature.

This is the difference between:

> storing data

and:

> creating predictive information.

---

# 7. Data Quality Checks

Feature engineering can create bad values.

Example:

```text
annual_income = 1
loan_amount = 35,000
```

Then:

```text
loan_to_income = 35,000
```

Mathematically valid.

Financially suspicious.

So the pipeline needed validation after feature engineering.

This is a key lesson:

> Always validate engineered features, not just raw columns.

---

# 8. Train/Test Split

We used:

```text
80% train
20% test
```

approximately:

```text
1,045,109 train
261,278 test
```

The class ratio stayed approximately:

```text
20.09% bad
```

in both sets.

Why?

Because if the training set had a very different class distribution from the test set, model evaluation could become misleading.

---

# 9. Data Leakage

A very important concept.

Suppose we calculate a median using:

```text
train + test
```

before splitting.

Then the test set has indirectly influenced the training transformation.

That is leakage.

Instead:

```text
TRAIN
  ↓
learn median
  ↓
save median
  ↓
apply median to TRAIN

TEST
  ↓
use TRAIN median
```

The same idea applies to categorical vocabularies and other preprocessing.

---

# 10. Logistic Regression

Logistic Regression was the baseline.

Why start with it?

Because a baseline tells us:

> Is a complicated model actually helping?

The scaled Logistic Regression produced:

```text
ROC-AUC = 0.6415
F1 = 0.3728
```

Not terrible, but limited.

---

# 11. Why XGBoost?

Credit risk relationships are rarely purely linear.

For example:

```text
High DTI
+
High utilization
+
Short employment
+
High interest rate
```

may create more risk together than each variable suggests individually.

Tree-based models are good at learning interactions and nonlinear boundaries.

XGBoost improved ROC-AUC to approximately:

```text
0.73
```

---

# 12. Class Imbalance

We had:

```text
~80% good
~20% bad
```

A model could get 80% accuracy by predicting every loan as good.

That would be useless.

Therefore we looked at:

```text
Precision
Recall
F1
ROC-AUC
```

especially recall for bad loans.

---

# 13. Recall

For bad loans:

```text
Recall =
caught bad loans /
all actual bad loans
```

Our final recall was about:

```text
66.43%
```

Meaning the model identified roughly two-thirds of the actual bad loans at its classification threshold.

---

# 14. Precision

Precision asks:

> Of the loans the model called bad, how many were actually bad?

Final precision:

```text
~33.5%
```

This is lower than recall.

That is a trade-off.

If the lender wants to catch more risky applicants, it will usually have to accept more false positives.

---

# 15. F1

F1 combines:

```text
precision
+
recall
```

Final:

```text
~0.446
```

F1 is useful when both false positives and false negatives matter.

---

# 16. ROC-AUC

ROC-AUC measures how well the model ranks risky loans above safer loans across thresholds.

Final:

```text
~0.73
```

Interpretation:

The model has meaningful discriminatory power, but it is not perfect.

---

# 17. Why Remove Grade?

This was one of the most interesting modeling decisions.

The original model heavily relied on:

```text
grade_A
grade_B
...
sub_grade_...
```

But grade is already an assessment assigned by the original lending process.

So we asked:

> Can the model predict risk without simply relying on an existing risk grade?

We removed:

```text
42 grade/sub-grade features
```

The resulting model still achieved:

```text
ROC-AUC ≈ 0.73
```

This made the experiment more meaningful.

---

# 18. Hyperparameter Tuning

The final model was selected after comparing six configurations.

The selected configuration:

```text
max_depth = 6
learning_rate = 0.05
min_child_weight = 5
subsample = 0.7
colsample_bytree = 0.7
```

### What these mean

`max_depth`

Controls tree complexity.

```text
higher → more complex
lower → simpler
```

`learning_rate`

Controls how strongly each tree contributes.

```text
lower → slower learning
higher → faster learning
```

`subsample`

Controls how much training data each boosting round sees.

`colsample_bytree`

Controls how many features are sampled for each tree.

`min_child_weight`

Controls how easily the tree creates new splits.

---

# 19. Probability vs Class

This is extremely important.

The model can output:

```text
0.3621
```

Instead of simply:

```text
1
```

That allows the business layer to create different thresholds.

For example:

```text
0.20 → low-ish risk
0.50 → medium
0.70 → high
```

The exact business thresholds should be chosen using validation and business cost considerations.

---

# 20. Risk Tiers

The project converted probabilities into:

```text
Low Risk
Medium Risk
High Risk
```

The final test data showed:

```text
Low Risk:
actual bad rate ≈ 6.17%

Medium Risk:
actual bad rate ≈ 18.02%

High Risk:
actual bad rate ≈ 39.90%
```

That monotonic increase is exactly what we want from a useful risk-ranking system.

---

# 21. Why FastAPI?

A model file sitting on a laptop is not an application.

FastAPI turns the model into a service:

```text
POST /predict
```

A client sends JSON.

The server:

```text
validate
→ transform
→ predict
→ return JSON
```

This separates the UI from the model.

---

# 22. Why Pydantic?

Pydantic validates the incoming request.

For example:

```python
annual_income: float
loan_amount: float
interest_rate: float
```

If the client sends malformed or missing fields, FastAPI can return:

```text
422 Unprocessable Entity
```

We actually encountered this during development.

The UI initially sent:

```json
{
    "annual_income": 65000
}
```

while the API expected:

```json
{
    "features": {...}
}
```

The schemas were then aligned.

That was a real example of why API contracts matter.

---

# 23. Why Streamlit?

The goal was to create a usable demo without building a separate React frontend.

Streamlit allowed:

```text
input form
   ↓
HTTP request
   ↓
FastAPI
   ↓
model
   ↓
risk result
```

The final UI intentionally hides most advanced model inputs so the user isn't overwhelmed.

---

# 24. Why the Dataset Is Not Loaded by Streamlit

The UI does not need the training dataset.

It only needs:

```text
model
+
prediction logic
```

Loading 1.3 million rows into the frontend would be wasteful.

This is a useful production principle:

> Training data and inference data have different responsibilities.

---

# 25. Full Mental Model

Remember the project using these five layers:

## Layer 1 — Data

```text
MySQL
CSV
SQL
Pandas
```

## Layer 2 — Intelligence

```text
Cleaning
Feature Engineering
EDA
```

## Layer 3 — Machine Learning

```text
Train/Test
Preprocessing
Logistic Regression
XGBoost
Tuning
Risk Probability
```

## Layer 4 — Backend

```text
FastAPI
Pydantic
Prediction endpoint
```

## Layer 5 — Frontend

```text
Streamlit
Applicant inputs
Risk score
Risk tier
```

If you remember these five layers, you can reconstruct the whole project.

---

# 26. Interview Questions You Should Practice

### Data

1. Why did you use MySQL?
2. Why did the dataset shrink from 2.26M to 1.3M?
3. How did you handle missing values?
4. Why did you process the dataset in chunks?
5. What caused your memory error?

### Feature Engineering

6. Why did you create loan-to-income?
7. What does installment-to-income represent?
8. Why are extreme ratios dangerous?
9. How did you handle invalid income values?

### ML

10. Why Logistic Regression first?
11. Why XGBoost?
12. Why was accuracy insufficient?
13. Explain precision vs recall.
14. What does ROC-AUC mean?
15. Why is class imbalance a problem?
16. What is `scale_pos_weight`?
17. Why remove grade/sub-grade?
18. How did you tune XGBoost?
19. What are the most important features?

### Deployment

20. Why FastAPI?
21. What is Pydantic?
22. Why did FastAPI return 422?
23. Why Streamlit?
24. How does `/predict` work?
25. How would you deploy this system?

---

# 27. The One-Minute Architecture Explanation

If someone asks you to draw the architecture on a whiteboard:

```text
                  ┌──────────────┐
                  │  Loan Data   │
                  │     CSV      │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    MySQL     │
                  └──────┬───────┘
                         │
                  SQL Analysis
                         │
                         ▼
               ┌───────────────────┐
               │ Python Processing  │
               │ Cleaning + FE +   │
               │ EDA               │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │ Train/Test Split  │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │ XGBoost Training  │
               │ + Tuning          │
               └─────────┬─────────┘
                         │
                         ▼
                xgboost_final.json
                         │
                         ▼
                 ┌─────────────┐
                 │   FastAPI   │
                 │  /predict   │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Streamlit  │
                 │     UI      │
                 └─────────────┘
```

---

# 28. Final Mental Summary

If you forget everything else, remember:

```text
I had a large loan dataset.

↓
I stored and analyzed it in MySQL.

↓
I couldn't load everything into RAM,
so I built chunk-based processing.

↓
I cleaned the data and engineered
financial-risk features.

↓
I created a good/bad loan target.

↓
I split the data without leaking
test information.

↓
I built a Logistic Regression baseline.

↓
XGBoost performed better.

↓
I tested removing grade/sub-grade
because I didn't want to blindly reproduce
an existing credit grade.

↓
I tuned XGBoost.

↓
The final model achieved ~0.73 ROC-AUC.

↓
I converted probabilities into
Low / Medium / High risk tiers.

↓
I served the model using FastAPI.

↓
I built a Streamlit frontend.

↓
A user can now enter an application
and receive a risk probability.
```

That is the entire project.
