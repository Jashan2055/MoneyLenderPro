# System Architecture

## High-level flow

```text
                         DATA LAYER
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  loan.csv ───────────────► MySQL `loans` table              │
│                                  │                          │
│                                  ▼                          │
│                         SQL Business Insights               │
│                                                             │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                         PROCESSING LAYER
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Chunked reading                                             │
│       ↓                                                     │
│  Cleaning                                                    │
│       ↓                                                     │
│  Missing-value handling                                     │
│       ↓                                                     │
│  Target creation                                             │
│       ↓                                                     │
│  Feature engineering                                         │
│       ↓                                                     │
│  Validation + diagnostics                                    │
│                                                             │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                           ML LAYER
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Train / Test Split                                         │
│       ↓                                                     │
│  Training-only preprocessing                                │
│       ↓                                                     │
│  Logistic Regression baseline                               │
│       ↓                                                     │
│  XGBoost baseline                                            │
│       ↓                                                     │
│  No-grade experiment                                         │
│       ↓                                                     │
│  Hyperparameter tuning                                      │
│       ↓                                                     │
│  Final XGBoost                                               │
│                                                             │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                         SERVING LAYER
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  xgboost_final.json                                         │
│          │                                                  │
│          ▼                                                  │
│      FastAPI                                                  │
│       /predict                                                │
│          │                                                  │
│          ▼                                                  │
│   prediction.py                                              │
│          │                                                  │
│          ▼                                                  │
│   risk probability                                           │
│          │                                                  │
│          ▼                                                  │
│      risk tier                                               │
│                                                             │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                           UI LAYER
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                      Streamlit UI                           │
│                                                             │
│   Applicant inputs → POST /predict → Risk result            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key design principle

The training pipeline and inference pipeline have different responsibilities.

### Training

```text
large dataset
→ cleaning
→ feature engineering
→ preprocessing
→ model training
→ model artifact
```

### Inference

```text
one applicant
→ same feature logic
→ same preprocessing assumptions
→ trained model
→ prediction
```

The model should never need the entire training dataset just to make one prediction.
