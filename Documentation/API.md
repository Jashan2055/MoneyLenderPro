# API Documentation

## Start the API

From the project root:

```bash
python -m uvicorn src.api.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## GET /health

Checks whether the API is running.

Example response:

```json
{
  "status": "healthy",
  "model": "xgboost_final"
}
```

---

## POST /predict

Accepts applicant and loan information and returns a risk prediction.

Example request:

```json
{
  "annual_income": 65000,
  "loan_amount": 12000,
  "funded_amount": 12000,
  "investor_funds": 12000,
  "term": 36,
  "interest_rate": 12.74,
  "installment": 375.43,
  "dti": 17.6,
  "emp_length": 6,
  "home_ownership": "RENT",
  "verification_status": "Not Verified",
  "addr_state": "CA",
  "purpose": "debt_consolidation",
  "pymnt_plan": "n",
  "revol_bal": 11130,
  "revol_util": 52.3,
  "total_rev_hi_lim": 24000,
  "open_acc": 11,
  "total_acc": 23,
  "delinq_2yrs": 0,
  "pub_rec": 0,
  "tot_cur_bal": 80452,
  "bc_util": 63.4,
  "credit_history_years": 14.75
}
```

The API validates the request using Pydantic.

A malformed request returns HTTP 422.

---

## Response

The prediction service returns the risk probability and risk tier used by the Streamlit application.

Example conceptually:

```json
{
  "risk_probability": 0.3621,
  "risk_percentage": 36.21,
  "risk_tier": "Medium Risk"
}
```

---

## Client architecture

```text
Streamlit
    │
    │ HTTP POST
    ▼
FastAPI /predict
    │
    ▼
prediction.py
    │
    ▼
XGBoost
    │
    ▼
JSON response
```
