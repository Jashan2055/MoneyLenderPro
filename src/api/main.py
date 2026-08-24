from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.prediction import predict_risk


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Micro-Lending Credit Risk API",
    description="Loan default risk prediction API",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class LoanApplication(BaseModel):

    annual_income: float
    loan_amount: float

    funded_amount: float | None = None
    investor_funds: float | None = None

    term: float = 36
    interest_rate: float
    installment: float

    dti: float | None = None
    emp_length: float | None = None

    home_ownership: str = "RENT"
    verification_status: str = "Not Verified"
    addr_state: str = "CA"

    purpose: str = "debt_consolidation"
    pymnt_plan: str = "n"

    revol_bal: float | None = None
    revol_util: float | None = None
    total_rev_hi_lim: float | None = None

    open_acc: float | None = None
    total_acc: float | None = None

    delinq_2yrs: float | None = None
    pub_rec: float | None = None

    tot_cur_bal: float | None = None
    bc_util: float | None = None

    credit_history_years: float | None = None


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "xgboost_final"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    try:

        data = application.model_dump()

        result = predict_risk(data)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )