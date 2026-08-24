from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "xgboost_final.json"
METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "preprocessing_metadata.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading XGBoost model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD PREPROCESSING METADATA
# ============================================================

print("Loading preprocessing metadata...")

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print("Preprocessing metadata loaded.")


# ============================================================
# NUMERIC FEATURES USED BY THE MODEL
# ============================================================

NUMERIC_FEATURES = [
    "emp_length",
    "annual_income",
    "loan_amount",
    "funded_amount",
    "investor_funds",
    "term",
    "interest_rate",
    "installment",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "total_rev_hi_lim",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",

    # engineered features
    "income_unreliable",
    "dti_unavailable",
    "loan_to_income",
    "monthly_income",
    "installment_to_income",
    "credit_history_years",
    "active_account_ratio",
    "funded_amount_ratio",
    "revolving_balance_ratio",
    "delinquency_ratio",
    "public_record_ratio",
    "balance_to_income",
    "utilization_difference",
    "high_revolving_utilization",
    "high_dti",
    "long_employment",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "addr_state",
    "pymnt_plan",
    "purpose",
]


# ============================================================
# MEDIANS
# ============================================================

MEDIANS = {
    "emp_length": 6.0,
    "annual_income": 65000.0,
    "loan_amount": 12000.0,
    "funded_amount": 12000.0,
    "investor_funds": 12000.0,
    "term": 36.0,
    "interest_rate": 12.74,
    "installment": 375.43,
    "dti": 17.6,
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 0.0,
    "mths_since_last_delinq": 31.0,
    "mths_since_last_record": 71.0,
    "open_acc": 11.0,
    "pub_rec": 0.0,
    "revol_bal": 11130.0,
    "revol_util": 52.3,
    "total_acc": 23.0,
    "acc_now_delinq": 0.0,
    "tot_coll_amt": 0.0,
    "tot_cur_bal": 80452.0,
    "open_acc_6m": 1.0,
    "open_act_il": 2.0,
    "open_il_12m": 1.0,
    "open_il_24m": 1.0,
    "mths_since_rcnt_il": 12.0,
    "total_bal_il": 24208.0,
    "il_util": 75.0,
    "open_rv_12m": 1.0,
    "open_rv_24m": 2.0,
    "max_bal_bc": 4194.0,
    "all_util": 60.0,
    "total_rev_hi_lim": 24000.0,
    "inq_fi": 1.0,
    "total_cu_tl": 0.0,
    "inq_last_12m": 2.0,
    "acc_open_past_24mths": 4.0,
    "avg_cur_bal": 7426.0,
    "bc_open_to_buy": 4656.0,
    "bc_util": 63.4,
    "mo_sin_old_il_acct": 129.0,
    "mo_sin_old_rev_tl_op": 164.0,
    "mo_sin_rcnt_rev_tl_op": 8.0,
    "mo_sin_rcnt_tl": 5.0,
    "mort_acc": 1.0,
    "mths_since_recent_bc": 13.0,
    "mths_since_recent_bc_dlq": 38.0,
    "mths_since_recent_inq": 5.0,
    "mths_since_recent_revol_delinq": 33.0,
    "num_accts_ever_120_pd": 0.0,
    "num_actv_bc_tl": 3.0,
    "num_actv_rev_tl": 5.0,
    "num_bc_sats": 4.0,
    "num_bc_tl": 7.0,
    "num_il_tl": 7.0,
    "num_op_rev_tl": 7.0,
    "num_rev_accts": 13.0,
    "num_rev_tl_bal_gt_0": 5.0,
    "num_sats": 11.0,
    "num_tl_120dpd_2m": 0.0,
    "num_tl_30dpd": 0.0,
    "num_tl_90g_dpd_24m": 0.0,
    "num_tl_op_past_12m": 2.0,
    "pct_tl_nvr_dlq": 98.0,
    "percent_bc_gt_75": 44.4,
    "pub_rec_bankruptcies": 0.0,
    "tax_liens": 0.0,
    "tot_hi_cred_lim": 112461.0,
    "total_bal_ex_mort": 37292.0,
    "total_bc_limit": 15000.0,
    "total_il_high_credit_limit": 31632.0,

    "income_unreliable": 0.0,
    "dti_unavailable": 0.0,
    "loan_to_income": 0.2,
    "monthly_income": 5416.666666666667,
    "installment_to_income": 0.0722715789473684,
    "credit_history_years": 14.74880219028063,
    "active_account_ratio": 0.4814814814814814,
    "funded_amount_ratio": 1.0,
    "revolving_balance_ratio": 0.526,
    "delinquency_ratio": 0.0,
    "public_record_ratio": 0.0,
    "balance_to_income": 1.2447087378640778,
    "utilization_difference": -4.9,
    "high_revolving_utilization": 0.0,
    "high_dti": 0.0,
    "long_employment": 1.0,
}


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(a, b):
    if b is None or pd.isna(b) or b == 0:
        return np.nan

    return a / b


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(data: dict) -> dict:

    row = {}

    # --------------------------------------------------------
    # Copy known numeric values
    # --------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        if feature in data and data[feature] is not None:
            row[feature] = data[feature]

        else:
            row[feature] = MEDIANS.get(feature, 0.0)

    # --------------------------------------------------------
    # Core values
    # --------------------------------------------------------

    income = data.get(
        "annual_income",
        MEDIANS["annual_income"]
    )

    loan = data.get(
        "loan_amount",
        MEDIANS["loan_amount"]
    )

    installment = data.get(
        "installment",
        MEDIANS["installment"]
    )

    funded = data.get(
        "funded_amount",
        loan
    )

    revol_bal = data.get(
        "revol_bal",
        MEDIANS["revol_bal"]
    )

    total_rev_hi_lim = data.get(
        "total_rev_hi_lim",
        MEDIANS["total_rev_hi_lim"]
    )

    open_acc = data.get(
        "open_acc",
        MEDIANS["open_acc"]
    )

    total_acc = data.get(
        "total_acc",
        MEDIANS["total_acc"]
    )

    delinq_2yrs = data.get(
        "delinq_2yrs",
        MEDIANS["delinq_2yrs"]
    )

    pub_rec = data.get(
        "pub_rec",
        MEDIANS["pub_rec"]
    )

    # --------------------------------------------------------
    # Engineered features
    # --------------------------------------------------------

    row["income_unreliable"] = int(
        income is None or
        pd.isna(income) or
        income <= 1000
    )

    row["dti_unavailable"] = int(
        data.get("dti") is None or
        pd.isna(data.get("dti"))
    )

    row["monthly_income"] = (
        income / 12
        if income and income > 0
        else MEDIANS["monthly_income"]
    )

    row["loan_to_income"] = safe_divide(
        loan,
        income
    )

    row["installment_to_income"] = safe_divide(
        installment,
        row["monthly_income"]
    )

    # --------------------------------------------------------
    # Credit history
    # --------------------------------------------------------

    credit_history_years = data.get(
        "credit_history_years"
    )

    if credit_history_years is not None:
        row["credit_history_years"] = credit_history_years

    # --------------------------------------------------------
    # Account ratios
    # --------------------------------------------------------

    row["active_account_ratio"] = safe_divide(
        open_acc,
        total_acc
    )

    row["funded_amount_ratio"] = safe_divide(
        funded,
        loan
    )

    row["revolving_balance_ratio"] = safe_divide(
        revol_bal,
        total_rev_hi_lim
    )

    row["delinquency_ratio"] = safe_divide(
        delinq_2yrs,
        total_acc
    )

    row["public_record_ratio"] = safe_divide(
        pub_rec,
        total_acc
    )

    row["balance_to_income"] = safe_divide(
        data.get(
            "tot_cur_bal",
            MEDIANS["tot_cur_bal"]
        ),
        income
    )

    row["utilization_difference"] = (
        data.get(
            "revol_util",
            MEDIANS["revol_util"]
        )
        -
        data.get(
            "bc_util",
            MEDIANS["bc_util"]
        )
    )

    row["high_revolving_utilization"] = int(
        data.get(
            "revol_util",
            MEDIANS["revol_util"]
        ) > 80
    )

    row["high_dti"] = int(
        data.get(
            "dti",
            MEDIANS["dti"]
        ) > 40
    )

    row["long_employment"] = int(
        data.get(
            "emp_length",
            MEDIANS["emp_length"]
        ) >= 5
    )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        value = row.get(feature)

        if value is None or pd.isna(value):
            row[feature] = MEDIANS.get(
                feature,
                0.0
            )

        elif not np.isfinite(float(value)):
            row[feature] = MEDIANS.get(
                feature,
                0.0
            )

    return row


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(data: dict) -> pd.DataFrame:

    engineered = engineer_features(data)

    df = pd.DataFrame(
        [engineered]
    )

    # --------------------------------------------------------
    # Categorical values
    # --------------------------------------------------------

    categorical_values = {
        "home_ownership": data.get(
            "home_ownership",
            "RENT"
        ),

        "verification_status": data.get(
            "verification_status",
            "Not Verified"
        ),

        "addr_state": data.get(
            "addr_state",
            "CA"
        ),

        "pymnt_plan": data.get(
            "pymnt_plan",
            "n"
        ),

        "purpose": data.get(
            "purpose",
            "debt_consolidation"
        ),
    }

    # --------------------------------------------------------
    # One-hot encode using training-style naming
    # --------------------------------------------------------

    for column, value in categorical_values.items():

        df[column] = value

    encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_FEATURES,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Remove grade/sub-grade if supplied
    # --------------------------------------------------------

    grade_columns = [
        c for c in encoded.columns
        if c.startswith("grade_")
        or c.startswith("sub_grade_")
    ]

    if grade_columns:
        encoded = encoded.drop(
            columns=grade_columns
        )

    # --------------------------------------------------------
    # Expected categorical columns
    # --------------------------------------------------------

    expected_categories = {
        "home_ownership": [
            "ANY",
            "MORTGAGE",
            "NONE",
            "OTHER",
            "OWN",
            "RENT",
        ],

        "verification_status": [
            "Not Verified",
            "Source Verified",
            "Verified",
        ],

        "pymnt_plan": [
            "n",
        ],

        "purpose": [
            "car",
            "credit_card",
            "debt_consolidation",
            "educational",
            "home_improvement",
            "house",
            "major_purchase",
            "medical",
            "moving",
            "other",
            "renewable_energy",
            "small_business",
            "vacation",
            "wedding",
        ],
    }

    # --------------------------------------------------------
    # Add missing categorical columns
    # --------------------------------------------------------

    for column, categories in expected_categories.items():

        for category in categories:

            feature_name = f"{column}_{category}"

            if feature_name not in encoded.columns:
                encoded[feature_name] = 0.0

    # addr_state
    states = [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT",
        "DC", "DE", "FL", "GA", "HI", "IA", "ID",
        "IL", "IN", "KS", "KY", "LA", "MA", "MD",
        "ME", "MI", "MN", "MO", "MS", "MT", "NC",
        "ND", "NE", "NH", "NJ", "NM", "NV", "NY",
        "OH", "OK", "OR", "PA", "RI", "SC", "SD",
        "TN", "TX", "UT", "VA", "VT", "WA", "WI",
        "WV", "WY"
    ]

    for state in states:

        feature_name = f"addr_state_{state}"

        if feature_name not in encoded.columns:
            encoded[feature_name] = 0.0

    # --------------------------------------------------------
    # Keep ONLY numeric + categorical encoded features
    # --------------------------------------------------------

    feature_columns = (
        NUMERIC_FEATURES
        + [
            f"home_ownership_{x}"
            for x in expected_categories["home_ownership"]
        ]
        + [
            f"verification_status_{x}"
            for x in expected_categories["verification_status"]
        ]
        + [
            f"addr_state_{x}"
            for x in states
        ]
        + [
            f"pymnt_plan_{x}"
            for x in expected_categories["pymnt_plan"]
        ]
        + [
            f"purpose_{x}"
            for x in expected_categories["purpose"]
        ]
    )

    # --------------------------------------------------------
    # Guarantee exact feature count/order
    # --------------------------------------------------------

    for column in feature_columns:

        if column not in encoded.columns:
            encoded[column] = 0.0

    encoded = encoded[
        feature_columns
    ]

    encoded = encoded.astype(
        np.float32
    )

    return encoded


# ============================================================
# PREDICTION
# ============================================================

def predict_risk(data: dict):

    X = create_model_input(
        data
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    # --------------------------------------------------------
    # Risk tier
    # --------------------------------------------------------

    if probability < 0.30:
        risk_tier = "Low Risk"

    elif probability < 0.60:
        risk_tier = "Medium Risk"

    else:
        risk_tier = "High Risk"

    return {
        "risk_probability": round(
            probability,
            4
        ),

        "risk_percentage": round(
            probability * 100,
            2
        ),

        "risk_tier": risk_tier,

        "model": "Tuned XGBoost",

        "features_used": X.shape[1],
    }