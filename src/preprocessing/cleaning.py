import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# --------------------------------------------------
# Project setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from database.connection import get_engine

engine = get_engine()


# --------------------------------------------------
# Loan outcome definition
# --------------------------------------------------

GOOD_STATUSES = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid"
]

BAD_STATUSES = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off"
]


# --------------------------------------------------
# Read customer + loan data from MySQL
# --------------------------------------------------

print("Reading data from MySQL...")

QUERY = """
SELECT
    l.loan_id,
    l.customer_id,

    -- Customer information
    c.emp_title,
    c.emp_length,
    c.home_ownership,
    c.annual_income,
    c.verification_status,
    c.addr_state,

    -- Loan information
    l.loan_amount,
    l.funded_amount,
    l.investor_funds,
    l.term,
    l.interest_rate,
    l.installment,
    l.grade,
    l.sub_grade,
    l.issue_d,
    l.pymnt_plan,
    l.purpose,
    l.title,
    l.zip_code,
    l.dti,

    -- Credit history
    l.delinq_2yrs,
    l.earliest_cr_line,
    l.inq_last_6mths,
    l.mths_since_last_delinq,
    l.mths_since_last_record,
    l.open_acc,
    l.pub_rec,
    l.revol_bal,
    l.revol_util,
    l.total_acc,

    -- Credit behaviour
    l.acc_now_delinq,
    l.tot_coll_amt,
    l.tot_cur_bal,

    -- Account information
    l.open_acc_6m,
    l.open_act_il,
    l.open_il_12m,
    l.open_il_24m,
    l.mths_since_rcnt_il,
    l.total_bal_il,
    l.il_util,
    l.open_rv_12m,
    l.open_rv_24m,
    l.max_bal_bc,
    l.all_util,
    l.total_rev_hi_lim,
    l.inq_fi,
    l.total_cu_tl,
    l.inq_last_12m,

    -- Account age / utilization
    l.acc_open_past_24mths,
    l.avg_cur_bal,
    l.bc_open_to_buy,
    l.bc_util,

    -- Credit history depth
    l.mo_sin_old_il_acct,
    l.mo_sin_old_rev_tl_op,
    l.mo_sin_rcnt_rev_tl_op,
    l.mo_sin_rcnt_tl,
    l.mort_acc,
    l.mths_since_recent_bc,
    l.mths_since_recent_bc_dlq,
    l.mths_since_recent_inq,
    l.mths_since_recent_revol_delinq,

    -- Account counts
    l.num_accts_ever_120_pd,
    l.num_actv_bc_tl,
    l.num_actv_rev_tl,
    l.num_bc_sats,
    l.num_bc_tl,
    l.num_il_tl,
    l.num_op_rev_tl,
    l.num_rev_accts,
    l.num_rev_tl_bal_gt_0,
    l.num_sats,

    -- Delinquency
    l.num_tl_120dpd_2m,
    l.num_tl_30dpd,
    l.num_tl_90g_dpd_24m,
    l.num_tl_op_past_12m,

    -- Credit quality
    l.pct_tl_nvr_dlq,
    l.percent_bc_gt_75,
    l.pub_rec_bankruptcies,
    l.tax_liens,

    -- Credit limits
    l.tot_hi_cred_lim,
    l.total_bal_ex_mort,
    l.total_bc_limit,
    l.total_il_high_credit_limit,

    -- Target
    l.loan_status

FROM loans l
JOIN customers c
    ON l.customer_id = c.customer_id

WHERE l.loan_status IN (
    'Fully Paid',
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Fully Paid',
    'Does not meet the credit policy. Status:Charged Off'
)
"""


print("Executing query...")

print("Executing query in chunks...")

# --------------------------------------------------
# Output
# --------------------------------------------------

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "model_dataset.csv"


# Remove previous output if it exists
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()


# --------------------------------------------------
# Process data in chunks
# --------------------------------------------------

CHUNK_SIZE = 50_000

first_chunk = True
total_processed = 0

for chunk_number, df in enumerate(
    pd.read_sql_query(
        QUERY,
        engine,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    print(
        f"\nProcessing chunk {chunk_number} "
        f"({len(df):,} rows)..."
    )

    # --------------------------------------------------
    # Create target
    # --------------------------------------------------

    df["risk_label"] = df["loan_status"].map(
        lambda status: (
            0 if status in GOOD_STATUSES else 1
        )
    )
    if chunk_number == 1:
    print("\nLoan status distribution in model dataset:")
    print(df["loan_status"].value_counts())

    # --------------------------------------------------
    # Remove loan_status
    # --------------------------------------------------

    df.drop(
        columns=["loan_status"],
        inplace=True
    )

    # --------------------------------------------------
    # String cleaning
    # --------------------------------------------------

    string_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in string_columns:
        df[column] = df[column].str.strip()

    # --------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------

    numeric_columns = [
        "annual_income",
        "loan_amount",
        "funded_amount",
        "investor_funds",
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
        "total_il_high_credit_limit"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------
    # Employment length
    # --------------------------------------------------

    df["emp_length"] = (
        df["emp_length"]
        .replace({
            "< 1 year": "0",
            "10+ years": "10"
        })
        .str.extract(
            r"(\d+)",
            expand=False
        )
    )

    df["emp_length"] = pd.to_numeric(
        df["emp_length"],
        errors="coerce"
    )

    # --------------------------------------------------
    # Term
    # --------------------------------------------------

    df["term"] = (
        df["term"]
        .str.extract(
            r"(\d+)",
            expand=False
        )
    )

    df["term"] = pd.to_numeric(
        df["term"],
        errors="coerce"
    )

    # --------------------------------------------------
    # Dates
    # --------------------------------------------------

    df["issue_d"] = pd.to_datetime(
        df["issue_d"],
        format="%b-%Y",
        errors="coerce"
    )

    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"],
        format="%b-%Y",
        errors="coerce"
    )

    # --------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------

    categorical_columns = [
        "emp_title",
        "home_ownership",
        "verification_status",
        "addr_state",
        "grade",
        "sub_grade",
        "purpose",
        "title",
        "pymnt_plan",
        "application_type",
        "disbursement_method"
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    # --------------------------------------------------
    # Remove database identifiers
    # --------------------------------------------------

    df.drop(
        columns=[
            "loan_id",
            "customer_id"
        ],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------
    # Write chunk
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    total_processed += len(df)

    print(
        f"Chunk {chunk_number} completed. "
        f"Total processed: {total_processed:,}"
    )

    # Explicitly release memory
    del df

 print("\n========================================")
print("CLEANING COMPLETED")
print("========================================")

print(f"Total rows processed: {total_processed:,}")
print(f"Output file: {OUTPUT_FILE}")

# --------------------------------------------------
# Dataset summary
# --------------------------------------------------

TOTAL_LOANS = 2_260_668
MODEL_LOANS = total_processed
EXCLUDED_LOANS = TOTAL_LOANS - MODEL_LOANS

print("\n========================================")
print("DATASET SUMMARY")
print("========================================")

print(f"Total loans in MySQL:        {TOTAL_LOANS:,}")
print(f"Definitive outcome loans:    {MODEL_LOANS:,}")
print(f"Unfinished outcome loans:    {EXCLUDED_LOANS:,}")

print("\nUnfinished loans are NOT deleted.")
print("They remain available in MySQL for EDA and portfolio analysis.")