# # import pandas as pd

# # file_path = "/data/raw/loan.csv"

# # df = pd.read_csv(file_path)

# # print("Rows:", df.shape[0])
# # print("Columns:", df.shape[1])

# # print(df.head())
# # print(df.info())


# from pathlib import Path
# import pandas as pd

# # Project root directory
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Path to CSV file
# file_path = BASE_DIR / "data" / "raw" / "loan.csv"

# print("Looking for CSV at:", file_path)

# df = pd.read_csv(file_path)

# print("Data loaded successfully!")

# print("Shape:", df.shape)

# print("\nColumns:")
# print(df.columns.tolist())

# print("\nFirst 5 rows:")
# print(df.head())

# print("\nData types:")
# print(df.dtypes)

# print("\nMissing values:")
# print(df.isnull().sum().sort_values(ascending=False).head(20))



# print("\nNon-null counts for potential customer identifiers:")
# print(df[[
#     "id",
#     "member_id",
#     "emp_title",
#     "emp_length",
#     "home_ownership",
#     "annual_inc",
#     "verification_status",
#     "addr_state"
# ]].notna().sum())

# print("\nUnique values:")
# print(df[[
#     "emp_title",
#     "emp_length",
#     "home_ownership",
#     "annual_inc",
#     "verification_status",
#     "addr_state"
# ]].nunique())

# print("\nIssue dates:")
# print(df["issue_d"].head(10))

# print("\nLoan status:")
# print(df["loan_status"].value_counts(dropna=False).head(20))

# # ============================================================
# # CREATE CUSTOMER DATASET
# # ============================================================

# print("\nCreating customers dataset...")

# # Since member_id is completely NULL in this dataset,
# # we create our own customer_id.
# df["customer_id"] = range(1, len(df) + 1)

# # Each row represents one loan/application in this dataset.
# # Therefore, we assign one customer_id to each source row.

# customer_columns = [
#     "customer_id",
#     "emp_title",
#     "emp_length",
#     "home_ownership",
#     "annual_inc",
#     "verification_status",
#     "addr_state"
# ]

# customers = df[customer_columns].copy()

# customers.rename(
#     columns={
#         "annual_inc": "annual_income"
#     },
#     inplace=True
# )

# print("Customers DataFrame created.")
# print("Customers shape:", customers.shape)
# print(customers.head())


# # ============================================================
# # CREATE LOANS DATASET
# # ============================================================

# print("\nCreating loans dataset...")

# # Generate our own loan ID
# df["loan_id"] = range(1, len(df) + 1)

# loan_columns = [
#     "loan_id",
#     "customer_id",

#     # Loan information
#     "loan_amnt",
#     "funded_amnt",
#     "funded_amnt_inv",
#     "term",
#     "int_rate",
#     "installment",
#     "grade",
#     "sub_grade",

#     # Application information
#     "issue_d",
#     "loan_status",
#     "pymnt_plan",
#     "purpose",
#     "title",
#     "zip_code",
#     "dti",

#     # Credit information
#     "delinq_2yrs",
#     "earliest_cr_line",
#     "inq_last_6mths",
#     "mths_since_last_delinq",
#     "mths_since_last_record",
#     "open_acc",
#     "pub_rec",
#     "revol_bal",
#     "revol_util",
#     "total_acc",

#     # Loan repayment information
#     "initial_list_status",
#     "out_prncp",
#     "out_prncp_inv",
#     "total_pymnt",
#     "total_pymnt_inv",
#     "total_rec_prncp",
#     "total_rec_int",
#     "total_rec_late_fee",
#     "recoveries",
#     "collection_recovery_fee",
#     "last_pymnt_d",
#     "last_pymnt_amnt",
#     "next_pymnt_d",
#     "last_credit_pull_d",

#     # Additional credit information
#     "collections_12_mths_ex_med",
#     "mths_since_last_major_derog",
#     "policy_code",
#     "application_type",
#     "annual_inc_joint",
#     "dti_joint",
#     "verification_status_joint",
#     "acc_now_delinq",
#     "tot_coll_amt",
#     "tot_cur_bal",

#     # Account activity
#     "open_acc_6m",
#     "open_act_il",
#     "open_il_12m",
#     "open_il_24m",
#     "mths_since_rcnt_il",
#     "total_bal_il",
#     "il_util",
#     "open_rv_12m",
#     "open_rv_24m",
#     "max_bal_bc",
#     "all_util",
#     "total_rev_hi_lim",
#     "inq_fi",
#     "total_cu_tl",
#     "inq_last_12m",
#     "acc_open_past_24mths",
#     "avg_cur_bal",
#     "bc_open_to_buy",
#     "bc_util",

#     # Credit history
#     "chargeoff_within_12_mths",
#     "delinq_amnt",
#     "mo_sin_old_il_acct",
#     "mo_sin_old_rev_tl_op",
#     "mo_sin_rcnt_rev_tl_op",
#     "mo_sin_rcnt_tl",
#     "mort_acc",
#     "mths_since_recent_bc",
#     "mths_since_recent_bc_dlq",
#     "mths_since_recent_inq",
#     "mths_since_recent_revol_delinq",
#     "num_accts_ever_120_pd",
#     "num_actv_bc_tl",
#     "num_actv_rev_tl",
#     "num_bc_sats",
#     "num_bc_tl",
#     "num_il_tl",
#     "num_op_rev_tl",
#     "num_rev_accts",
#     "num_rev_tl_bal_gt_0",
#     "num_sats",
#     "num_tl_120dpd_2m",
#     "num_tl_30dpd",
#     "num_tl_90g_dpd_24m",
#     "num_tl_op_past_12m",
#     "pct_tl_nvr_dlq",
#     "percent_bc_gt_75",
#     "pub_rec_bankruptcies",
#     "tax_liens",
#     "tot_hi_cred_lim",
#     "total_bal_ex_mort",
#     "total_bc_limit",
#     "total_il_high_credit_limit",

#     # Joint applicant information
#     "revol_bal_joint",
#     "sec_app_earliest_cr_line",
#     "sec_app_inq_last_6mths",
#     "sec_app_mort_acc",
#     "sec_app_open_acc",
#     "sec_app_revol_util",
#     "sec_app_open_act_il",
#     "sec_app_num_rev_accts",
#     "sec_app_chargeoff_within_12_mths",
#     "sec_app_collections_12_mths_ex_med",
#     "sec_app_mths_since_last_major_derog",

#     # Hardship information
#     "hardship_flag",
#     "hardship_type",
#     "hardship_reason",
#     "hardship_status",
#     "deferral_term",
#     "hardship_amount",
#     "hardship_start_date",
#     "hardship_end_date",
#     "payment_plan_start_date",
#     "hardship_length",
#     "hardship_dpd",
#     "hardship_loan_status",
#     "orig_projected_additional_accrued_interest",
#     "hardship_payoff_balance_amount",
#     "hardship_last_payment_amount",

#     # Settlement information
#     "disbursement_method",
#     "debt_settlement_flag",
#     "debt_settlement_flag_date",
#     "settlement_status",
#     "settlement_date",
#     "settlement_amount",
#     "settlement_percentage",
#     "settlement_term"
# ]

# loans = df[loan_columns].copy()

# loans.rename(
#     columns={
#         "loan_amnt": "loan_amount",
#         "funded_amnt": "funded_amount",
#         "funded_amnt_inv": "investor_funds",
#         "int_rate": "interest_rate",
#         "annual_inc_joint": "joint_annual_income"
#     },
#     inplace=True
# )

# print("Loans DataFrame created.")
# print("Loans shape:", loans.shape)
# print(loans.head())

# # ============================================================
# # MYSQL CONNECTION
# # ============================================================
# import os
# from dotenv import load_dotenv
# from sqlalchemy import create_engine

# load_dotenv()

# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")


# connection_url = URL.create(
#     "mysql+pymysql",
#     username=os.getenv("DB_USER"),
#     password=os.getenv("DB_PASSWORD"),
#     host=os.getenv("DB_HOST"),
#     port=int(os.getenv("DB_PORT", 3306)),
#     database=os.getenv("DB_NAME")
# )

# engine = create_engine(
#     connection_string,
#     pool_pre_ping=True
# )

# print("\nMySQL engine created successfully.")

# # ============================================================
# # LOAD CUSTOMERS INTO MYSQL
# # ============================================================

# print("\nLoading customers into MySQL...")

# customers.to_sql(
#     name="customers",
#     con=engine,
#     if_exists="append",
#     index=False,
#     chunksize=5000
# )

# print(f"Customers loaded successfully: {len(customers):,}")


# # ============================================================
# # LOAD LOANS INTO MYSQL
# # ============================================================

# print("\nLoading loans into MySQL...")

# loans.to_sql(
#     name="loans",
#     con=engine,
#     if_exists="append",
#     index=False,
#     chunksize=5000
# )

# print(f"Loans loaded successfully: {len(loans):,}")


from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "raw" / "loan.csv"

print("Looking for CSV at:", FILE_PATH)


# ============================================================
# 2. MYSQL CONFIGURATION
# ============================================================

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")


connection_url = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME
)


engine = create_engine(
    connection_url,
    pool_pre_ping=True
)

print("MySQL connection created successfully.")

# ============================================================
# 3. CUSTOMER COLUMNS
# ============================================================

customer_columns = [
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "addr_state"
]


# ============================================================
# 4. LOAN COLUMNS
# ============================================================

loan_columns = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "issue_d",
    "loan_status",
    "pymnt_plan",
    "purpose",
    "title",
    "zip_code",
    "dti",

    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",

    "initial_list_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",

    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "policy_code",
    "application_type",
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
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

    "chargeoff_within_12_mths",
    "delinq_amnt",
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

    "revol_bal_joint",

    "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",

    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",

    "disbursement_method",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term"
]


# ============================================================
# 5. LOAD CUSTOMERS IN CHUNKS
# ============================================================

print("\n========================================")
print("LOADING CUSTOMERS")
print("========================================")

customer_count = 0

customer_reader = pd.read_csv(
    FILE_PATH,
    usecols=customer_columns,
    chunksize=5000,
    low_memory=False
)

for chunk_number, customers in enumerate(customer_reader, start=1):

    # Generate customer IDs for this chunk.
    start_id = customer_count + 1
    end_id = start_id + len(customers)

    customers.insert(
        0,
        "customer_id",
        range(start_id, end_id)
    )

    customers.rename(
        columns={
            "annual_inc": "annual_income"
        },
        inplace=True
    )

    customers.to_sql(
        name="customers",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000
    )

    customer_count += len(customers)

    if chunk_number % 10 == 0:
        print(
            f"Customers loaded: {customer_count:,}"
        )


print(
    f"Finished loading customers: "
    f"{customer_count:,}"
)


# ============================================================
# 6. LOAD LOANS IN CHUNKS
# ============================================================

print("\n========================================")
print("LOADING LOANS")
print("========================================")

loan_count = 0

loan_reader = pd.read_csv(
    FILE_PATH,
    usecols=loan_columns,
    chunksize=5000,
    low_memory=False
)

for chunk_number, chunk in enumerate(loan_reader, start=1):

    start_id = loan_count + 1
    end_id = start_id + len(chunk)

    ids = pd.DataFrame({
        "loan_id": range(start_id, end_id),
        "customer_id": range(start_id, end_id)
    })

    loans = pd.concat(
        [ids, chunk.reset_index(drop=True)],
        axis=1
    )

    loans.rename(
        columns={
            "loan_amnt": "loan_amount",
            "funded_amnt": "funded_amount",
            "funded_amnt_inv": "investor_funds",
            "int_rate": "interest_rate",
            "annual_inc_joint": "joint_annual_income"
        },
        inplace=True
    )

    loans.to_sql(
        name="loans",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000
    )

    loan_count += len(loans)

    if chunk_number % 10 == 0:
        print(f"Loans loaded: {loan_count:,}")


print(
    f"Finished loading loans: "
    f"{loan_count:,}"
)


# ============================================================
# 7. VERIFY DATABASE
# ============================================================

print("\n========================================")
print("VERIFYING DATABASE")
print("========================================")

with engine.connect() as connection:

    customer_db_count = connection.execute(
        text("SELECT COUNT(*) FROM customers")
    ).scalar()

    loan_db_count = connection.execute(
        text("SELECT COUNT(*) FROM loans")
    ).scalar()


print(
    f"Customers in MySQL: "
    f"{customer_db_count:,}"
)

print(
    f"Loans in MySQL: "
    f"{loan_db_count:,}"
)

print("\n========================================")
print("DATA LOADING COMPLETE")
print("========================================")