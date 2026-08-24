from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "model_dataset.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "model_features.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 50_000


# ============================================================
# Helper functions
# ============================================================

def create_features(df):
    """
    Create engineered features for the credit-risk model.
    """

    # --------------------------------------------------------
    # 0. Handle unreliable source values
    # --------------------------------------------------------

    # Very small income values produce meaningless ratios.
    # Keep the original dataset untouched, but treat these
    # values as missing for the modeling dataset.

    unreliable_income = df["annual_income"] < 1000

    df["income_unreliable"] = (
        unreliable_income
    ).astype("int8")

    df.loc[
        unreliable_income,
        "annual_income"
    ] = pd.NA

    # DTI = 999 is a special/sentinel value, not a
    # meaningful DTI measurement.

    dti_unavailable = df["dti"] == 999

    df["dti_unavailable"] = (
        dti_unavailable
    ).astype("int8")

    df.loc[
        dti_unavailable,
        "dti"
    ] = pd.NA


    # --------------------------------------------------------
    # 1. Loan-to-income ratio
    # --------------------------------------------------------

    df["loan_to_income"] = (
        df["loan_amount"] /
        df["annual_income"]
    )


    # --------------------------------------------------------
    # 2. Monthly income
    # --------------------------------------------------------

    df["monthly_income"] = (
        df["annual_income"] / 12
    )


    # --------------------------------------------------------
    # 3. Installment-to-income ratio
    # --------------------------------------------------------

    df["installment_to_income"] = (
        df["installment"] /
        df["monthly_income"]
    )


    # --------------------------------------------------------
    # 4. Credit history length
    # --------------------------------------------------------

    df["credit_history_years"] = (
        (
            df["issue_d"] -
            df["earliest_cr_line"]
        ).dt.days / 365.25
    )


    # --------------------------------------------------------
    # 5. Active credit account ratio
    # --------------------------------------------------------

    df["active_account_ratio"] = (
        df["open_acc"] /
        df["total_acc"].replace(0, pd.NA)
    )


    # --------------------------------------------------------
    # 6. Funded amount ratio
    # --------------------------------------------------------

    df["funded_amount_ratio"] = (
        df["funded_amount"] /
        df["loan_amount"].replace(0, pd.NA)
    )


    # --------------------------------------------------------
    # 7. Revolving balance ratio
    # --------------------------------------------------------

    df["revolving_balance_ratio"] = (
        df["revol_bal"] /
        df["total_rev_hi_lim"].replace(0, pd.NA)
    )


    # --------------------------------------------------------
    # 8. Delinquency ratio
    # --------------------------------------------------------

    df["delinquency_ratio"] = (
        df["delinq_2yrs"] /
        df["total_acc"].replace(0, pd.NA)
    )


    # --------------------------------------------------------
    # 9. Public record ratio
    # --------------------------------------------------------

    df["public_record_ratio"] = (
        df["pub_rec"] /
        df["total_acc"].replace(0, pd.NA)
    )


    # --------------------------------------------------------
    # 10. Balance-to-income ratio
    # --------------------------------------------------------

    df["balance_to_income"] = (
        df["tot_cur_bal"] /
        df["annual_income"]
    )


    # --------------------------------------------------------
    # 11. Utilization difference
    # --------------------------------------------------------

    if (
        "revol_util" in df.columns
        and "bc_util" in df.columns
    ):
        df["utilization_difference"] = (
            df["revol_util"] -
            df["bc_util"]
        )


    # --------------------------------------------------------
    # 12. High revolving utilization
    # --------------------------------------------------------

    if "revol_util" in df.columns:

        df["high_revolving_utilization"] = (
            df["revol_util"] >= 80
        ).astype("int8")


    # --------------------------------------------------------
    # 13. High DTI
    # --------------------------------------------------------

    if "dti" in df.columns:

        df["high_dti"] = (
            df["dti"] >= 30
        ).astype("int8")


    # --------------------------------------------------------
    # 14. Long employment
    # --------------------------------------------------------

    if "emp_length" in df.columns:

        df["long_employment"] = (
            df["emp_length"] >= 5
        ).astype("int8")


    return df
    """
    Create engineered features for the credit-risk model.
    """

    # --------------------------------------------------------
    # 1. Loan-to-income ratio
    # --------------------------------------------------------

    df["loan_to_income"] = (
        df["loan_amount"] /
        df["annual_income"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 2. Monthly income
    # --------------------------------------------------------

    df["monthly_income"] = (
        df["annual_income"] / 12
    )

    # --------------------------------------------------------
    # 3. Installment-to-income ratio
    # --------------------------------------------------------

    df["installment_to_income"] = (
        df["installment"] /
        df["monthly_income"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 4. Credit history length
    # --------------------------------------------------------

    df["credit_history_years"] = (
        (
            df["issue_d"] -
            df["earliest_cr_line"]
        ).dt.days / 365.25
    )

    # --------------------------------------------------------
    # 5. Active credit account ratio
    # --------------------------------------------------------

    df["active_account_ratio"] = (
        df["open_acc"] /
        df["total_acc"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 6. Total funded amount ratio
    # --------------------------------------------------------

    df["funded_amount_ratio"] = (
        df["funded_amount"] /
        df["loan_amount"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 7. Revolving balance to credit limit
    # --------------------------------------------------------

    df["revolving_balance_ratio"] = (
        df["revol_bal"] /
        df["total_rev_hi_lim"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 8. Delinquency ratio
    # --------------------------------------------------------

    df["delinquency_ratio"] = (
        df["delinq_2yrs"] /
        df["total_acc"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 9. Public record ratio
    # --------------------------------------------------------

    df["public_record_ratio"] = (
        df["pub_rec"] /
        df["total_acc"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 10. Total balance to income
    # --------------------------------------------------------

    df["balance_to_income"] = (
        df["tot_cur_bal"] /
        df["annual_income"].replace(0, pd.NA)
    )

    # --------------------------------------------------------
    # 11. Credit utilization difference
    # --------------------------------------------------------

    if "revol_util" in df.columns and "bc_util" in df.columns:
        df["utilization_difference"] = (
            df["revol_util"] - df["bc_util"]
        )

    # --------------------------------------------------------
    # 12. High utilization indicator
    # --------------------------------------------------------

    if "revol_util" in df.columns:
        df["high_revolving_utilization"] = (
            df["revol_util"] >= 80
        ).astype("int8")

    # --------------------------------------------------------
    # 13. High DTI indicator
    # --------------------------------------------------------

    if "dti" in df.columns:
        df["high_dti"] = (
            df["dti"] >= 30
        ).astype("int8")

    # --------------------------------------------------------
    # 14. Long employment indicator
    # --------------------------------------------------------

    if "emp_length" in df.columns:
        df["long_employment"] = (
            df["emp_length"] >= 5
        ).astype("int8")

    return df


# ============================================================
# Main processing
# ============================================================

def main():

    print("Starting feature engineering...")
    print(f"Input: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # Remove previous output so we don't append to an old file
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    first_chunk = True
    total_processed = 0
    chunk_count = 0

    # --------------------------------------------------------
    # Read and process in chunks
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_count += 1

        print(
            f"Processing chunk {chunk_count} "
            f"({len(chunk):,} rows)..."
        )

        # ----------------------------------------------------
        # Convert dates
        # ----------------------------------------------------

        chunk["issue_d"] = pd.to_datetime(
            chunk["issue_d"],
            errors="coerce"
        )

        chunk["earliest_cr_line"] = pd.to_datetime(
            chunk["earliest_cr_line"],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Create engineered features
        # ----------------------------------------------------

        chunk = create_features(chunk)

        # ----------------------------------------------------
        # Clean infinite values
        # ----------------------------------------------------

        chunk.replace(
            [float("inf"), float("-inf")],
            pd.NA,
            inplace=True
        )

        # ----------------------------------------------------
        # Write processed chunk
        # ----------------------------------------------------

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        total_processed += len(chunk)

        print(
            f"Completed. Total processed: "
            f"{total_processed:,}"
        )

        # Release chunk before processing the next one
        del chunk

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("========================================")

    print(f"Chunks processed: {chunk_count}")
    print(f"Rows processed:   {total_processed:,}")
    print(f"Output file:      {OUTPUT_FILE}")


if __name__ == "__main__":
    main()