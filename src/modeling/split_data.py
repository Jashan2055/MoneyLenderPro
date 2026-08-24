from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

TRAIN_FILE = OUTPUT_DIR / "train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 50_000
TEST_SIZE = 0.20


# ============================================================
# Columns that must NOT be used for prediction
# ============================================================

LEAKAGE_COLUMNS = [

    # Target source
    "loan_status",

    # Post-loan payment information
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",

    # Post-loan credit pull
    "last_credit_pull_d",

    # Repayment information
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",

    # Recovery / default outcome information
    "recoveries",
    "collection_recovery_fee",

    # Hardship information
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

    # Settlement information
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]


# ============================================================
# Columns that identify a record
# ============================================================

IDENTIFIER_COLUMNS = [
    "customer_id",
    "loan_id",
    "member_id",
    "id",
]


# ============================================================
# Main
# ============================================================

def main():

    print("Preparing train/test datasets...")
    print(f"Input: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # Remove previous files
    if TRAIN_FILE.exists():
        TRAIN_FILE.unlink()

    if TEST_FILE.exists():
        TEST_FILE.unlink()

    first_train_chunk = True
    first_test_chunk = True

    total_rows = 0
    train_rows = 0
    test_rows = 0

    train_bad = 0
    train_good = 0
    test_bad = 0
    test_good = 0

    # --------------------------------------------------------
    # Process dataset in chunks
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Processing chunk {chunk_number}..."
        )

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Remove leakage columns
        # ----------------------------------------------------

        columns_to_drop = [
            column
            for column in (
                LEAKAGE_COLUMNS +
                IDENTIFIER_COLUMNS
            )
            if column in chunk.columns
        ]

        chunk = chunk.drop(
            columns=columns_to_drop
        )

        # ----------------------------------------------------
        # Separate target
        # ----------------------------------------------------

        if "risk_label" not in chunk.columns:
            raise ValueError(
                "risk_label column not found."
            )

        # ----------------------------------------------------
        # Stratified split inside each chunk
        # ----------------------------------------------------

        train_chunk, test_chunk = train_test_split(
            chunk,
            test_size=TEST_SIZE,
            random_state=42 + chunk_number,
            stratify=chunk["risk_label"]
        )

        # ----------------------------------------------------
        # Track class counts
        # ----------------------------------------------------

        train_counts = (
            train_chunk["risk_label"]
            .value_counts()
        )

        test_counts = (
            test_chunk["risk_label"]
            .value_counts()
        )

        train_good += train_counts.get(0, 0)
        train_bad += train_counts.get(1, 0)

        test_good += test_counts.get(0, 0)
        test_bad += test_counts.get(1, 0)

        train_rows += len(train_chunk)
        test_rows += len(test_chunk)

        # ----------------------------------------------------
        # Write train
        # ----------------------------------------------------

        train_chunk.to_csv(
            TRAIN_FILE,
            mode="w" if first_train_chunk else "a",
            header=first_train_chunk,
            index=False
        )

        # ----------------------------------------------------
        # Write test
        # ----------------------------------------------------

        test_chunk.to_csv(
            TEST_FILE,
            mode="w" if first_test_chunk else "a",
            header=first_test_chunk,
            index=False
        )

        first_train_chunk = False
        first_test_chunk = False

        print(
            f"Train: {len(train_chunk):,} | "
            f"Test: {len(test_chunk):,}"
        )

        del chunk
        del train_chunk
        del test_chunk

    # ========================================================
    # Final summary
    # ========================================================

    print("\n========================================")
    print("TRAIN / TEST SPLIT COMPLETE")
    print("========================================")

    print(
        f"Total rows:  {total_rows:,}"
    )

    print(
        f"Train rows:  {train_rows:,}"
    )

    print(
        f"Test rows:   {test_rows:,}"
    )

    print(
        f"\nTrain good:  {train_good:,}"
    )

    print(
        f"Train bad:   {train_bad:,}"
    )

    print(
        f"Test good:   {test_good:,}"
    )

    print(
        f"Test bad:    {test_bad:,}"
    )

    print("\nTrain bad percentage:")

    print(
        f"{train_bad / train_rows * 100:.2f}%"
    )

    print("\nTest bad percentage:")

    print(
        f"{test_bad / test_rows * 100:.2f}%"
    )

    print("\nFiles created:")

    print(TRAIN_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()