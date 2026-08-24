from pathlib import Path

import pandas as pd
import numpy as np


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

REPORT_DIR = (
    BASE_DIR
    / "data"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHUNK_SIZE = 50_000


# ============================================================
# Main EDA
# ============================================================

def main():

    print("Starting EDA...")
    print(f"Input: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # First inspect columns using a small sample
    # --------------------------------------------------------

    sample = pd.read_csv(
        INPUT_FILE,
        nrows=1000,
        low_memory=False
    )

    print("\nDataset columns:")
    print(f"Total columns: {len(sample.columns)}")

    print("\nColumns:")
    for column in sample.columns:
        print(f" - {column}")

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    total_rows = 0

    risk_counts = {
        0: 0,
        1: 0
    }

    missing_counts = pd.Series(
        0,
        index=sample.columns,
        dtype="int64"
    )

    # Numeric summary accumulators
    numeric_columns = sample.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_sum = pd.Series(
        0.0,
        index=numeric_columns
    )

    numeric_sum_sq = pd.Series(
        0.0,
        index=numeric_columns
    )

    numeric_count = pd.Series(
        0,
        index=numeric_columns,
        dtype="int64"
    )

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        total_rows += len(chunk)

        print(
            f"Processing chunk {chunk_number}..."
        )

        # ----------------------------------------------
        # Target distribution
        # ----------------------------------------------

        counts = chunk["risk_label"].value_counts()

        for label in [0, 1]:
            risk_counts[label] += counts.get(
                label,
                0
            )

        # ----------------------------------------------
        # Missing values
        # ----------------------------------------------

        missing_counts += chunk.isnull().sum()

        # ----------------------------------------------
        # Numeric statistics
        # ----------------------------------------------

        for column in numeric_columns:

            values = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )

            valid = values.dropna()

            numeric_sum[column] += valid.sum()
            numeric_sum_sq[column] += (
                (valid ** 2).sum()
            )
            numeric_count[column] += len(valid)

        del chunk

    # ========================================================
    # Results
    # ========================================================

    print("\n========================================")
    print("EDA SUMMARY")
    print("========================================")

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Total columns: {len(sample.columns)}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    target_df = pd.DataFrame({
        "risk_label": [0, 1],
        "count": [
            risk_counts[0],
            risk_counts[1]
        ]
    })

    target_df["percentage"] = (
        target_df["count"]
        / total_rows
        * 100
    )

    print("\nRisk distribution:")
    print(target_df)

    target_df.to_csv(
        REPORT_DIR / "risk_distribution.csv",
        index=False
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_df = pd.DataFrame({
        "column": missing_counts.index,
        "missing_count": missing_counts.values
    })

    missing_df["missing_percentage"] = (
        missing_df["missing_count"]
        / total_rows
        * 100
    )

    missing_df = missing_df.sort_values(
        "missing_count",
        ascending=False
    )

    print("\nTop missing-value columns:")
    print(
        missing_df
        .head(20)
        .to_string(index=False)
    )

    missing_df.to_csv(
        REPORT_DIR / "missing_values.csv",
        index=False
    )

    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    numeric_summary = pd.DataFrame({
        "count": numeric_count,
        "mean": (
            numeric_sum /
            numeric_count.replace(0, np.nan)
        )
    })

    variance = (
        numeric_sum_sq
        / numeric_count.replace(0, np.nan)
        - numeric_summary["mean"] ** 2
    )

    numeric_summary["std"] = np.sqrt(
        variance.clip(lower=0)
    )

    numeric_summary = numeric_summary.sort_index()

    print("\nNumeric feature summary:")
    print(numeric_summary.to_string())

    numeric_summary.to_csv(
        REPORT_DIR / "numeric_summary.csv"
    )

    print("\n========================================")
    print("EDA COMPLETED")
    print("========================================")

    print(f"Reports saved to:")
    print(REPORT_DIR)


if __name__ == "__main__":
    main()