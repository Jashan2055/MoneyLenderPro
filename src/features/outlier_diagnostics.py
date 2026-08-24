from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_features.csv"
)

CHUNK_SIZE = 50_000


def main():

    print("Searching for extreme feature values...")

    suspicious_rows = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(f"Processing chunk {chunk_number}...")

        # --------------------------------------------------
        # Find suspicious ratio values
        # --------------------------------------------------

        mask = (
            (chunk["loan_to_income"] > 2)
            |
            (chunk["installment_to_income"] > 1)
            |
            (chunk["balance_to_income"] > 10)
            |
            (chunk["dti"] > 50)
        )

        suspicious = chunk.loc[
            mask,
            [
                "annual_income",
                "loan_amount",
                "installment",
                "tot_cur_bal",
                "dti",
                "loan_to_income",
                "installment_to_income",
                "balance_to_income",
                "risk_label",
                "grade",
                "interest_rate"
            ]
        ]

        if not suspicious.empty:

            suspicious_rows.append(
                suspicious.head(100)
            )

        del chunk

    # --------------------------------------------------
    # Combine only suspicious observations
    # --------------------------------------------------

    if not suspicious_rows:

        print("No suspicious rows found.")
        return

    result = pd.concat(
        suspicious_rows,
        ignore_index=True
    )

    # Remove duplicates
    result = result.drop_duplicates()

    # Sort by most extreme loan-to-income
    result = result.sort_values(
        "loan_to_income",
        ascending=False
    )

    print("\n========================================")
    print("EXTREME VALUE DIAGNOSTICS")
    print("========================================")

    print(
        result.head(30).to_string(
            index=False
        )
    )

    output = (
        BASE_DIR
        / "data"
        / "reports"
        / "outlier_diagnostics.csv"
    )

    result.to_csv(
        output,
        index=False
    )

    print("\nSaved diagnostic report:")
    print(output)


if __name__ == "__main__":
    main()