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

    print("Validating engineered features...")
    print(f"Input: {INPUT_FILE}")

    features_to_check = [
        "dti",
        "loan_to_income",
        "installment_to_income",
        "balance_to_income",
        "revolving_balance_ratio",
        "interest_rate",
        "revol_util",
        "loan_amount",
    ]

    values = {
        feature: []
        for feature in features_to_check
    }

    # --------------------------------------------------
    # Read chunks
    # --------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            usecols=features_to_check,
            low_memory=False
        ),
        start=1
    ):

        print(f"Processing chunk {chunk_number}...")

        for feature in features_to_check:

            series = pd.to_numeric(
                chunk[feature],
                errors="coerce"
            ).dropna()

            # Sample values from each chunk
            values[feature].extend(
                series.sample(
                    n=min(5000, len(series)),
                    random_state=42
                ).tolist()
            )

        del chunk

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n========================================")
    print("FEATURE VALIDATION")
    print("========================================")

    for feature in features_to_check:

        series = pd.Series(values[feature])

        print(f"\n{feature}")
        print("-" * len(feature))

        print(
            series.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                    0.999
                ]
            )
        )

        # Extreme counts
        if feature == "dti":

            print(
                "\nDTI > 50:",
                (series > 50).sum()
            )

            print(
                "DTI > 100:",
                (series > 100).sum()
            )

            print(
                "DTI > 200:",
                (series > 200).sum()
            )

        elif feature == "loan_to_income":

            print(
                "\nLoan/Income > 1:",
                (series > 1).sum()
            )

            print(
                "Loan/Income > 2:",
                (series > 2).sum()
            )

            print(
                "Loan/Income > 5:",
                (series > 5).sum()
            )

        elif feature == "installment_to_income":

            print(
                "\nInstallment/Income > 0.5:",
                (series > 0.5).sum()
            )

            print(
                "Installment/Income > 1:",
                (series > 1).sum()
            )

        elif feature == "balance_to_income":

            print(
                "\nBalance/Income > 5:",
                (series > 5).sum()
            )

            print(
                "Balance/Income > 10:",
                (series > 10).sum()
            )

        elif feature == "revol_util":

            print(
                "\nRevol Util > 100:",
                (series > 100).sum()
            )

        elif feature == "interest_rate":

            print(
                "\nInterest Rate > 25:",
                (series > 25).sum()
            )

    print("\n========================================")
    print("VALIDATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()