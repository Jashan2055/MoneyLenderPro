from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_dataset.csv"
)

CHUNK_SIZE = 50_000


def main():

    total_rows = 0

    income_below_1000 = 0
    income_zero = 0

    dti_999 = 0
    dti_above_50 = 0
    dti_above_100 = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            usecols=[
                "annual_income",
                "dti"
            ],
            low_memory=False
        ),
        start=1
    ):

        print(f"Processing chunk {chunk_number}...")

        total_rows += len(chunk)

        income = pd.to_numeric(
            chunk["annual_income"],
            errors="coerce"
        )

        dti = pd.to_numeric(
            chunk["dti"],
            errors="coerce"
        )

        income_below_1000 += (
            income < 1000
        ).sum()

        income_zero += (
            income <= 0
        ).sum()

        dti_999 += (
            dti == 999
        ).sum()

        dti_above_50 += (
            dti > 50
        ).sum()

        dti_above_100 += (
            dti > 100
        ).sum()

        del chunk

    print("\n========================================")
    print("DATA QUALITY SUMMARY")
    print("========================================")

    print(f"Total rows: {total_rows:,}")

    print(
        f"\nAnnual income < $1,000: "
        f"{income_below_1000:,}"
    )

    print(
        f"Annual income <= $0: "
        f"{income_zero:,}"
    )

    print(
        f"\nDTI = 999: "
        f"{dti_999:,}"
    )

    print(
        f"DTI > 50: "
        f"{dti_above_50:,}"
    )

    print(
        f"DTI > 100: "
        f"{dti_above_100:,}"
    )


if __name__ == "__main__":
    main()