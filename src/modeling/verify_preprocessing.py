from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_prepared.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_prepared.csv"
)


def inspect_file(path, name):

    print("\n========================================")
    print(f"{name}")
    print("========================================")

    df = pd.read_csv(
        path,
        nrows=1000,
        low_memory=False
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Sample rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    print("\nRisk distribution:")

    print(
        df["risk_label"]
        .value_counts()
    )

    print("\nRisk percentages:")

    print(
        (
            df["risk_label"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isnull().sum().sum()

    print(
        f"\nMissing values in sample: {missing:,}"
    )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric = df.select_dtypes(
        include="number"
    )

    infinite_count = (
        numeric
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(
        f"Infinite values in sample: "
        f"{infinite_count:,}"
    )

    # --------------------------------------------------------
    # Duplicate columns
    # --------------------------------------------------------

    duplicate_columns = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    print(
        f"Duplicate columns: "
        f"{duplicate_columns}"
    )

    return df


def main():

    print("Verifying prepared datasets...")

    train = inspect_file(
        TRAIN_FILE,
        "TRAIN"
    )

    test = inspect_file(
        TEST_FILE,
        "TEST"
    )

    # ========================================================
    # Compare feature structure
    # ========================================================

    print("\n========================================")
    print("TRAIN / TEST STRUCTURE")
    print("========================================")

    train_columns = train.columns.tolist()
    test_columns = test.columns.tolist()

    print(
        f"Train columns: {len(train_columns)}"
    )

    print(
        f"Test columns:  {len(test_columns)}"
    )

    if train_columns == test_columns:

        print(
            "\n✓ Train and test columns match."
        )

    else:

        print(
            "\n✗ Train and test columns DO NOT match."
        )

        train_only = set(
            train_columns
        ) - set(test_columns)

        test_only = set(
            test_columns
        ) - set(train_columns)

        print(
            "\nTrain-only columns:"
        )

        print(train_only)

        print(
            "\nTest-only columns:"
        )

        print(test_only)

    # ========================================================
    # Check target
    # ========================================================

    print("\n========================================")
    print("TARGET CHECK")
    print("========================================")

    print(
        "Train risk distribution:"
    )

    print(
        train["risk_label"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        "\nTest risk distribution:"
    )

    print(
        test["risk_label"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print("\n========================================")
    print("VERIFICATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()