from pathlib import Path
import json

import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "test.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "train_prepared.csv"
)

TEST_OUTPUT = (
    OUTPUT_DIR
    / "test_prepared.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "preprocessing_metadata.json"
)


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 50_000


# ============================================================
# Columns we don't want as ML features
# ============================================================

HIGH_CARDINALITY_COLUMNS = [
    "emp_title",
    "title",
    "zip_code",
]


# Raw date columns are not directly useful to the first model.
# We already created credit_history_years from the dates.
DATE_COLUMNS = [
    "issue_d",
    "earliest_cr_line",
]


# ============================================================
# Helper
# ============================================================

def get_feature_columns():

    """
    Read the first chunk and determine which columns are
    numeric/categorical after removing identifiers and
    high-cardinality fields.
    """

    sample = pd.read_csv(
        TRAIN_FILE,
        nrows=1000,
        low_memory=False
    )

    sample.drop(
        columns=[
            column
            for column in (
                HIGH_CARDINALITY_COLUMNS
                + DATE_COLUMNS
            )
            if column in sample.columns
        ],
        inplace=True
    )

    sample.drop(
        columns=["risk_label"],
        inplace=True
    )

    numeric_columns = (
        sample
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = (
        sample
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    return numeric_columns, categorical_columns


# ============================================================
# Pass 1
# Learn preprocessing parameters from TRAIN ONLY
# ============================================================

def learn_preprocessing():

    print("\n========================================")
    print("PASS 1: LEARNING TRAIN PREPROCESSING")
    print("========================================")

    numeric_columns, categorical_columns = (
        get_feature_columns()
    )

    print(
        f"\nNumeric columns: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical columns: "
        f"{len(categorical_columns)}"
    )

    # --------------------------------------------------------
    # Numeric medians
    # --------------------------------------------------------

    numeric_values = {
        column: []
        for column in numeric_columns
    }

    # --------------------------------------------------------
    # Categorical vocabularies
    # --------------------------------------------------------

    category_values = {
        column: set()
        for column in categorical_columns
    }

    total_rows = 0

    # --------------------------------------------------------
    # Scan TRAIN in chunks
    # --------------------------------------------------------

    for chunk_number, df in enumerate(
        pd.read_csv(
            TRAIN_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Learning from train chunk "
            f"{chunk_number}..."
        )

        total_rows += len(df)

        # ----------------------------------------------------
        # Numeric values
        # ----------------------------------------------------

        for column in numeric_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            # Store only valid values.
            numeric_values[column].append(
                values.dropna()
            )

        # ----------------------------------------------------
        # Categorical values
        # ----------------------------------------------------

        for column in categorical_columns:

            values = (
                df[column]
                .fillna("Unknown")
                .astype(str)
            )

            category_values[column].update(
                values.unique()
            )

        del df

    # --------------------------------------------------------
    # Calculate numeric medians
    # --------------------------------------------------------

    numeric_medians = {}

    for column in numeric_columns:

        if numeric_values[column]:

            combined = pd.concat(
                numeric_values[column],
                ignore_index=True
            )

            numeric_medians[column] = float(
                combined.median()
            )

        else:

            numeric_medians[column] = 0.0

    # --------------------------------------------------------
    # Sort categorical vocabularies
    # --------------------------------------------------------

    categories = {}

    for column in categorical_columns:

        categories[column] = sorted(
            category_values[column]
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\nTraining rows scanned:")
    print(f"{total_rows:,}")

    print("\nLearned numeric medians:")

    for column, median in numeric_medians.items():

        print(
            f"{column}: {median}"
        )

    print("\nCategorical vocabulary sizes:")

    for column, values in categories.items():

        print(
            f"{column}: {len(values)} categories"
        )

    return (
        numeric_columns,
        categorical_columns,
        numeric_medians,
        categories
    )


# ============================================================
# Transform one chunk
# ============================================================

def transform_chunk(
    df,
    numeric_columns,
    categorical_columns,
    numeric_medians,
    categories
):

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # Remove high-cardinality fields
    # --------------------------------------------------------

    df.drop(
        columns=[
            column
            for column in HIGH_CARDINALITY_COLUMNS
            if column in df.columns
        ],
        inplace=True
    )

    # --------------------------------------------------------
    # Remove raw dates
    # --------------------------------------------------------

    df.drop(
        columns=[
            column
            for column in DATE_COLUMNS
            if column in df.columns
        ],
        inplace=True
    )

    # --------------------------------------------------------
    # Separate target
    # --------------------------------------------------------

    y = (
        df["risk_label"]
        .reset_index(drop=True)
        .astype("int8")
    )

    X = df.drop(
        columns=["risk_label"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    for column in numeric_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        X[column] = X[column].fillna(
            numeric_medians[column]
        )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    for column in categorical_columns:

        X[column] = (
            X[column]
            .fillna("Unknown")
            .astype(str)
        )

        known_categories = set(
            categories[column]
        )

        X.loc[
            ~X[column].isin(known_categories),
            column
        ] = "Unknown"

    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    encoded_parts = []

    # Numeric features
    numeric_part = (
        X[numeric_columns]
        .reset_index(drop=True)
    )

    encoded_parts.append(
        numeric_part
    )

    # Categorical features
    for column in categorical_columns:

        categorical = pd.Categorical(
            X[column],
            categories=categories[column]
        )

        encoded = pd.get_dummies(
            categorical,
            prefix=column,
            dtype="int8"
        )

        # VERY IMPORTANT:
        # Keep exactly one row per input observation.
        encoded = encoded.reset_index(
            drop=True
        )

        encoded_parts.append(
            encoded
        )

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    X = pd.concat(
        encoded_parts,
        axis=1
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(X) != len(y):

        raise ValueError(
            f"Row mismatch after preprocessing: "
            f"X={len(X)}, y={len(y)}"
        )

    # --------------------------------------------------------
    # Add target
    # --------------------------------------------------------

    X["risk_label"] = y.values

    return X


# ============================================================
# Pass 2
# Transform a complete file in chunks
# ============================================================

def transform_file(
    input_file,
    output_file,
    numeric_columns,
    categorical_columns,
    numeric_medians,
    categories
):

    print("\n========================================")
    print(f"TRANSFORMING: {input_file.name}")
    print("========================================")

    if output_file.exists():
        output_file.unlink()

    first_chunk = True
    total_rows = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            input_file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Transforming chunk {chunk_number}..."
        )

        processed = transform_chunk(
            df,
            numeric_columns,
            categorical_columns,
            numeric_medians,
            categories
        )

        processed.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        total_rows += len(processed)

        print(
            f"Completed. Total rows: "
            f"{total_rows:,}"
        )

        del df
        del processed

    print(
        f"\nFinished {output_file.name}: "
        f"{total_rows:,} rows"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("Starting ML preprocessing...")

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test file not found:\n{TEST_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # PASS 1
    # ========================================================

    (
        numeric_columns,
        categorical_columns,
        numeric_medians,
        categories
    ) = learn_preprocessing()

    # ========================================================
    # Save metadata
    # ========================================================

    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_medians": numeric_medians,
        "categories": categories,
        "chunk_size": CHUNK_SIZE
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nPreprocessing metadata saved to:"
        f"\n{METADATA_FILE}"
    )

    # ========================================================
    # PASS 2 - TRAIN
    # ========================================================

    transform_file(
        TRAIN_FILE,
        TRAIN_OUTPUT,
        numeric_columns,
        categorical_columns,
        numeric_medians,
        categories
    )

    # ========================================================
    # PASS 2 - TEST
    # ========================================================

    transform_file(
        TEST_FILE,
        TEST_OUTPUT,
        numeric_columns,
        categorical_columns,
        numeric_medians,
        categories
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n========================================")
    print("PREPROCESSING COMPLETED")
    print("========================================")

    print(
        f"\nTrain output:"
        f"\n{TRAIN_OUTPUT}"
    )

    print(
        f"\nTest output:"
        f"\n{TEST_OUTPUT}"
    )

    print(
        f"\nMetadata:"
        f"\n{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()