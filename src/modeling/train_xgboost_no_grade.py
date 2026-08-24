from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Paths
# ============================================================

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

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_no_grade.json"
)

IMPORTANCE_FILE = (
    BASE_DIR
    / "data"
    / "reports"
    / "xgboost_no_grade_feature_importance.csv"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "risk_label"

CHUNK_SIZE = 50_000

SAMPLE_SIZE = 300_000

RANDOM_STATE = 42


# ============================================================
# Remove grade-related features
# ============================================================

def remove_grade_features(df):

    columns_to_remove = [
        column
        for column in df.columns
        if (
            column == "grade"
            or column == "sub_grade"
            or column.startswith("grade_")
            or column.startswith("sub_grade_")
        )
    ]

    if columns_to_remove:

        df = df.drop(
            columns=columns_to_remove
        )

    return df, columns_to_remove


# ============================================================
# Create representative training sample
# ============================================================

def create_training_sample():

    print("\n========================================")
    print("CREATING REPRESENTATIVE TRAINING SAMPLE")
    print("========================================")

    print(
        f"Target sample size: "
        f"{SAMPLE_SIZE:,}"
    )

    # --------------------------------------------------------
    # First determine total number of rows
    # --------------------------------------------------------

    print(
        "\nCounting training rows..."
    )

    total_rows = 0

    for chunk in pd.read_csv(
        TRAIN_FILE,
        chunksize=CHUNK_SIZE,
        usecols=[TARGET]
    ):
        total_rows += len(chunk)

    print(
        f"Total training rows: "
        f"{total_rows:,}"
    )

    # --------------------------------------------------------
    # Generate random row positions
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    selected_rows = np.sort(
        rng.choice(
            total_rows,
            size=SAMPLE_SIZE,
            replace=False
        )
    )

    print(
        f"Random rows selected: "
        f"{len(selected_rows):,}"
    )

    # --------------------------------------------------------
    # Read the dataset again in chunks
    # --------------------------------------------------------

    sampled_chunks = []

    current_start = 0

    selected_pointer = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            TRAIN_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Reading chunk {chunk_number}..."
        )

        current_end = (
            current_start
            + len(df)
        )

        # ----------------------------------------------------
        # Find selected rows belonging to this chunk
        # ----------------------------------------------------

        while (
            selected_pointer
            < len(selected_rows)
            and selected_rows[
                selected_pointer
            ] < current_end
        ):

            selected_pointer += 1

        chunk_selected = (
            selected_rows[
                (
                    selected_rows
                    >= current_start
                )
                &
                (
                    selected_rows
                    < current_end
                )
            ]
        )

        if len(chunk_selected) > 0:

            local_indices = (
                chunk_selected
                - current_start
            )

            sampled_chunks.append(
                df.iloc[
                    local_indices
                ]
            )

        current_start = current_end

        del df

    # --------------------------------------------------------
    # Combine selected rows
    # --------------------------------------------------------

    sample = pd.concat(
        sampled_chunks,
        ignore_index=True
    )

    del sampled_chunks

    # --------------------------------------------------------
    # Shuffle final sample
    # --------------------------------------------------------

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    print(
        f"\nFinal sample size: "
        f"{len(sample):,}"
    )

    print(
        "\nSample class distribution:"
    )

    print(
        sample[TARGET]
        .value_counts(
            normalize=True
        ).sort_index()
    )

    return sample

# ============================================================
# Main
# ============================================================

def main():

    print(
        "Starting XGBoost WITHOUT grade/sub-grade..."
    )

    # ========================================================
    # Create representative sample
    # ========================================================

    train = create_training_sample()

    # ========================================================
    # Separate target
    # ========================================================

    y_train = (
        train[TARGET]
        .astype("int8")
    )

    X_train = train.drop(
        columns=[TARGET]
    )

    # ========================================================
    # Remove grade/sub-grade
    # ========================================================

    X_train, removed_columns = (
        remove_grade_features(
            X_train
        )
    )

    print("\n========================================")
    print("REMOVING GRADE FEATURES")
    print("========================================")

    print(
        f"Removed columns: "
        f"{len(removed_columns)}"
    )

    print(
        "\nRemoved features:"
    )

    for column in removed_columns:
        print(
            f"  {column}"
        )

    feature_columns = (
        X_train.columns.tolist()
    )

    print(
        f"\nRemaining features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # Convert to float32
    # ========================================================

    X_train = X_train.astype(
        "float32"
    )

    # ========================================================
    # Class weight
    # ========================================================

    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative / positive
    )

    print(
        f"\nNegative samples: "
        f"{negative:,}"
    )

    print(
        f"Positive samples: "
        f"{positive:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================
    # Train XGBoost
    # ========================================================

    print("\n========================================")
    print("TRAINING XGBOOST WITHOUT GRADE")
    print("========================================")

    model = xgb.XGBClassifier(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=5,

        objective="binary:logistic",

        eval_metric="auc",

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",

        max_bin=256,

        random_state=RANDOM_STATE,

        n_jobs=4
    )

    model.fit(
        X_train,
        y_train,
        verbose=True
    )

    print(
        "\nXGBoost training completed."
    )

    # ========================================================
    # Save model
    # ========================================================

    model.save_model(
        MODEL_FILE
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_FILE}"
    )

    # ========================================================
    # Free training data
    # ========================================================

    del train
    del X_train
    del y_train

    # ========================================================
    # Evaluate full test set
    # ========================================================

    print("\n========================================")
    print("EVALUATING FULL TEST DATA")
    print("========================================")

    all_predictions = []
    all_probabilities = []
    all_actual = []

    total_test_rows = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Evaluating test chunk "
            f"{chunk_number}..."
        )

        y_test_chunk = (
            df[TARGET]
            .astype("int8")
        )

        X_test = df.drop(
            columns=[TARGET]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Remove exactly the same grade features
        # ----------------------------------------------------

        X_test, _ = (
            remove_grade_features(
                X_test
            )
        )

        # ----------------------------------------------------
        # Make sure feature order matches training
        # ----------------------------------------------------

        X_test = X_test[
            feature_columns
        ]

        X_test = X_test.astype(
            "float32"
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype("int8")

        all_predictions.append(
            predictions
        )

        all_probabilities.append(
            probabilities
        )

        all_actual.append(
            y_test_chunk.to_numpy()
        )

        total_test_rows += len(df)

        print(
            f"Rows evaluated: "
            f"{total_test_rows:,}"
        )

        del df
        del X_test
        del y_test_chunk

    # ========================================================
    # Combine results
    # ========================================================

    y_test = np.concatenate(
        all_actual
    )

    y_pred = np.concatenate(
        all_predictions
    )

    y_probability = np.concatenate(
        all_probabilities
    )

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # ========================================================
    # Results
    # ========================================================

    print("\n========================================")
    print("XGBOOST WITHOUT GRADE RESULTS")
    print("========================================")

    print(
        f"Training rows: "
        f"{SAMPLE_SIZE:,}"
    )

    print(
        f"Test rows: "
        f"{total_test_rows:,}"
    )

    print(
        f"Features used: "
        f"{len(feature_columns)}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print(
        "\nRows    = Actual"
    )

    print(
        "Columns = Predicted"
    )

    # ========================================================
    # Classification report
    # ========================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Good Loan",
                "Bad Loan"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # Feature importance
    # ========================================================

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(
        ascending=False
    )

    print("\n========================================")
    print("TOP 20 FEATURE IMPORTANCES")
    print("========================================")

    print(
        importance.head(20)
    )

    # ========================================================
    # Save feature importance
    # ========================================================

    importance.to_csv(
        IMPORTANCE_FILE,
        header=["importance"]
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{IMPORTANCE_FILE}"
    )


if __name__ == "__main__":
    main()