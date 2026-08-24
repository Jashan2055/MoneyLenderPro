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
    / "xgboost_baseline.json"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "risk_label"

CHUNK_SIZE = 50_000

SAMPLE_SIZE = 300_000

RANDOM_STATE = 42


# ============================================================
# Random sampling from entire training set
# ============================================================

def create_training_sample():

    print("\n========================================")
    print("CREATING REPRESENTATIVE TRAINING SAMPLE")
    print("========================================")

    print(
        f"Target sample size: "
        f"{SAMPLE_SIZE:,}"
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sampled_chunks = []

    total_rows = 0

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

        total_rows += len(df)

        # ----------------------------------------------------
        # Determine how many rows to sample
        # ----------------------------------------------------

        remaining_rows = (
            SAMPLE_SIZE
            - sum(
                len(x)
                for x in sampled_chunks
            )
        )

        if remaining_rows <= 0:
            break

        # Probability of selecting a row.
        #
        # This is an approximate uniform sample across
        # the entire training dataset.

        probability = min(
            1.0,
            remaining_rows / len(df)
        )

        mask = (
            rng.random(len(df))
            < probability
        )

        selected = df.loc[
            mask
        ]

        if not selected.empty:

            sampled_chunks.append(
                selected
            )

        current_count = sum(
            len(x)
            for x in sampled_chunks
        )

        print(
            f"Sampled so far: "
            f"{current_count:,}"
        )

        del df
        del selected

    sample = pd.concat(
        sampled_chunks,
        ignore_index=True
    )

    # --------------------------------------------------------
    # If sampling slightly exceeded target
    # --------------------------------------------------------

    if len(sample) > SAMPLE_SIZE:

        sample = sample.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        ).reset_index(
            drop=True
        )

    print(
        f"\nFull training rows scanned: "
        f"{total_rows:,}"
    )

    print(
        f"Final sample size: "
        f"{len(sample):,}"
    )

    print(
        "\nSample class distribution:"
    )

    print(
        sample[TARGET]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    return sample


# ============================================================
# Main
# ============================================================

def main():

    print("Starting representative XGBoost training...")

    # ========================================================
    # Create representative training sample
    # ========================================================

    train = create_training_sample()

    # ========================================================
    # Separate X/y
    # ========================================================

    X_train = train.drop(
        columns=[TARGET]
    )

    y_train = train[TARGET].astype(
        "int8"
    )

    feature_columns = (
        X_train.columns.tolist()
    )

    X_train = X_train.astype(
        "float32"
    )

    print(
        f"\nTraining shape: "
        f"{X_train.shape}"
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
    # XGBoost
    # ========================================================

    print("\n========================================")
    print("TRAINING XGBOOST")
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
    # Evaluate TEST in chunks
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

        X_test = df.drop(
            columns=[TARGET]
        )

        y_test_chunk = (
            df[TARGET]
            .astype("int8")
        )

        X_test = X_test.astype(
            "float32"
        )

        # ----------------------------------------------------
        # Probability predictions
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # Default threshold = 0.5
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
    # Combine evaluation results
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
    print("XGBOOST RESULTS")
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

    importance_file = (
        BASE_DIR
        / "data"
        / "reports"
        / "xgboost_feature_importance.csv"
    )

    importance.to_csv(
        importance_file,
        header=["importance"]
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{importance_file}"
    )


if __name__ == "__main__":
    main()