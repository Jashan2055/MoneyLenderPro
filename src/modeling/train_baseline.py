from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
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


# ============================================================
# Configuration
# ============================================================

TARGET = "risk_label"
CHUNK_SIZE = 50_000


# ============================================================
# Main
# ============================================================

def main():

    print("Starting scaled Logistic Regression baseline...")

    # ========================================================
    # PASS 1: Learn scaling parameters
    # ========================================================

    print("\n========================================")
    print("PASS 1: FITTING STANDARD SCALER")
    print("========================================")

    scaler = StandardScaler()

    total_rows = 0
    first_chunk = True

    feature_columns = None

    for chunk_number, df in enumerate(
        pd.read_csv(
            TRAIN_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Scaler chunk {chunk_number}..."
        )

        X = df.drop(
            columns=[TARGET]
        )

        if first_chunk:

            feature_columns = X.columns.tolist()
            first_chunk = False

        X = X.astype("float32")

        scaler.partial_fit(X)

        total_rows += len(X)

        del df
        del X

    print(
        f"\nScaler fitted on "
        f"{total_rows:,} training rows."
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # ========================================================
    # PASS 2: Train model
    # ========================================================

    print("\n========================================")
    print("PASS 2: TRAINING LOGISTIC REGRESSION")
    print("========================================")

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        max_iter=1,
        class_weight={
            0: 0.5,
            1: 2.0
        },
        random_state=42
    )

    classes = np.array([0, 1])

    total_train_rows = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            TRAIN_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Training chunk {chunk_number}..."
        )

        X = df.drop(
            columns=[TARGET]
        )

        y = df[TARGET].astype("int8")

        X = X.astype("float32")

        # ----------------------------------------------------
        # Scale using TRAINING scaler
        # ----------------------------------------------------

        X_scaled = scaler.transform(X)

        # ----------------------------------------------------
        # Incremental training
        # ----------------------------------------------------

        model.partial_fit(
            X_scaled,
            y,
            classes=classes
        )

        total_train_rows += len(X)

        print(
            f"Rows processed: "
            f"{total_train_rows:,}"
        )

        del df
        del X
        del X_scaled
        del y

    print(
        f"\nTraining complete."
        f"\nTotal rows: "
        f"{total_train_rows:,}"
    )

    # ========================================================
    # PASS 3: Evaluate
    # ========================================================

    print("\n========================================")
    print("PASS 3: EVALUATING TEST SET")
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

        X = df.drop(
            columns=[TARGET]
        )

        y = df[TARGET].astype("int8")

        X = X.astype("float32")

        # ----------------------------------------------------
        # Use scaler learned ONLY from training data
        # ----------------------------------------------------

        X_scaled = scaler.transform(X)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = model.predict(
            X_scaled
        )

        probabilities = (
            model.predict_proba(
                X_scaled
            )[:, 1]
        )

        all_predictions.append(
            predictions
        )

        all_probabilities.append(
            probabilities
        )

        all_actual.append(
            y.to_numpy()
        )

        total_test_rows += len(X)

        print(
            f"Rows evaluated: "
            f"{total_test_rows:,}"
        )

        del df
        del X
        del X_scaled
        del y

    # ========================================================
    # Combine predictions
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
    print("SCALED LOGISTIC REGRESSION RESULTS")
    print("========================================")

    print(
        f"Test rows:  {total_test_rows:,}"
    )

    print(
        f"Accuracy:   {accuracy:.4f}"
    )

    print(
        f"Precision:  {precision:.4f}"
    )

    print(
        f"Recall:     {recall:.4f}"
    )

    print(
        f"F1 Score:   {f1:.4f}"
    )

    print(
        f"ROC-AUC:    {roc_auc:.4f}"
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")

    print(cm)

    print(
        "\nRows    = Actual"
    )

    print(
        "Columns = Predicted"
    )

    # ========================================================
    # Classification Report
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


if __name__ == "__main__":
    main()