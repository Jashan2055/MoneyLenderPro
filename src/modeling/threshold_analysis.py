from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_prepared.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "xgboost_baseline.json"
)

REPORT_FILE = (
    BASE_DIR
    / "data"
    / "reports"
    / "threshold_analysis.csv"
)

CHUNK_SIZE = 50_000
TARGET = "risk_label"


# ============================================================
# Main
# ============================================================

def main():

    print("Loading XGBoost model...")

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_FILE
    )

    print(
        f"Model loaded from:\n"
        f"{MODEL_FILE}"
    )

    # ========================================================
    # Generate test probabilities
    # ========================================================

    print("\n========================================")
    print("GENERATING TEST PROBABILITIES")
    print("========================================")

    probabilities = []
    actual = []

    total_rows = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        print(
            f"Processing chunk "
            f"{chunk_number}..."
        )

        X = df.drop(
            columns=[TARGET]
        )

        y = df[TARGET].astype(
            "int8"
        )

        X = X.astype(
            "float32"
        )

        prob = model.predict_proba(
            X
        )[:, 1]

        probabilities.append(
            prob
        )

        actual.append(
            y.to_numpy()
        )

        total_rows += len(df)

        del df
        del X
        del y

    y_probability = np.concatenate(
        probabilities
    )

    y_test = np.concatenate(
        actual
    )

    print(
        f"\nTotal test rows: "
        f"{total_rows:,}"
    )

    # ========================================================
    # Threshold analysis
    # ========================================================

    print("\n========================================")
    print("THRESHOLD ANALYSIS")
    print("========================================")

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    results = []

    for threshold in thresholds:

        y_pred = (
            y_probability >= threshold
        ).astype("int8")

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

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                y_pred
            ).ravel()
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_good": tn,
            "false_good": fp,
            "missed_bad": fn,
            "caught_bad": tp,
        })

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # Display
    # ========================================================

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Best F1
    # ========================================================

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n========================================")
    print("BEST F1 THRESHOLD")
    print("========================================")

    print(
        best_f1.to_string()
    )

    # ========================================================
    # Save
    # ========================================================

    results_df.to_csv(
        REPORT_FILE,
        index=False
    )

    print(
        f"\nReport saved to:"
        f"\n{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()