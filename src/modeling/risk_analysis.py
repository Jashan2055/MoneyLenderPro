from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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
    / "xgboost_final.json"
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

RISK_REPORT = (
    REPORT_DIR
    / "risk_tier_analysis.csv"
)

CHUNK_SIZE = 50_000

TARGET = "risk_label"


# ============================================================
# Remove grade features
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

    return df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Starting final risk analysis..."
    )

    # ========================================================
    # Load model
    # ========================================================

    print(
        "\nLoading final XGBoost model..."
    )

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_FILE
    )

    print(
        f"Model loaded from:"
        f"\n{MODEL_FILE}"
    )

    # ========================================================
    # Generate predictions
    # ========================================================

    all_probabilities = []

    all_actual = []

    total_rows = 0

    print(
        "\n========================================"
    )

    print(
        "GENERATING FINAL TEST PREDICTIONS"
    )

    print(
        "========================================"
    )

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

        y = (
            df[TARGET]
            .astype("int8")
        )

        X = df.drop(
            columns=[TARGET]
        )

        X = remove_grade_features(
            X
        )

        # ----------------------------------------------------
        # Feature order must match training model
        # ----------------------------------------------------

        X = X.astype(
            "float32"
        )

        probabilities = (
            model.predict_proba(
                X
            )[:, 1]
        )

        all_probabilities.append(
            probabilities
        )

        all_actual.append(
            y.to_numpy()
        )

        total_rows += len(df)

        print(
            f"Rows processed: "
            f"{total_rows:,}"
        )

        del df
        del X
        del y

    # ========================================================
    # Combine
    # ========================================================

    probabilities = np.concatenate(
        all_probabilities
    )

    actual = np.concatenate(
        all_actual
    )

    # ========================================================
    # Probability distribution
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "RISK PROBABILITY DISTRIBUTION"
    )

    print(
        "========================================"
    )

    probability_series = pd.Series(
        probabilities
    )

    print(
        probability_series.describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )

    # ========================================================
    # Risk tiers
    # ========================================================

    # These are intentionally broad starting thresholds.
    # We'll inspect the actual bad-loan rates and adjust
    # if necessary.

    low_threshold = 0.30
    high_threshold = 0.60

    risk_tier = np.select(
        [
            probabilities < low_threshold,
            probabilities < high_threshold
        ],
        [
            "Low Risk",
            "Medium Risk"
        ],
        default="High Risk"
    )

    # ========================================================
    # Create analysis dataframe
    # ========================================================

    analysis = pd.DataFrame({

        "actual_risk": actual,

        "predicted_probability": (
            probabilities
        ),

        "risk_tier": risk_tier
    })

    # ========================================================
    # Tier summary
    # ========================================================

    tier_summary = (
        analysis
        .groupby(
            "risk_tier",
            observed=True
        )
        .agg(
            applicants=(
                "actual_risk",
                "count"
            ),

            actual_bad_loans=(
                "actual_risk",
                "sum"
            ),

            average_predicted_risk=(
                "predicted_probability",
                "mean"
            )
        )
        .reset_index()
    )

    tier_summary[
        "applicant_percentage"
    ] = (
        tier_summary["applicants"]
        / len(analysis)
        * 100
    )

    tier_summary[
        "actual_bad_rate"
    ] = (
        tier_summary["actual_bad_loans"]
        / tier_summary["applicants"]
        * 100
    )

    # ========================================================
    # Order tiers
    # ========================================================

    tier_order = [
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]

    tier_summary[
        "risk_tier"
    ] = pd.Categorical(
        tier_summary["risk_tier"],
        categories=tier_order,
        ordered=True
    )

    tier_summary = (
        tier_summary
        .sort_values(
            "risk_tier"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Display
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "RISK TIER ANALYSIS"
    )

    print(
        "========================================"
    )

    print(
        tier_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Binary high-risk analysis
    # ========================================================

    high_risk = (
        probabilities
        >= high_threshold
    ).astype("int8")

    print(
        "\n========================================"
    )

    print(
        "HIGH-RISK CLASSIFICATION"
    )

    print(
        "========================================"
    )

    print(
        f"Threshold: "
        f"{high_threshold}"
    )

    print(
        f"Precision: "
        f"{precision_score(actual, high_risk, zero_division=0):.4f}"
    )

    print(
        f"Recall: "
        f"{recall_score(actual, high_risk, zero_division=0):.4f}"
    )

    print(
        f"F1: "
        f"{f1_score(actual, high_risk, zero_division=0):.4f}"
    )

    # ========================================================
    # Save report
    # ========================================================

    tier_summary.to_csv(
        RISK_REPORT,
        index=False
    )

    print(
        f"\nRisk analysis saved to:"
        f"\n{RISK_REPORT}"
    )


if __name__ == "__main__":
    main()