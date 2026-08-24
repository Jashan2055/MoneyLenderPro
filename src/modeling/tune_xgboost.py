from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
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
REPORT_DIR = BASE_DIR / "data" / "reports"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_final.json"
)

RESULTS_FILE = (
    REPORT_DIR
    / "xgboost_tuning_results.csv"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "risk_label"

CHUNK_SIZE = 50_000

SAMPLE_SIZE = 300_000

RANDOM_STATE = 42


# ============================================================
# Feature removal
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

    return df


# ============================================================
# Create representative random sample
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
    # Pass 1: count rows
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
    # Select random row positions
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

    # --------------------------------------------------------
    # Pass 2: retrieve selected rows
    # --------------------------------------------------------

    sampled_chunks = []

    current_start = 0

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

        mask = (
            (selected_rows >= current_start)
            &
            (selected_rows < current_end)
        )

        chunk_selected = (
            selected_rows[mask]
        )

        if len(chunk_selected) > 0:

            local_indices = (
                chunk_selected
                - current_start
            )

            sampled_chunks.append(
                df.iloc[local_indices]
            )

        current_start = current_end

        del df

    sample = pd.concat(
        sampled_chunks,
        ignore_index=True
    )

    del sampled_chunks

    # --------------------------------------------------------
    # Shuffle
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
        "Starting XGBoost hyperparameter tuning..."
    )

    # ========================================================
    # Load representative training sample
    # ========================================================

    train = create_training_sample()

    # ========================================================
    # Target
    # ========================================================

    y = (
        train[TARGET]
        .astype("int8")
    )

    X = train.drop(
        columns=[TARGET]
    )

    # ========================================================
    # Remove grade/sub-grade
    # ========================================================

    X = remove_grade_features(
        X
    )

    feature_columns = (
        X.columns.tolist()
    )

    print(
        f"\nFeatures after removing grade:"
        f" {len(feature_columns)}"
    )

    X = X.astype(
        "float32"
    )

    # ========================================================
    # Class weight
    # ========================================================

    negative = (
        y == 0
    ).sum()

    positive = (
        y == 1
    ).sum()

    scale_pos_weight = (
        negative / positive
    )

    print(
        f"\nScale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================
    # Train / validation split
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "CREATING VALIDATION SPLIT"
    )

    print(
        "========================================"
    )

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_valid):,}"
    )

    # ========================================================
    # Hyperparameter configurations
    # ========================================================

    configurations = [

        {
            "name": "baseline",
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },

        {
            "name": "shallower",
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },

        {
            "name": "deeper",
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },

        {
            "name": "higher_min_child",
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 10,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },

        {
            "name": "more_sampling",
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.70,
            "colsample_bytree": 0.70,
        },

        {
            "name": "slower_learning",
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
        },
    ]

    results = []

    # ========================================================
    # Tune
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "XGBOOST HYPERPARAMETER TUNING"
    )

    print(
        "========================================"
    )

    for i, config in enumerate(
        configurations,
        start=1
    ):

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Configuration {i}/"
            f"{len(configurations)}"
        )

        print(
            f"Name: {config['name']}"
        )

        print(
            f"max_depth: "
            f"{config['max_depth']}"
        )

        print(
            f"learning_rate: "
            f"{config['learning_rate']}"
        )

        print(
            f"min_child_weight: "
            f"{config['min_child_weight']}"
        )

        print(
            f"subsample: "
            f"{config['subsample']}"
        )

        print(
            f"colsample_bytree: "
            f"{config['colsample_bytree']}"
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = xgb.XGBClassifier(

            n_estimators=300,

            max_depth=config[
                "max_depth"
            ],

            learning_rate=config[
                "learning_rate"
            ],

            min_child_weight=config[
                "min_child_weight"
            ],

            subsample=config[
                "subsample"
            ],

            colsample_bytree=config[
                "colsample_bytree"
            ],

            objective="binary:logistic",

            eval_metric="auc",

            scale_pos_weight=(
                scale_pos_weight
            ),

            tree_method="hist",

            max_bin=256,

            random_state=RANDOM_STATE,

            n_jobs=4
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_valid, y_valid)
            ],
            verbose=False
        )

        # ----------------------------------------------------
        # Validation predictions
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype("int8")

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_valid,
            predictions
        )

        precision = precision_score(
            y_valid,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_valid,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_valid,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_valid,
            probabilities
        )

        print(
            f"\nValidation ROC-AUC: "
            f"{roc_auc:.4f}"
        )

        print(
            f"Validation F1: "
            f"{f1:.4f}"
        )

        print(
            f"Validation Recall: "
            f"{recall:.4f}"
        )

        results.append({

            "name": config["name"],

            "max_depth": config[
                "max_depth"
            ],

            "learning_rate": config[
                "learning_rate"
            ],

            "min_child_weight": config[
                "min_child_weight"
            ],

            "subsample": config[
                "subsample"
            ],

            "colsample_bytree": config[
                "colsample_bytree"
            ],

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": roc_auc
        })

        del model

    # ========================================================
    # Results
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "roc_auc",
        ascending=False
    ).reset_index(
        drop=True
    )

    print(
        "\n========================================"
    )

    print(
        "TUNING RESULTS"
    )

    print(
        "========================================"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Save tuning results
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nTuning results saved to:"
        f"\n{RESULTS_FILE}"
    )

    # ========================================================
    # Best configuration
    # ========================================================

    best = results_df.iloc[0]

    print(
        "\n========================================"
    )

    print(
        "BEST CONFIGURATION"
    )

    print(
        "========================================"
    )

    print(
        best.to_string()
    )

    # ========================================================
    # Retrain best model on FULL 300k sample
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "TRAINING FINAL MODEL"
    )

    print(
        "========================================"
    )

    final_model = xgb.XGBClassifier(

        n_estimators=300,

        max_depth=int(
            best["max_depth"]
        ),

        learning_rate=float(
            best["learning_rate"]
        ),

        min_child_weight=int(
            best["min_child_weight"]
        ),

        subsample=float(
            best["subsample"]
        ),

        colsample_bytree=float(
            best["colsample_bytree"]
        ),

        objective="binary:logistic",

        eval_metric="auc",

        scale_pos_weight=(
            scale_pos_weight
        ),

        tree_method="hist",

        max_bin=256,

        random_state=RANDOM_STATE,

        n_jobs=4
    )

    final_model.fit(
        X,
        y,
        verbose=True
    )

    # ========================================================
    # Save final model
    # ========================================================

    final_model.save_model(
        MODEL_FILE
    )

    print(
        f"\nFinal model saved to:"
        f"\n{MODEL_FILE}"
    )

    # ========================================================
    # Free training data
    # ========================================================

    del train
    del X
    del y
    del X_train
    del X_valid
    del y_train
    del y_valid

    # ========================================================
    # Evaluate on COMPLETE test set
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "========================================"
    )

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

        X_test = remove_grade_features(
            X_test
        )

        X_test = X_test[
            feature_columns
        ]

        X_test = X_test.astype(
            "float32"
        )

        probabilities = (
            final_model.predict_proba(
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
    # Final metrics
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
    # Final results
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL XGBOOST RESULTS"
    )

    print(
        "========================================"
    )

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

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print(
        "\nClassification Report:"
    )

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
        final_model.feature_importances_,
        index=feature_columns
    ).sort_values(
        ascending=False
    )

    importance_file = (
        REPORT_DIR
        / "final_xgboost_feature_importance.csv"
    )

    importance.to_csv(
        importance_file,
        header=["importance"]
    )

    print(
        "\nTop 20 features:"
    )

    print(
        importance.head(20)
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{importance_file}"
    )


if __name__ == "__main__":
    main()