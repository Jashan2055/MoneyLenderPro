from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    BASE_DIR
    / "data"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Main
# ============================================================

def main():

    print("\n========================================")
    print("FINAL MODEL EVALUATION REPORT")
    print("========================================")

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    model_comparison = pd.DataFrame({

        "model": [
            "Logistic Regression",
            "XGBoost",
            "XGBoost without Grade",
            "Tuned XGBoost"
        ],

        "roc_auc": [
            0.6415,
            0.7266,
            0.7298,
            0.7299
        ],

        "accuracy": [
            0.6317,
            0.6764,
            0.6678,
            0.6678
        ],

        "precision": [
            0.2833,
            0.3386,
            0.3349,
            0.3352
        ],

        "recall": [
            0.5448,
            0.6408,
            0.6631,
            0.6643
        ],

        "f1": [
            0.3728,
            0.4431,
            0.4451,
            0.4455
        ]
    })

    print("\nMODEL COMPARISON")
    print("----------------")

    print(
        model_comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Risk tier analysis
    # --------------------------------------------------------

    risk_file = (
        REPORT_DIR
        / "risk_tier_analysis.csv"
    )

    if risk_file.exists():

        risk_tiers = pd.read_csv(
            risk_file
        )

        print(
            "\nRISK TIER ANALYSIS"
        )

        print(
            "------------------"
        )

        print(
            risk_tiers.to_string(
                index=False
            )
        )

    else:

        print(
            "\nRisk tier report not found."
        )

        risk_tiers = None

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_file = (
        REPORT_DIR
        / "final_xgboost_feature_importance.csv"
    )

    if importance_file.exists():

        importance = pd.read_csv(
            importance_file
        )

        print(
            "\nTOP 20 FEATURES"
        )

        print(
            "----------------"
        )

        print(
            importance.head(20)
            .to_string(index=False)
        )

    else:

        print(
            "\nFeature importance report not found."
        )

    # --------------------------------------------------------
    # Save model comparison
    # --------------------------------------------------------

    comparison_file = (
        REPORT_DIR
        / "model_comparison.csv"
    )

    model_comparison.to_csv(
        comparison_file,
        index=False
    )

    print(
        f"\nModel comparison saved to:"
        f"\n{comparison_file}"
    )

    # --------------------------------------------------------
    # Create markdown report
    # --------------------------------------------------------

    report_file = (
        REPORT_DIR
        / "final_model_report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Micro-Lending Credit Risk Model\n\n"
        )

        f.write(
            "## Dataset\n\n"
        )

        f.write(
            "- Original dataset: 2,260,668 loans\n"
            "- Training rows: 1,045,109\n"
            "- Test rows: 261,278\n"
            "- Final training sample: 300,000\n\n"
        )

        f.write(
            "## Model Comparison\n\n"
        )

        f.write(
            model_comparison.to_markdown(
                index=False
            )
        )

        f.write(
            "\n\n"
        )

        f.write(
            "## Final Model\n\n"
        )

        f.write(
            "The final model is an XGBoost binary "
            "classification model using 162 engineered "
            "and preprocessed features. LendingClub "
            "grade and sub-grade features were excluded "
            "from the final model.\n\n"
        )

        f.write(
            "### Final Performance\n\n"
        )

        f.write(
            "- ROC-AUC: **0.7299**\n"
            "- Accuracy: **0.6678**\n"
            "- Precision: **0.3352**\n"
            "- Recall: **0.6643**\n"
            "- F1 Score: **0.4455**\n\n"
        )

        f.write(
            "## Risk Tier Analysis\n\n"
        )

        if risk_tiers is not None:

            f.write(
                risk_tiers.to_markdown(
                    index=False
                )
            )

            f.write(
                "\n\n"
            )

        f.write(
            "## Business Interpretation\n\n"
        )

        f.write(
            "The model produces a probability representing "
            "the estimated risk of a loan being classified "
            "as a bad loan. These probabilities can be "
            "grouped into risk tiers to support lending "
            "decisions and additional manual review.\n\n"
        )

        f.write(
            "The observed bad-loan rate increases "
            "substantially across the Low, Medium, and "
            "High Risk groups, indicating that the model "
            "provides useful risk segmentation.\n\n"
        )

        f.write(
            "## Important Features\n\n"
        )

        if importance_file.exists():

            for _, row in importance.head(
                20
            ).iterrows():

                f.write(
                    f"- {row.iloc[0]}\n"
                )

        f.write(
            "\n## Conclusion\n\n"
        )

        f.write(
            "The final XGBoost model provides a meaningful "
            "baseline for automated credit-risk assessment. "
            "Its probability outputs can be used to segment "
            "loan applications into risk tiers and support "
            "risk-aware lending decisions.\n"
        )

    print(
        f"\nFinal report saved to:"
        f"\n{report_file}"
    )

    print(
        "\n========================================"
    )

    print(
        "REPORT GENERATION COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()