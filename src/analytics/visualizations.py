from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_features.csv"
)

REPORT_DIR = BASE_DIR / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50_000
SAMPLE_PER_CHUNK = 2_000

samples = []

print("Collecting EDA sample...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):
    sample = chunk.sample(
        n=min(SAMPLE_PER_CHUNK, len(chunk)),
        random_state=42
    )

    samples.append(sample)

    print(f"Processed chunk {chunk_number}")

    del chunk


df = pd.concat(
    samples,
    ignore_index=True
)

print(f"\nEDA sample size: {len(df):,}")


# ============================================================
# 1. Risk distribution
# ============================================================

risk_counts = (
    df["risk_label"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 5))

plt.bar(
    ["Good Loan", "Bad Loan"],
    risk_counts.values
)

plt.title("Good vs Bad Loan Distribution")
plt.ylabel("Number of Loans")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "risk_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 2. Loan amount vs risk
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="loan_amount",
    by="risk_label"
)

plt.title("Loan Amount by Risk")
plt.suptitle("")
plt.xlabel("Risk Label")
plt.ylabel("Loan Amount")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "loan_amount_by_risk.png",
    dpi=150
)

plt.close()


# ============================================================
# 3. Interest rate vs risk
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="interest_rate",
    by="risk_label"
)

plt.title("Interest Rate by Risk")
plt.suptitle("")
plt.xlabel("Risk Label")
plt.ylabel("Interest Rate")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "interest_rate_by_risk.png",
    dpi=150
)

plt.close()


# ============================================================
# 4. DTI vs risk
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="dti",
    by="risk_label"
)

plt.title("Debt-to-Income Ratio by Risk")
plt.suptitle("")
plt.xlabel("Risk Label")
plt.ylabel("DTI")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "dti_by_risk.png",
    dpi=150
)

plt.close()


# ============================================================
# 5. Grade vs default rate
# ============================================================

grade_risk = (
    df.groupby("grade")["risk_label"]
    .mean()
    .sort_index()
    * 100
)

plt.figure(figsize=(8, 5))

plt.bar(
    grade_risk.index,
    grade_risk.values
)

plt.title("Bad Loan Rate by Loan Grade")
plt.xlabel("Grade")
plt.ylabel("Bad Loan Rate (%)")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "bad_rate_by_grade.png",
    dpi=150
)

plt.close()


# ============================================================
# 6. Revolving utilization vs risk
# ============================================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="revol_util",
    by="risk_label"
)

plt.title("Revolving Utilization by Risk")
plt.suptitle("")
plt.xlabel("Risk Label")
plt.ylabel("Revolving Utilization (%)")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "revol_util_by_risk.png",
    dpi=150
)

plt.close()


print("\n========================================")
print("VISUALIZATION COMPLETE")
print("========================================")

print(f"Plots saved to:")
print(REPORT_DIR)