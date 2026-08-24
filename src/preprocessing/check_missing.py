from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import text

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from database.connection import get_engine

engine = get_engine()

# Get loan table columns
with engine.connect() as conn:
    result = conn.execute(
        text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'micro_lending'
              AND table_name = 'loans'
            ORDER BY ordinal_position
        """)
    )

    columns = [row[0] for row in result]


# Build missing-value query
expressions = [
    f"SUM(`{column}` IS NULL) AS `{column}`"
    for column in columns
]

query = f"""
SELECT
    {", ".join(expressions)}
FROM loans
"""


print("Checking missing values...")
print("This may take a little while because the table has 2.26M rows.\n")


with engine.connect() as conn:
    result = conn.execute(text(query))
    row = result.fetchone()


total_rows = 2_260_668

missing_data = []

for column, count in zip(columns, row):

    if count > 0:
        missing_data.append({
            "column": column,
            "missing_count": count,
            "missing_percentage": round(
                count / total_rows * 100,
                2
            )
        })


missing_df = pd.DataFrame(missing_data)


if missing_df.empty:

    print("No missing values found.")

else:

    missing_df = missing_df.sort_values(
        "missing_count",
        ascending=False
    )

    print("Missing-value summary:\n")
    print(missing_df.to_string(index=False))

    # Save the report
    output_file = BASE_DIR / "src" / "preprocessing" / "missing_values_report.csv"

    missing_df.to_csv(
        output_file,
        index=False
    )

    print(f"\nReport saved to:")
    print(output_file)