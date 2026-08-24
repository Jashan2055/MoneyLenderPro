import pandas as pd

from config import CSV_PATH, CHUNK_SIZE
from connection import get_engine
from columns import (
    LOAN_COLUMNS,
    LOAN_RENAME_MAP
)


START_ROW = 1


def load_loans(start_row=0):

    print("========================================")
    print("LOADING LOANS")
    print("========================================")

    engine = get_engine()

    loan_count = start_row

    if start_row == 0:

        reader = pd.read_csv(
            CSV_PATH,
            usecols=LOAN_COLUMNS,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

    else:

        reader = pd.read_csv(
            CSV_PATH,
            usecols=LOAN_COLUMNS,
            skiprows=range(1, start_row + 1),
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

    for chunk_number, chunk in enumerate(reader, start=1):

        row_count = len(chunk)

        start_id = loan_count + 1
        end_id = start_id + row_count

        ids = pd.DataFrame({
            "loan_id": range(start_id, end_id),
            "customer_id": range(start_id, end_id)
        })

        loans = pd.concat(
            [
                ids,
                chunk.reset_index(drop=True)
            ],
            axis=1
        )

        loans.rename(
            columns=LOAN_RENAME_MAP,
            inplace=True
        )

        loans.to_sql(
            name="loans",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000
        )

        loan_count += row_count

        if chunk_number % 10 == 0:
            print(
                f"Loans loaded: "
                f"{loan_count:,}"
            )

    print(
        f"Finished loading loans: "
        f"{loan_count:,}"
    )


if __name__ == "__main__":
    load_loans(start_row=START_ROW)