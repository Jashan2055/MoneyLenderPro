import pandas as pd

from config import CSV_PATH, CHUNK_SIZE
from connection import get_engine
from columns import (
    CUSTOMER_COLUMNS,
    CUSTOMER_RENAME_MAP
)


def load_customers():

    print("========================================")
    print("LOADING CUSTOMERS")
    print("========================================")

    engine = get_engine()

    customer_count = 0

    reader = pd.read_csv(
        CSV_PATH,
        usecols=CUSTOMER_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk_number, customers in enumerate(reader, start=1):

        start_id = customer_count + 1
        end_id = start_id + len(customers)

        customers = customers.copy()

        customers["customer_id"] = range(
            start_id,
            end_id
        )

        customers = customers[
            ["customer_id"] + CUSTOMER_COLUMNS
        ]

        customers.rename(
            columns=CUSTOMER_RENAME_MAP,
            inplace=True
        )

        customers.to_sql(
            name="customers",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000
        )

        customer_count += len(customers)

        if chunk_number % 10 == 0:
            print(
                f"Customers loaded: "
                f"{customer_count:,}"
            )

    print(
        f"Finished loading customers: "
        f"{customer_count:,}"
    )


if __name__ == "__main__":
    load_customers()