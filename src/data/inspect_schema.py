"""
Schema Inspector: Reads row 1 of raw CSV to list column names and index numbers.
"""
import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "loan.csv")

def inspect():
    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] File not found: {RAW_PATH}")
        return

    # Read row 0 (headers only) without loading data rows
    header_df = pd.read_csv(RAW_PATH, nrows=0)
    columns = header_df.columns.tolist()

    print(f"Total columns in file: {len(columns)}")
    print("-" * 45)
    for index, col_name in enumerate(columns, start=1):
        print(f"{index:02d}. {col_name}")

if __name__ == "__main__":
    inspect()