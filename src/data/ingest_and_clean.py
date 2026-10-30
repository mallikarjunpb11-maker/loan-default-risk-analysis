"""
Data Ingestion and Target Engineering Pipeline
----------------------------------------------
Project: Early-Stage Loan Defaults & Risk Segmentation
"""

import os
import numpy as np
import pandas as pd

# Core credit risk features needed for statistical testing and segmentation
COLS = [
    "loan_status",     # Primary target (Fully Paid vs Charged Off)
    "loan_amnt",       # Loan amount requested ($)
    "term",            # Loan tenure (' 36 months' or ' 60 months')
    "int_rate",        # Annual interest rate (%)
    "grade",           # Risk grade (A through G)
    "sub_grade",       # Granular risk sub-grade (A1 through G5)
    "annual_inc",      # Self-reported annual income ($)
    "dti",             # Debt-to-income ratio (%)
    "home_ownership",  # RENT, OWN, MORTGAGE, etc.
    "purpose",         # Reason for loan (debt_consolidation, credit_card, etc.)
    "addr_state",      # US borrower residence state
    "issue_d"          # Origination month-year (e.g., 'Dec-2015')
]


def clean_data():
    raw_path = os.path.join("data", "raw", "loan.csv")
    out_path = os.path.join("data", "processed", "cleaned_loans.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Source file not found at: {raw_path}")

    print("Step 1: Reading relevant columns from data/raw/loan.csv...")
    # Read only the 12 relevant columns to prevent out-of-memory crashes
    preview = pd.read_csv(raw_path, nrows=5)
    cols_to_use = [c for c in COLS if c in preview.columns]
    df = pd.read_csv(raw_path, usecols=cols_to_use, low_memory=False)
    print(f"-> Loaded {len(df):,} total loan records across {len(cols_to_use)} columns.")

    print("\nStep 2: Subsetting to closed loans and defining binary target...")
    # Isolate concluded loans: exclude active/ongoing loans ('Current', 'In Grace Period')
    target_map = {
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1
    }
    df = df[df["loan_status"].isin(target_map.keys())].copy()
    df["is_default"] = df["loan_status"].map(target_map).astype(int)

    default_rate = df["is_default"].mean()
    print(f"-> Concluded loans retained: {len(df):,}")
    print(f"-> Baseline Portfolio Default Rate: {default_rate:.2%}")

    print("\nStep 3: Cleaning formatted text fields into numeric values...")
    # 1. Term: extract integer months from ' 36 months' or ' 60 months'
    if "term" in df.columns:
        df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    # 2. Interest rate: strip '%' if stored as string, convert to float
    if "int_rate" in df.columns and df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False).str.strip().astype(float)

    # 3. Extract origination year from issue date for vintage analysis
    if "issue_d" in df.columns:
        df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year

    print("\nStep 4: Handling missing values and extreme leverage points...")
    # Standardize rare home ownership entries
    if "home_ownership" in df.columns:
        valid_homes = ["RENT", "MORTGAGE", "OWN"]
        df["home_ownership"] = df["home_ownership"].apply(lambda x: x if x in valid_homes else "OTHER")

    # Median imputation for continuous financial metrics
    for col in ["annual_inc", "dti", "int_rate"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Remove non-physical DTI entries (>100%) and cap extreme incomes at the 99.9th percentile
    if "dti" in df.columns:
        df = df[df["dti"] <= 100].copy()

    if "annual_inc" in df.columns:
        income_cap = df["annual_inc"].quantile(0.999)
        df["annual_inc"] = np.clip(df["annual_inc"], 0, income_cap)

    print("\nStep 5: Exporting processed file...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"-> SUCCESS! Clean dataset saved to: {out_path} ({len(df):,} rows)")


if __name__ == "__main__":
    clean_data()