"""
Data Ingestion and Target Engineering Pipeline
Project: Early-Stage Loan Defaults & Risk Segmentation
"""
import os
import numpy as np
import pandas as pd

# The 12 columns we picked from inspect_schema.py
COLS = [
    "loan_status",     # Target (Fully Paid vs Charged Off)
    "loan_amnt",       # Loan amount ($)
    "term",            # Loan duration (36 or 60 months)
    "int_rate",        # Interest rate (%)
    "grade",           # Risk grade (A through G)
    "sub_grade",       # Detailed risk grade (A1 to G5)
    "annual_inc",      # Annual borrower income ($)
    "dti",             # Debt-to-income ratio (%)
    "home_ownership",  # RENT, OWN, MORTGAGE
    "purpose",         # Reason for loan
    "addr_state",      # Borrower US state
    "issue_d"          # Month/year loan was issued
]

def clean_data():
    raw_path = os.path.join("data", "raw", "loan.csv")
    out_path = os.path.join("data", "processed", "cleaned_loans.csv")

    print("[1/5] Reading relevant columns from data/raw/loan.csv...")
    preview = pd.read_csv(raw_path, nrows=5)
    cols_to_use = [c for c in COLS if c in preview.columns]
    df = pd.read_csv(raw_path, usecols=cols_to_use, low_memory=False)
    print(f"      Loaded {len(df):,} total loan records.")

    print("[2/5] Filtering finished loans (Fully Paid vs Charged Off)...")
    target_map = {
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1
    }
    df = df[df["loan_status"].isin(target_map.keys())].copy()
    df["is_default"] = df["loan_status"].map(target_map).astype(int)
    print(f"      Kept {len(df):,} completed loans.")
    print(f"      Portfolio Default Rate: {df['is_default'].mean():.2%}")

    print("[3/5] Converting text columns into numbers...")
    if "term" in df.columns:
        df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    if "int_rate" in df.columns and df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False).str.strip().astype(float)
    if "issue_d" in df.columns:
        df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year

    print("[4/5] Handling outliers and missing values...")
    if "home_ownership" in df.columns:
        valid_homes = ["RENT", "MORTGAGE", "OWN"]
        df["home_ownership"] = df["home_ownership"].apply(lambda x: x if x in valid_homes else "OTHER")

    for col in ["annual_inc", "dti", "int_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "dti" in df.columns:
        df = df[df["dti"] <= 100].copy()

    if "annual_inc" in df.columns:
        income_cap = df["annual_inc"].quantile(0.999)
        df["annual_inc"] = np.clip(df["annual_inc"], 0, income_cap)

    print("[5/5] Saving cleaned file...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Clean dataset saved to {out_path} ({len(df):,} rows)")

if __name__ == "__main__":
    clean_data()