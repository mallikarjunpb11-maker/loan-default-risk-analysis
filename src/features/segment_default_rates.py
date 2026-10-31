"""
Portfolio Risk Segmentation and Default Rate Analysis (Question 1)
------------------------------------------------------------------
Quantifies default rate variance across:
- Loan Grade & Subgrade
- Loan Term (Tenure)
- Stated Purpose
- Geographic Footprint (US States)
- Origination Vintage (Issue Year)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual formatting standards
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

PROCESSED_PATH = os.path.join("data", "processed", "cleaned_loans.csv")
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_cleaned_portfolio(filepath: str) -> pd.DataFrame:
    print(f"[*] Loading processed portfolio from: {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df):,} loans. Overall Default Rate: {df['is_default'].mean():.2%}")
    return df


def calculate_segment_metrics(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Computes total loan count, default count, default rate, and share of total volume."""
    grouped = df.groupby(segment_col).agg(
        total_loans=("is_default", "count"),
        defaults=("is_default", "sum"),
        default_rate=("is_default", "mean"),
        avg_int_rate=("int_rate", "mean"),
        avg_dti=("dti", "mean")
    ).reset_index()

    grouped["volume_share"] = grouped["total_loans"] / len(df)
    grouped = grouped.sort_values(by="default_rate", ascending=False)
    return grouped


def analyze_and_plot(df: pd.DataFrame):
    overall_dr = df["is_default"].mean()

    # -------------------------------------------------------------
    # 1. Default Rate by Grade
    # -------------------------------------------------------------
    grade_df = calculate_segment_metrics(df, "grade").sort_values(by="grade")
    print("\n" + "=" * 65)
    print("LOAN GRADE BREAKDOWN:")
    print("=" * 65)
    print(grade_df[["grade", "total_loans", "volume_share", "default_rate", "avg_int_rate"]].to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    
    sns.barplot(data=grade_df, x="grade", y="total_loans", ax=ax1, color="#4C72B0", alpha=0.6)
    sns.lineplot(data=grade_df, x="grade", y="default_rate", ax=ax2, color="#C44E52", marker="o", linewidth=2.5)
    
    ax1.set_ylabel("Origination Volume (Loans)", color="#4C72B0", fontweight="bold")
    ax2.set_ylabel("Default Rate", color="#C44E52", fontweight="bold")
    ax2.axhline(overall_dr, color="black", linestyle="--", alpha=0.7, label=f"Portfolio Mean ({overall_dr:.1%})")
    ax2.yaxis.set_major_formatter("{x:.0%}")
    plt.title("Origination Volume vs. Default Rate by Loan Grade", fontsize=14, fontweight="bold", pad=15)
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "default_rate_by_grade.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Default Rate by Term (Tenure)
    # -------------------------------------------------------------
    term_df = calculate_segment_metrics(df, "term_months")
    print("\n" + "=" * 65)
    print("LOAN TERM BREAKDOWN (Months):")
    print("=" * 65)
    print(term_df[["term_months", "total_loans", "volume_share", "default_rate", "avg_int_rate"]].to_string(index=False))

    # -------------------------------------------------------------
    # 3. Default Rate by Stated Purpose
    # -------------------------------------------------------------
    purpose_df = calculate_segment_metrics(df, "purpose").sort_values(by="total_loans", ascending=False)
    print("\n" + "=" * 65)
    print("LOAN PURPOSE BREAKDOWN (Ranked by Volume):")
    print("=" * 65)
    print(purpose_df[["purpose", "total_loans", "volume_share", "default_rate"]].to_string(index=False))

    plt.figure(figsize=(10, 6))
    purpose_plot = purpose_df.sort_values(by="default_rate", ascending=True)
    plt.barh(purpose_plot["purpose"], purpose_plot["default_rate"], color="#55A868", edgecolor="black", alpha=0.8)
    plt.axvline(overall_dr, color="#C44E52", linestyle="--", linewidth=2, label=f"Portfolio Baseline ({overall_dr:.1%})")
    plt.gca().xaxis.set_major_formatter("{x:.0%}")
    plt.xlabel("Default Rate", fontweight="bold")
    plt.title("Default Rate Dispersion Across Stated Loan Purposes", fontsize=13, fontweight="bold", pad=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "default_rate_by_purpose.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Default Rate by Origination Vintage (Issue Year)
    # -------------------------------------------------------------
    if "issue_year" in df.columns:
        vintage_df = calculate_segment_metrics(df, "issue_year").sort_values(by="issue_year")
        print("\n" + "=" * 65)
        print("VINTAGE BREAKDOWN (Origination Year):")
        print("=" * 65)
        print(vintage_df[["issue_year", "total_loans", "volume_share", "default_rate"]].to_string(index=False))

    # -------------------------------------------------------------
    # 5. Geographic Concentration (Top 10 Risk States, min 5,000 loans)
    # -------------------------------------------------------------
    if "addr_state" in df.columns:
        state_df = calculate_segment_metrics(df, "addr_state")
        state_df_filtered = state_df[state_df["total_loans"] >= 5000].sort_values(by="default_rate", ascending=False)
        print("\n" + "=" * 65)
        print("TOP 10 RISKIEST STATES (Min 5,000 Originations):")
        print("=" * 65)
        print(state_df_filtered.head(10)[["addr_state", "total_loans", "default_rate"]].to_string(index=False))


if __name__ == "__main__":
    loans_df = load_cleaned_portfolio(PROCESSED_PATH)
    analyze_and_plot(loans_df)