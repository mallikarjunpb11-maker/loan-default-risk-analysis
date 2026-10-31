"""
Underwriting Policy Simulator & Risk-Return Optimization (Question 6)
---------------------------------------------------------------------
Simulates 4 targeted underwriting rule adjustments on the cleaned portfolio:
  - Policy 0: Status Quo (Baseline)
  - Policy 1: Surgical Grade Cut (Reject Grades F & G entirely)
  - Policy 2: Debt-to-Income Cap (Cap maximum DTI at 35%)
  - Policy 3: High-Risk Term Restriction (Ban 60-month loans for Grades D & lower)
  - Policy 4: Recommended Combined Policy (Policy 1 + Policy 3)
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_PATH = os.path.join("data", "processed", "cleaned_loans.csv")
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    print(f"[*] Loading dataset from {filepath}...")
    return pd.read_csv(filepath)


def evaluate_policy(df: pd.DataFrame, policy_name: str, mask: pd.Series, baseline_loans: int, baseline_defaults: int):
    """Calculates volume retained, default reduction, and new default rate under an underwriting rule."""
    approved_df = df[mask]
    approved_loans = len(approved_df)
    approved_defaults = approved_df["is_default"].sum()
    new_default_rate = approved_df["is_default"].mean()

    volume_retained_pct = (approved_loans / baseline_loans) * 100
    volume_dropped_pct = 100 - volume_retained_pct
    defaults_prevented = baseline_defaults - approved_defaults
    default_reduction_pct = (defaults_prevented / baseline_defaults) * 100

    return {
        "Policy": policy_name,
        "Approved_Loans": approved_loans,
        "Volume_Retained_%": round(volume_retained_pct, 2),
        "Volume_Cut_%": round(volume_dropped_pct, 2),
        "Defaults_Prevented": defaults_prevented,
        "Default_Reduction_%": round(default_reduction_pct, 2),
        "New_Default_Rate_%": round(new_default_rate * 100, 2)
    }


def run_simulations(df: pd.DataFrame):
    baseline_loans = len(df)
    baseline_defaults = df["is_default"].sum()
    baseline_dr = df["is_default"].mean() * 100

    print("\n" + "=" * 75)
    print(f"BASELINE: {baseline_loans:,} loans | {baseline_defaults:,} defaults | {baseline_dr:.2f}% default rate")
    print("=" * 75)

    # Define Underwriting Filter Masks
    policies = [
        ("Status Quo (No Change)", pd.Series(True, index=df.index)),
        ("Policy 1: Reject Grades F & G", ~df["grade"].isin(["F", "G"])),
        ("Policy 2: Hard DTI Cap at 35%", df["dti"] <= 35),
        ("Policy 3: Ban 60-Month Terms for Grades D, E, F, G", ~((df["grade"].isin(["D", "E", "F", "G"])) & (df["term_months"] == 60))),
        ("Policy 4: Combined Optimization (P1 + P3)", (~df["grade"].isin(["F", "G"])) & ~((df["grade"].isin(["D", "E"])) & (df["term_months"] == 60)))
    ]

    results = []
    for name, mask in policies:
        res = evaluate_policy(df, name, mask, baseline_loans, baseline_defaults)
        results.append(res)

    results_df = pd.DataFrame(results)
    print("\nUNDERWRITING POLICY SIMULATION MATRIX:")
    print("=" * 75)
    print(results_df.to_string(index=False))

    # -------------------------------------------------------------
    # Visual Comparison: Volume Retained vs Default Reduction
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    x = range(len(results_df))
    plt.bar([i - 0.2 for i in x], results_df["Volume_Retained_%"], width=0.4, label="Volume Retained (%)", color="#4C72B0")
    plt.bar([i + 0.2 for i in x], results_df["Default_Reduction_%"], width=0.4, label="Defaults Prevented (%)", color="#C44E52")
    plt.xticks(x, [p.split(":")[0] for p in results_df["Policy"]], rotation=15, ha="right", fontweight="bold")
    plt.ylabel("Percentage (%)", fontweight="bold")
    plt.title("Trade-Off: Origination Volume Retained vs Defaults Prevented", fontsize=13, fontweight="bold", pad=12)
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, "underwriting_policy_tradeoff.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n[+] Saved policy trade-off chart to {chart_path}")


if __name__ == "__main__":
    loans = load_data(PROCESSED_PATH)
    run_simulations(loans)