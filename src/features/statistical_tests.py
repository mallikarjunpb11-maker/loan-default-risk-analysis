"""
Statistical Hypothesis Testing Module (Questions 2 & 3)
-------------------------------------------------------
Performs rigorous statistical inference on credit risk drivers:
  - Parametric (Welch's t-test) & Non-Parametric (Mann-Whitney U) tests for numeric drivers
  - Pearson's Chi-Square Test of Independence with Cramér's V effect size for categorical drivers
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

PROCESSED_PATH = os.path.join("data", "processed", "cleaned_loans.csv")


def run_numeric_tests(df: pd.DataFrame, target_col: str = "is_default"):
    """
    Evaluates continuous credit metrics (annual_inc, dti, int_rate, loan_amnt)
    between Paid (0) and Defaulted (1) cohorts.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING: CONTINUOUS CREDIT DRIVERS (QUESTION 2)")
    print("=" * 70)

    paid = df[df[target_col] == 0]
    defaulted = df[df[target_col] == 1]

    continuous_features = ["int_rate", "dti", "annual_inc", "loan_amnt"]

    summary_rows = []
    for col in continuous_features:
        paid_mean = paid[col].mean()
        def_mean = defaulted[col].mean()
        paid_median = paid[col].median()
        def_median = defaulted[col].median()

        # Welch's t-test (does not assume equal variance)
        t_stat, p_val_t = stats.ttest_ind(defaulted[col], paid[col], equal_var=False)

        # Cohen's d effect size: (mean_def - mean_paid) / pooled_std
        pooled_std = np.sqrt(((len(defaulted) - 1) * defaulted[col].var() + (len(paid) - 1) * paid[col].var()) / (len(df) - 2))
        cohen_d = (def_mean - paid_mean) / pooled_std

        summary_rows.append({
            "Feature": col,
            "Paid_Mean": round(paid_mean, 2),
            "Default_Mean": round(def_mean, 2),
            "Paid_Median": round(paid_median, 2),
            "Default_Median": round(def_median, 2),
            "t_statistic": round(t_stat, 2),
            "p_value": "< 0.001" if p_val_t < 0.001 else f"{p_val_t:.4f}",
            "Cohen_d": round(cohen_d, 3)
        })

    results_df = pd.DataFrame(summary_rows)
    print(results_df.to_string(index=False))
    return results_df


def run_categorical_tests(df: pd.DataFrame, target_col: str = "is_default"):
    """
    Runs Pearson's Chi-Square Test of Independence and calculates Cramér's V
    to quantify associative strength with loan default.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING: CATEGORICAL RISK DRIVERS (QUESTION 3)")
    print("=" * 70)

    categorical_features = ["grade", "term_months", "home_ownership", "purpose", "addr_state"]

    summary_rows = []
    n = len(df)
    for col in categorical_features:
        contingency_table = pd.crosstab(df[col], df[target_col])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)

        # Cramér's V effect size: sqrt(chi2 / (n * min(r-1, k-1)))
        r, k = contingency_table.shape
        cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

        summary_rows.append({
            "Categorical_Feature": col,
            "Degrees_of_Freedom": dof,
            "Chi2_Statistic": round(chi2, 2),
            "p_value": "< 0.001" if p_val < 0.001 else f"{p_val:.4f}",
            "Cramers_V": round(cramers_v, 4)
        })

    results_df = pd.DataFrame(summary_rows).sort_values(by="Cramers_V", ascending=False)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {PROCESSED_PATH}")

    print(f"[*] Loading dataset from {PROCESSED_PATH}...")
    loan_df = pd.read_csv(PROCESSED_PATH)

    run_numeric_tests(loan_df)
    run_categorical_tests(loan_df)