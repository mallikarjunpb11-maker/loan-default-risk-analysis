# Loan Default Risk Analysis


An end-to-end credit risk analytics pipeline and decision framework built on 1.3 million closed consumer loan records. This project identifies core drivers of default, develops an interpretable scorecard, and evaluates simulated underwriting policy interventions to reduce charge-offs without impairing origination volume.

---

## Executive Summary & Business Impact

* **Baseline Portfolio Default Rate:** **20.07%** across 1,303,162 mature loans ($1$ = Charged Off/Default, $0$ = Fully Paid).
* **Predictive Scorecard Performance:** Logistic regression model achieved an **ROC-AUC of 0.7018** and captured **67% of all defaults** at a balanced decision threshold.
* **Optimal Policy Recommendation (Policy 4):**
  * **Defaults Prevented:** **68,879 defaults eliminated** (26.33% reduction in charge-offs).
  * **Volume Preserved:** **87.01% of originations retained** (1,133,840 loans).
  * **Portfolio Impact:** Brings overall default rate down from **20.07% to 16.99%**.

---

## Key Analytical Findings

### 1. Statistical Hypothesis Testing
* **Pricing & Debt Burden:** Interest rate is the single strongest continuous discriminator (Welch's $t = 291.10$, $p < 0.001$, Cohen's $d = 0.668$), followed by Debt-to-Income ($t = 121.99$, Cohen's $d = 0.275$).
* **Categorical Dominance:** Internal risk grade exhibits strong associative power with default (Chi-Square $\chi^2 = 89,600.91$, Cramér's $V = 0.2622$), followed by loan term duration (Cramér's $V = 0.1759$). Stated purpose and geographic residence demonstrate weak standalone explanatory power (Cramér's $V < 0.07$).

### 2. Multivariate Odds Ratios (Controlling for Confounders)
* **Risk Grade Multipliers:** Grade B carries 1.81x the odds of default relative to Grade A, scaling monotonically up to **3.56x for Grade G**.
* **Tenure Penalty:** Borrowers on 60-month loan terms have **53.15% higher odds of default** compared to 36-month terms ($\text{OR} = 1.5315$).
* **Residential Instability:** Renters have **42.50% higher default odds** than homeowners with an active mortgage ($\text{OR} = 1.4250$).
* **Income Protection:** Higher annual income acts as an empirical risk buffer ($\text{OR} = 0.9006$ per standard deviation increase).

---

## Underwriting Policy Simulation Matrix

| Policy Strategy | Approved Loans | Volume Retained (%) | Defaults Prevented | Default Reduction (%) | Portfolio Default Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Status Quo (Baseline)** | 1,303,162 | 100.00% | 0 | 0.00% | 20.07% |
| **Policy 1: Reject Grades F & G** | 1,262,753 | 96.90% | 18,735 | 7.16% | 19.23% |
| **Policy 2: Hard DTI Cap at 35%** | 1,270,474 | 97.49% | 10,323 | 3.95% | 19.77% |
| **Policy 3: Ban 60M Terms (Grades D-G)** | 1,142,453 | 87.67% | 65,430 | 25.02% | 17.17% |
| **Policy 4: Combined Optimization** | **1,133,840** | **87.01%** | **68,879** | **26.33%** | **16.99%** |

---

## Project Architecture & Pipeline

```text
├── data/
│   ├── raw/                # Sourced raw datasets (gitignored)
│   └── processed/          # Analytical datasets (cleaned_loans.csv)
├── reports/
│   └── figures/            # Visualizations (ROC curves, policy tradeoffs)
├── src/
│   ├── data/
│   │   ├── inspect_schema.py      # Low-overhead schema inspection
│   │   └── ingest_and_clean.py    # Target engineering and memory-conscious ingestion
│   ├── features/
│   │   ├── segment_default_rates.py  # Segment default rate aggregations
│   │   └── statistical_tests.py      # Parametric/non-parametric & Chi-square inference
│   └── models/
│       ├── train_logistic_model.py   # Balanced scorecard training & odds ratios
│       └── policy_simulation.py      # Underwriting policy simulator
├── .gitignore
├── README.md
└── requirements.txt