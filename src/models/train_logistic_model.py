"""
Credit Risk Logistic Regression Model (Questions 4 & 5)
-------------------------------------------------------
Builds an interpretable credit scoring model:
  - Stratified 80/20 train/test split
  - Standardization of continuous features
  - Logistic regression with balanced class weighting
  - Odds Ratio extraction for credit risk attribution
  - Model evaluation: ROC-AUC, PR-AUC, Confusion Matrix
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED_PATH = os.path.join("data", "processed", "cleaned_loans.csv")
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def prepare_modeling_data(filepath: str):
    print(f"[*] Loading dataset from: {filepath}...")
    df = pd.read_csv(filepath)

    # Core modeling predictors based on hypothesis testing results
    numeric_features = ["int_rate", "dti", "annual_inc", "loan_amnt"]
    categorical_features = ["grade", "term_months", "home_ownership"]

    # Filter complete cases
    model_df = df[numeric_features + categorical_features + ["is_default"]].dropna().copy()

    # One-hot encode categorical features (drop first to prevent multicollinearity)
    model_df = pd.get_dummies(model_df, columns=categorical_features, drop_first=True)

    X = model_df.drop(columns=["is_default"])
    y = model_df["is_default"]

    # Train / Test split (80% train, 20% test) with target stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize numeric columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    print("\n[*] Training Logistic Regression model with class balance...")
    # class_weight='balanced' offsets the 80/20 non-default/default imbalance
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # -------------------------------------------------------------
    # Odds Ratios & Feature Importance (Question 4 & 5)
    # -------------------------------------------------------------
    odds_ratios = np.exp(model.coef_[0])
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": np.round(model.coef_[0], 4),
        "Odds_Ratio": np.round(odds_ratios, 4)
    }).sort_values(by="Odds_Ratio", ascending=False)

    print("\n" + "=" * 65)
    print("CREDIT RISK ODDS RATIOS (MULTIPLIER ON ODDS OF DEFAULT):")
    print("=" * 65)
    print(feature_importance_df.to_string(index=False))

    # -------------------------------------------------------------
    # Performance Evaluation Metrics
    # -------------------------------------------------------------
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("\n" + "=" * 65)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("=" * 65)
    print("\nClassification Report (Balanced Decision Boundary):")
    print(classification_report(y_test, y_pred))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    plt.title("Credit Risk ROC-AUC Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "logistic_roc_curve.png"), dpi=300)
    plt.close()
    print(f"\n[+] Saved ROC Curve figure to: {os.path.join(FIGURES_DIR, 'logistic_roc_curve.png')}")


if __name__ == "__main__":
    X_train_s, X_test_s, y_tr, y_te, features = prepare_modeling_data(PROCESSED_PATH)
    train_and_evaluate(X_train_s, X_test_s, y_tr, y_te, features)