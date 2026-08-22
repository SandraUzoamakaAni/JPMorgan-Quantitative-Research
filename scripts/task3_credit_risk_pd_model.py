"""
JP Morgan Forage — Quantitative Research Task 3
Credit Risk Analysis: Probability of Default (PD) & Expected Loss
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

CSV_PATH = "Task_3_and_4_Loan_Data.csv"
RECOVERY_RATE = 0.10
FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

df = pd.read_csv(CSV_PATH)
X = df[FEATURES]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
log_reg_auc = roc_auc_score(y_test, log_reg_probs)
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))

rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("=== Model comparison (held-out test set) ===")
print(f"{'Model':<20}{'AUC':>10}{'Accuracy':>12}")
print(f"{'Logistic Regression':<20}{log_reg_auc:>10.4f}{log_reg_acc:>12.4f}")
print(f"{'Random Forest':<20}{rf_auc:>10.4f}{rf_acc:>12.4f}")

print("\nLogistic regression coefficients (scaled features, higher |value| = stronger driver):")
for feat, coef in sorted(zip(FEATURES, log_reg.coef_[0]), key=lambda x: -abs(x[1])):
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {feat:<28} {coef:+.3f}  ({direction} default risk)")

print("\nRandom Forest feature importances:")
for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<28} {imp:.3f}")


def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    recovery_rate=RECOVERY_RATE,
):
    """Return estimated probability of default and expected loss."""
    row = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines_outstanding,
        "loan_amt_outstanding": loan_amt_outstanding,
        "total_debt_outstanding": total_debt_outstanding,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score,
    }])[FEATURES]

    row_scaled = scaler.transform(row)
    pd_estimate = float(log_reg.predict_proba(row_scaled)[0, 1])
    loss = pd_estimate * (1 - recovery_rate) * loan_amt_outstanding

    return {
        "probability_of_default": round(pd_estimate, 4),
        "expected_loss": round(loss, 2),
    }


if __name__ == "__main__":
    print("\n=== Example expected_loss() calls ===")
    low_risk = expected_loss(1, 3000, 2000, 80000, 8, 750)
    print("Low-risk borrower:", low_risk)
    high_risk = expected_loss(5, 9000, 15000, 25000, 1, 520)
    print("High-risk borrower:", high_risk)

    test_df = X_test.copy()
    test_df["actual_default"] = y_test.values
    test_df["pd_estimate"] = log_reg_probs
    test_df["expected_loss"] = test_df["pd_estimate"] * (1 - RECOVERY_RATE) * test_df["loan_amt_outstanding"]
    actual_loss = (test_df["actual_default"] * (1 - RECOVERY_RATE) * test_df["loan_amt_outstanding"]).sum()
    predicted_loss = test_df["expected_loss"].sum()
    print(f"\nPortfolio check (test set): predicted total loss = ${predicted_loss:,.0f} vs. actual total loss = ${actual_loss:,.0f}")
