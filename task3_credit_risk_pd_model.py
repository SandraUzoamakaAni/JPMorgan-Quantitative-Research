"""
JP Morgan Forage — Quantitative Research Task 3
Credit Risk Analysis: Probability of Default (PD) & Expected Loss

Trains a model to estimate a borrower's probability of default from their
loan/customer characteristics, then wraps it in an expected_loss() function
that a risk analyst can call directly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

CSV_PATH = "Task_3_and_4_Loan_Data.csv"  # update path if needed
RECOVERY_RATE = 0.10  # given: if a borrower defaults, 10% of exposure is recovered

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
X = df[FEATURES]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 2. Model A — Logistic Regression (interpretable baseline)
#    Features are scaled first so coefficients are comparable to each other.
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
log_reg_auc = roc_auc_score(y_test, log_reg_probs)
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))

# ---------------------------------------------------------------
# 3. Model B — Random Forest (captures non-linear patterns)
# ---------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# ---------------------------------------------------------------
# 4. Comparative results
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 5. Final model choice: Logistic Regression
#    AUC is essentially tied with the Random Forest here, and logistic
#    regression is far easier to explain to risk/compliance stakeholders
#    ("each extra outstanding credit line raises PD by X") - so it's the
#    one wrapped for production use below.
# ---------------------------------------------------------------
def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    recovery_rate=RECOVERY_RATE,
):
    """
    Given a borrower's characteristics, return their estimated probability
    of default (PD) and the expected loss on their loan.

    Expected loss = PD x (1 - recovery_rate) x exposure_at_default
    where exposure_at_default is taken as the loan amount outstanding.
    """
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

    # A low-risk-looking borrower: high FICO, low debt, stable employment
    low_risk = expected_loss(
        credit_lines_outstanding=1,
        loan_amt_outstanding=3000,
        total_debt_outstanding=2000,
        income=80000,
        years_employed=8,
        fico_score=750,
    )
    print("Low-risk borrower:", low_risk)

    # A higher-risk-looking borrower: low FICO, high debt relative to income
    high_risk = expected_loss(
        credit_lines_outstanding=5,
        loan_amt_outstanding=9000,
        total_debt_outstanding=15000,
        income=25000,
        years_employed=1,
        fico_score=520,
    )
    print("High-risk borrower:", high_risk)

    # Sanity check: expected loss across the whole test set vs. actual losses
    test_df = X_test.copy()
    test_df["actual_default"] = y_test.values
    test_df["pd_estimate"] = log_reg_probs
    test_df["expected_loss"] = test_df["pd_estimate"] * (1 - RECOVERY_RATE) * test_df["loan_amt_outstanding"]
    actual_loss = (test_df["actual_default"] * (1 - RECOVERY_RATE) * test_df["loan_amt_outstanding"]).sum()
    predicted_loss = test_df["expected_loss"].sum()
    print(f"\nPortfolio check (test set): predicted total loss = ${predicted_loss:,.0f} "
          f"vs. actual total loss = ${actual_loss:,.0f}")
