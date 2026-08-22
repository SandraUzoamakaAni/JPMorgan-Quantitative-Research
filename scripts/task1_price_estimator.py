"""
JP Morgan Forage — Quantitative Research Task 1
Natural Gas Price Estimator

Given monthly natural gas price snapshots (Oct 2020 - Sep 2024), this script:
  1. Fits a model capturing (a) the long-term price trend and (b) the
     yearly seasonal pattern (gas is more expensive in winter, cheaper in summer).
  2. Exposes `estimate_price(date)` which returns a price estimate for
     ANY date - whether it falls inside the historical range (interpolation)
     or up to a year beyond the last data point (extrapolation).
  3. Plots the actual data against the fitted/extrapolated curve.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

CSV_PATH = "Nat_Gas.csv"  # update path if needed

df = pd.read_csv(CSV_PATH)
df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
df = df.sort_values("Dates").reset_index(drop=True)

START_DATE = df["Dates"].min()
df["t"] = (df["Dates"] - START_DATE).dt.days / 30.44

def _build_features(t):
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        t,
        np.sin(2 * np.pi * t / 12),
        np.cos(2 * np.pi * t / 12),
    ])

_X = _build_features(df["t"])
_y = df["Prices"].values

_model = LinearRegression()
_model.fit(_X, _y)

def estimate_price(date_input):
    """Return an estimated natural gas purchase price for any given date."""
    date = pd.to_datetime(date_input)
    t = (date - START_DATE).days / 30.44
    X_new = _build_features([t])
    return float(_model.predict(X_new)[0])

if __name__ == "__main__":
    for d in ["2021-03-15", "2023-11-01", "2024-09-30", "2025-04-30", "2025-09-30"]:
        print(f"{d}: ${estimate_price(d):.2f}")

    future_dates = pd.date_range(df["Dates"].min(), df["Dates"].max() + pd.DateOffset(years=1), freq="D")
    future_t = (future_dates - START_DATE).days / 30.44
    future_prices = _model.predict(_build_features(future_t))

    plt.figure(figsize=(11, 5))
    plt.plot(df["Dates"], df["Prices"], "o", label="Actual monthly snapshots", color="#2563eb")
    plt.plot(future_dates, future_prices, "-", label="Model fit + 1yr extrapolation", color="#f97316", linewidth=2)
    plt.axvline(df["Dates"].max(), color="gray", linestyle="--", linewidth=1, label="Last known data point")
    plt.title("Natural Gas Price: Historical Data & 1-Year Extrapolation")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("nat_gas_price_forecast.png", dpi=150)
    plt.show()
