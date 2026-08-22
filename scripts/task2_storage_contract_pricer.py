"""
JP Morgan Forage — Quantitative Research Task 2
Commodity (Natural Gas) Storage Contract Pricing Model
"""

import pandas as pd


def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_prices,
    withdrawal_prices,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_month,
    injection_withdrawal_cost_rate=0.0,
):
    """Value a natural gas storage contract with potentially multiple dates."""
    events = (
        [(pd.to_datetime(d), "inject", p) for d, p in zip(injection_dates, injection_prices)]
        + [(pd.to_datetime(d), "withdraw", p) for d, p in zip(withdrawal_dates, withdrawal_prices)]
    )
    events.sort(key=lambda e: e[0])

    volume_in_storage = 0.0
    purchase_cost = 0.0
    sale_revenue = 0.0
    injection_withdrawal_fees = 0.0
    skipped_events = []

    for date, action, price in events:
        if action == "inject":
            if volume_in_storage + injection_rate > max_volume:
                skipped_events.append((date, action, "would exceed max storage capacity"))
                continue
            volume_in_storage += injection_rate
            purchase_cost += injection_rate * price
            injection_withdrawal_fees += injection_rate * injection_withdrawal_cost_rate
        elif action == "withdraw":
            if volume_in_storage - withdrawal_rate < 0:
                skipped_events.append((date, action, "not enough gas in storage"))
                continue
            volume_in_storage -= withdrawal_rate
            sale_revenue += withdrawal_rate * price
            injection_withdrawal_fees += withdrawal_rate * injection_withdrawal_cost_rate

    if events:
        first_date = min(e[0] for e in events)
        last_date = max(e[0] for e in events)
        months_stored = max(1, round((last_date - first_date).days / 30.44))
    else:
        months_stored = 0
    storage_cost = months_stored * storage_cost_per_month
    contract_value = sale_revenue - purchase_cost - storage_cost - injection_withdrawal_fees

    return {
        "contract_value": round(contract_value, 2),
        "sale_revenue": round(sale_revenue, 2),
        "purchase_cost": round(purchase_cost, 2),
        "storage_cost": round(storage_cost, 2),
        "injection_withdrawal_fees": round(injection_withdrawal_fees, 2),
        "months_stored": months_stored,
        "final_volume_in_storage": volume_in_storage,
        "skipped_events": skipped_events,
    }


if __name__ == "__main__":
    result = price_storage_contract(
        injection_dates=["2024-06-01"],
        withdrawal_dates=["2024-10-01"],
        injection_prices=[2.0],
        withdrawal_prices=[3.0],
        injection_rate=1_000_000,
        withdrawal_rate=1_000_000,
        max_volume=1_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_rate=0.005,
    )
    print("Test 1 (matches background example):")
    for k, v in result.items():
        print(f"  {k}: {v}")

    result2 = price_storage_contract(
        injection_dates=["2024-06-01", "2024-07-01", "2024-08-01"],
        withdrawal_dates=["2024-12-01", "2025-01-01", "2025-02-01"],
        injection_prices=[10.7, 10.9, 10.8],
        withdrawal_prices=[11.8, 12.1, 12.4],
        injection_rate=100_000,
        withdrawal_rate=100_000,
        max_volume=300_000,
        storage_cost_per_month=50_000,
        injection_withdrawal_cost_rate=0.01,
    )
    print("\nTest 2 (multiple dates, staggered injection/withdrawal):")
    for k, v in result2.items():
        print(f"  {k}: {v}")
