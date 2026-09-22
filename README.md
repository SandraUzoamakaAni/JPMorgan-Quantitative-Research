# JPMorgan Chase Quantitative Research Job Simulation

> A portfolio project based on the **JPMorgan Chase Quantitative Research Job Simulation** completed through Forage.

## About the project

This repository contains my solutions to four quantitative research tasks covering **commodity price modeling, derivatives/storage-contract pricing, credit risk, and FICO score segmentation**.

The project demonstrates how quantitative methods can be applied to practical financial problems and translated into reusable Python functions.

## Tasks

### 1. Natural Gas Price Estimation
- Analyzed historical natural gas price data.
- Modeled trend and seasonality using regression.
- Built a date-based price estimation function.
- Supported interpolation within the historical period and extrapolation for future dates.
- Produced a visual forecast.

### 2. Natural Gas Storage Contract Pricing
- Built a generalized storage-contract pricing function.
- Accounted for injection and withdrawal dates.
- Included storage capacity constraints and transaction fees.
- Calculated contract value from the underlying gas price curve.

### 3. Credit Risk & Probability of Default
- Prepared loan/customer data for modeling.
- Compared **Logistic Regression** with **Random Forest** classification.
- Evaluated model performance using AUC and accuracy.
- Estimated Probability of Default (PD).
- Calculated expected credit loss using exposure and recovery assumptions.

### 4. FICO Score Bucketing
- Implemented dynamic programming to find optimal FICO score buckets.
- Optimized the provided log-likelihood objective.
- Created a reusable FICO rating map.
- Summarized default rates and risk across the resulting buckets.

## Key results

| Area | Result / takeaway |
|---|---|
| Natural gas model | Historical fit achieved approximately **R² = 0.93**. |
| Credit risk | Compared Logistic Regression and Random Forest for PD estimation, with Logistic Regression providing an interpretable final model. |
| FICO segmentation | Created five optimized risk buckets with different observed default probabilities. |

## Repository structure

```text
JPMorgan-Quantitative-Research/
├── README.md
├── requirements.txt
├── data/
│   ├── Nat_Gas.csv
│   └── Task_3_and_4_Loan_Data.csv
├── scripts/
│   ├── task1_price_estimator.py
│   ├── task2_storage_contract_pricer.py
│   ├── task3_credit_risk_pd_model.py
│   └── task4_fico_bucketing_dp.py
├── nat_gas_price_forecast.png
└── JPMorgan_Certificate.png
```

The forecast image and certificate are kept at the repository root because GitHub's file workflow used for this cleanup does not support moving existing binary files programmatically. They remain fully accessible from the repository.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Regression modeling
- Classification
- Dynamic programming
- Financial risk modeling

## Running the project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the scripts from the repository root:

```bash
python scripts/task1_price_estimator.py
python scripts/task2_storage_contract_pricer.py
python scripts/task3_credit_risk_pd_model.py
python scripts/task4_fico_bucketing_dp.py
```

Task 1 uses `data/Nat_Gas.csv`. Tasks 3 and 4 use `data/Task_3_and_4_Loan_Data.csv`.

## Certificate

The repository includes the **JPMorgan Chase Quantitative Research Job Simulation certificate of completion** earned through Forage.

## Disclaimer

This is an educational portfolio project based on a job simulation. It is not affiliated with, endorsed by, or representative of JPMorgan Chase beyond the completion of the simulation itself.
