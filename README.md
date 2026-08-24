# JPMorgan Chase Quantitative Research Job Simulation

> A portfolio project based on the **JPMorgan Chase Quantitative Research Job Simulation** completed through Forage.

## About the project

This repository contains my solutions to four quantitative research tasks covering **commodity price modeling, derivatives/storage-contract pricing, credit risk, and FICO score segmentation**.

The project focuses on applying quantitative methods to practical financial problems and turning those methods into reusable Python functions.

## Tasks

### 1. Natural Gas Price Estimation
- Analyzed historical natural gas price data.
- Modeled trend and seasonality using regression.
- Built a date-based price estimation function.
- Supported interpolation within the historical period and extrapolation for future dates.
- Produced a visual forecast of the modeled prices.

### 2. Natural Gas Storage Contract Pricing
- Built a generalized storage-contract pricing function.
- Accounted for injection and withdrawal dates.
- Included storage capacity constraints and transaction fees.
- Calculated the contract value from the underlying gas price curve.

### 3. Credit Risk & Probability of Default
- Prepared loan/customer data for modeling.
- Compared **Logistic Regression** with **Random Forest** classification.
- Evaluated the models using AUC and accuracy.
- Estimated Probability of Default (PD).
- Converted PD into expected credit loss using exposure and recovery assumptions.

### 4. FICO Score Bucketing
- Implemented dynamic programming to find optimal FICO score buckets.
- Optimized the provided log-likelihood objective.
- Created a reusable rating map where lower ratings represent stronger credit quality.
- Summarized default rates and risk across the resulting buckets.

## Key results

| Area | Result / takeaway |
|---|---|
| Natural gas model | Historical fit achieved approximately **R² = 0.93**. |
| Credit risk | Logistic Regression was selected as the final model because its performance was close to Random Forest while remaining easier to interpret. |
| FICO segmentation | Five risk buckets were created, with substantially different observed default probabilities across buckets. |

## Repository structure

```text
JPMorgan-Quantitative-Research/
├── README.md
├── Nat_Gas.csv
├── Task_3_and_4_Loan_Data.csv
├── JPMorgan_Certificate.png
├── nat_gas_price_forecast.png
├── scripts/
│   ├── task1_price_estimator.py
│   ├── task2_storage_contract_pricer.py
│   ├── task3_credit_risk_pd_model.py
│   └── task4_fico_bucketing_dp.py
└── data/
```

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

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Then run the task scripts from the repository root. The Task 3 and Task 4 scripts use `Task_3_and_4_Loan_Data.csv`, while Task 1 uses `Nat_Gas.csv`.

## Certificate

The repository includes the **JPMorgan Chase Quantitative Research Job Simulation certificate of completion** earned through Forage.

## Disclaimer

This is an educational portfolio project based on a job simulation. It is not affiliated with, endorsed by, or representative of JPMorgan Chase beyond the completion of the simulation itself.
