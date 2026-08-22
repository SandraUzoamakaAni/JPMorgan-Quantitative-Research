# JPMorgan Chase Quantitative Research Job Simulation

Completed as part of the [JPMorgan Chase Quantitative Research Job Simulation](https://www.theforage.com/) on Forage.

## Project overview
Four tasks simulating work on a quantitative research desk:

1. **Price data analysis** — modeled natural gas price data (trend + seasonality via regression) and built a function to estimate the price for any date, interpolating within the historical range and extrapolating up to a year forward.
2. **Commodity storage contract pricing** — built a generalized pricing function for a gas storage contract, handling multiple injection/withdrawal dates, storage capacity limits, and injection/withdrawal fees.
3. **Credit risk analysis** — compared logistic regression and random forest models to estimate probability of default from loan/customer data, and converted PD into expected loss.
4. **FICO score bucketing** — used dynamic programming to optimally bucket FICO scores into ratings, maximizing a log-likelihood objective over default outcomes.

## Repo structure
- `scripts/` — Python solutions for each task
- `data/` — input datasets provided in the simulation
- `charts/` — visualization from Task 1 (price trend & extrapolation)
- `certificate/` — Forage certificate of completion

## Key results
- Task 1 price model: R² = 0.93 on historical data.
- Task 3: logistic regression chosen over random forest for interpretability (near-identical AUC), with clear directional coefficients for risk drivers.
- Task 4: 5-bucket FICO rating map, PD ranging from ~5% (best bucket) to ~66% (worst bucket).

## Tools
Python (pandas, numpy, scikit-learn, matplotlib)
