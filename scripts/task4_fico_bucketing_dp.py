"""
JP Morgan Forage — Quantitative Research Task 4
Bucket FICO Scores via Dynamic Programming (log-likelihood optimal)
"""

import pandas as pd
import numpy as np

CSV_PATH = "Task_3_and_4_Loan_Data.csv"


def fit_fico_buckets(fico_scores, defaults, num_buckets):
    df = pd.DataFrame({"fico": fico_scores, "default": defaults})
    grouped = df.groupby("fico")["default"].agg(n="count", k="sum").reset_index()
    grouped = grouped.sort_values("fico").reset_index(drop=True)
    scores = grouped["fico"].values
    n_arr = grouped["n"].values.astype(float)
    k_arr = grouped["k"].values.astype(float)
    U = len(scores)

    cum_n = np.concatenate([[0], np.cumsum(n_arr)])
    cum_k = np.concatenate([[0], np.cumsum(k_arr)])

    def bucket_ll(a, b):
        n = cum_n[b] - cum_n[a]
        k = cum_k[b] - cum_k[a]
        if n == 0:
            return 0.0
        p = k / n
        ll = 0.0
        if k > 0:
            ll += k * np.log(p)
        if (n - k) > 0:
            ll += (n - k) * np.log(1 - p)
        return ll

    NEG_INF = float("-inf")
    dp = [[NEG_INF] * (U + 1) for _ in range(num_buckets + 1)]
    back = [[None] * (U + 1) for _ in range(num_buckets + 1)]
    dp[0][0] = 0.0

    for j in range(1, num_buckets + 1):
        for i in range(j, U + 1):
            best_val, best_m = NEG_INF, None
            for m in range(j - 1, i):
                if dp[j - 1][m] == NEG_INF:
                    continue
                val = dp[j - 1][m] + bucket_ll(m, i)
                if val > best_val:
                    best_val, best_m = val, m
            dp[j][i] = best_val
            back[j][i] = best_m

    edges = [0] * (num_buckets + 1)
    edges[num_buckets] = U
    i, j = U, num_buckets
    while j > 0:
        m = back[j][i]
        edges[j - 1] = m
        i, j = m, j - 1

    boundaries = [int(scores[edges[b]]) for b in range(1, num_buckets)]
    bucket_summary = []
    for b in range(num_buckets):
        a, c = edges[b], edges[b + 1]
        n = cum_n[c] - cum_n[a]
        k = cum_k[c] - cum_k[a]
        bucket_summary.append({
            "fico_range": (int(scores[a]), int(scores[c - 1])),
            "count": int(n),
            "defaults": int(k),
            "probability_of_default": round(k / n, 4) if n else None,
        })

    return {
        "boundaries": boundaries,
        "log_likelihood": dp[num_buckets][U],
        "bucket_summary": bucket_summary,
    }


def make_rating_map(bucket_summary):
    n_buckets = len(bucket_summary)

    def rating_map(fico_score):
        for idx, bucket in enumerate(bucket_summary):
            low, high = bucket["fico_range"]
            if low <= fico_score <= high:
                return n_buckets - idx
        if fico_score < bucket_summary[0]["fico_range"][0]:
            return n_buckets
        return 1

    return rating_map


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    NUM_BUCKETS = 5
    result = fit_fico_buckets(df["fico_score"], df["default"], NUM_BUCKETS)

    print(f"Optimal boundaries ({NUM_BUCKETS} buckets): {result['boundaries']}")
    print(f"Total log-likelihood: {result['log_likelihood']:.2f}\n")
    print("Bucket summary (sorted low score -> high score):")
    print(f"{'FICO range':<15}{'Count':>8}{'Defaults':>10}{'PD':>10}{'Rating':>8}")
    rating_map = make_rating_map(result["bucket_summary"])
    n_buckets = len(result["bucket_summary"])
    for idx, b in enumerate(result["bucket_summary"]):
        rating = n_buckets - idx
        print(f"{str(b['fico_range']):<15}{b['count']:>8}{b['defaults']:>10}{b['probability_of_default']:>10}{rating:>8}")

    print("\n--- Example rating_map() lookups ---")
    for score in [410, 590, 640, 700, 780, 845]:
        print(f"  FICO {score} -> rating {rating_map(score)}")
