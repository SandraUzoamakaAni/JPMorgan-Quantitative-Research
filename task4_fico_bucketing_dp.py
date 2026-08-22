"""
JP Morgan Forage — Quantitative Research Task 4
Bucket FICO Scores via Dynamic Programming (log-likelihood optimal)

Given a target number of buckets, finds the FICO score cutoffs that
maximize the log-likelihood function provided in the task instructions,
then wraps the result in a rating_map(fico_score) function.
Lower rating = better credit score (fewer expected defaults).
"""

import pandas as pd
import numpy as np

CSV_PATH = "Task_3_and_4_Loan_Data.csv"  # update path if needed


def fit_fico_buckets(fico_scores, defaults, num_buckets):
    """
    Find the FICO score boundaries that maximize the log-likelihood of
    the resulting buckets, via dynamic programming.

    Parameters
    ----------
    fico_scores : array-like of int — each borrower's FICO score
    defaults : array-like of 0/1 — whether that borrower defaulted
    num_buckets : int — how many buckets to split scores into

    Returns
    -------
    dict with:
      'boundaries' : the FICO score cutoffs between buckets
      'log_likelihood' : the best total log-likelihood achieved
      'bucket_summary' : per-bucket score range, count, defaults, PD
    """
    df = pd.DataFrame({"fico": fico_scores, "default": defaults})

    # 1. Aggregate to unique score values (this is where every candidate
    #    boundary can occur), sorted ascending.
    grouped = df.groupby("fico")["default"].agg(n="count", k="sum").reset_index()
    grouped = grouped.sort_values("fico").reset_index(drop=True)
    scores = grouped["fico"].values
    n_arr = grouped["n"].values.astype(float)
    k_arr = grouped["k"].values.astype(float)
    U = len(scores)  # number of unique score groups

    # 2. Prefix sums so we can get n/k for ANY range [a, b) in O(1)
    cum_n = np.concatenate([[0], np.cumsum(n_arr)])
    cum_k = np.concatenate([[0], np.cumsum(k_arr)])

    def bucket_ll(a, b):
        """Log-likelihood contribution of grouping records [a, b) into one bucket."""
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

    # 3. Dynamic programming table
    #    dp[j][i] = best total log-likelihood using j buckets over the
    #    first i score-groups. back[j][i] = the split point that achieved it.
    NEG_INF = float("-inf")
    dp = [[NEG_INF] * (U + 1) for _ in range(num_buckets + 1)]
    back = [[None] * (U + 1) for _ in range(num_buckets + 1)]
    dp[0][0] = 0.0

    for j in range(1, num_buckets + 1):
        for i in range(j, U + 1):          # need at least 1 group per bucket
            best_val, best_m = NEG_INF, None
            for m in range(j - 1, i):       # m = end of the previous j-1 buckets
                if dp[j - 1][m] == NEG_INF:
                    continue
                val = dp[j - 1][m] + bucket_ll(m, i)
                if val > best_val:
                    best_val, best_m = val, m
            dp[j][i] = best_val
            back[j][i] = best_m

    # 4. Backtrack to recover the actual split points.
    #    edges[b] = index of the first group in bucket b (edges[0]=0,
    #    edges[num_buckets]=U); there are num_buckets+1 edges bounding
    #    num_buckets buckets.
    edges = [0] * (num_buckets + 1)
    edges[num_buckets] = U
    i, j = U, num_buckets
    while j > 0:
        m = back[j][i]
        edges[j - 1] = m
        i, j = m, j - 1

    # 5. Translate the INTERNAL group-index boundaries into FICO score
    #    cutoffs (the boundaries *between* buckets), and summarize each bucket
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
    """
    Turn the fitted buckets into a rating_map(fico_score) -> rating function.
    Rating 1 = best (highest FICO / lowest PD) bucket, ascending from there
    — matching the task's requirement that "a lower rating signifies a
    better credit score."
    """
    # bucket_summary is already sorted low-score -> high-score;
    # the LAST bucket (highest scores) should get rating 1.
    n_buckets = len(bucket_summary)

    def rating_map(fico_score):
        for idx, bucket in enumerate(bucket_summary):
            low, high = bucket["fico_range"]
            if low <= fico_score <= high:
                # idx=0 is the lowest-score bucket -> worst rating (n_buckets)
                # idx=n_buckets-1 is the highest-score bucket -> best rating (1)
                return n_buckets - idx
        # scores outside the observed range: clip to nearest bucket
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
        print(f"{str(b['fico_range']):<15}{b['count']:>8}{b['defaults']:>10}"
              f"{b['probability_of_default']:>10}{rating:>8}")

    print("\n--- Example rating_map() lookups ---")
    for score in [410, 590, 640, 700, 780, 845]:
        print(f"  FICO {score} -> rating {rating_map(score)}")
