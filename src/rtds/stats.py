"""Statistics over datasets: per-dataset averages, bootstrap intervals and paired tests.

Patients within a dataset share one forest, so they are not independent. Every
interval and test here therefore treats the dataset as the unit, never the patient.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def per_dataset(table1):
    """Average over explained patients within each dataset: one row per (rho, dataset)."""
    t = table1.copy()
    total = t[["credit_visit_1", "credit_visit_2", "credit_visit_3"]].sum(axis=1)
    for k in (1, 2, 3):
        # Each visit's share of the patient's total credit (NaN if all credits are 0).
        t[f"share_visit_{k}"] = t[f"credit_visit_{k}"] / total.where(total > 0)
    cols = ["hit", "credit_share", "share_visit_1", "share_visit_2", "share_visit_3"]
    # mean() skips NaN shares; the NaN count is reported separately from the log.
    return t.groupby(["rho", "dataset_seed"])[cols].mean().reset_index()


def bootstrap_ci(values, gen, n_boot, level):
    """Mean and percentile interval from resampling the datasets with replacement."""
    values = np.asarray(values, dtype=float)
    idx = gen.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return values.mean(), np.quantile(means, alpha), np.quantile(means, 1 - alpha)


def paired_tests(per_ds, metrics, rhos):
    """Wilcoxon signed-rank between every pair of rho levels, pairing on dataset_seed."""
    rows = []
    for metric in metrics:
        wide = per_ds.pivot(index="dataset_seed", columns="rho", values=metric)
        for a, b in itertools.combinations(rhos, 2):
            diff = (wide[b] - wide[a]).dropna()
            if np.all(diff == 0):
                stat, p = np.nan, 1.0  # identical in every dataset: nothing to test
            else:
                stat, p = wilcoxon(wide.loc[diff.index, b], wide.loc[diff.index, a])
            rows.append({"metric": metric, "rho_a": a, "rho_b": b, "n_datasets": len(diff),
                         "mean_a": wide[a].mean(), "mean_b": wide[b].mean(),
                         "median_diff_b_minus_a": diff.median(), "wilcoxon_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank on two per-dataset series indexed by dataset_seed (pairs = same dataset).

    Returns (n_datasets, mean_a, mean_b, median of b - a, statistic, p-value)."""
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = both["b"] - both["a"]
    if np.all(diff == 0):
        stat, p = np.nan, 1.0  # identical in every dataset: nothing to test
    else:
        stat, p = wilcoxon(both["b"], both["a"])
    return len(both), both["a"].mean(), both["b"].mean(), diff.median(), stat, p


def holm(p_values):
    """Holm's step-down adjustment for many comparisons (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
