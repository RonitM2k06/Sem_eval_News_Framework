"""
Statistical Significance Tester using Paired Bootstrap Resampling.
Calculates p-values and 95% Confidence Intervals comparing NarrativeGraph against baselines.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def paired_bootstrap_test(
    scores_model_a: List[float],
    scores_model_b: List[float],
    num_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Performs paired bootstrap resampling to compute p-value and mean delta.
    Model A: Proposed Model
    Model B: Baseline
    """
    np.random.seed(seed)
    n = len(scores_model_a)
    if n == 0 or len(scores_model_b) != n:
        return {"p_value": 1.0, "mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    arr_a = np.array(scores_model_a)
    arr_b = np.array(scores_model_b)
    obs_diff = np.mean(arr_a) - np.mean(arr_b)

    boot_diffs = []
    count_smaller = 0

    for _ in range(num_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        sample_a = arr_a[idx]
        sample_b = arr_b[idx]
        diff = np.mean(sample_a) - np.mean(sample_b)
        boot_diffs.append(diff)
        if diff <= 0:
            count_smaller += 1

    p_value = count_smaller / num_bootstraps
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "p_value": float(p_value),
        "mean_diff": float(obs_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }
