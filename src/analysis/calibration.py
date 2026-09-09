"""
Expected Calibration Error (ECE) and Reliability Diagram calculator.
"""

from typing import List, Tuple
import numpy as np


def compute_expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidences.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    accuracies = (predictions == targets).astype(float)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)
