"""
Integration test for end-to-end training and evaluation pipeline on synthetic dataset.
"""

import os
import pytest
from scripts.prepare_data import prepare_all_data
from scripts.train import run_training
from scripts.validate_submission import validate_submission_file


def test_full_pipeline_synthetic():
    # 1. Generate data
    prepare_all_data()
    assert os.path.exists("data/synthetic/synthetic_train.json")

    # 2. Run single epoch training test
    best_metrics = run_training("configs/narrativegraph.yaml", seed=42)
    assert "macro_f1" in best_metrics
    assert best_metrics["macro_f1"] >= 0.0
