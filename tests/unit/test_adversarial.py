"""
Unit tests for Adversarial Narrative Robustness.
"""

import pytest
from src.reasoning.engine import NarrativeReasoningEngine
from src.metrics.narrative_reasoning import NarrativeReasoningMetrics


def test_adversarial_robustness():
    engine = NarrativeReasoningEngine()
    text = "The United Nations Security Council met today."
    res = engine.run_adversarial_robustness_test(text, "en", "ukraine_russia", "irrelevant_insertion")

    assert "prs_metrics" in res
    assert res["prs_metrics"]["prs_score"] >= 0.0


def test_prs_metric_sanity():
    res = NarrativeReasoningMetrics.compute_prs(0.90, 0.88, "irrelevant_insertion", False)
    assert res["prs_score"] >= 0.90
    assert res["is_robust"] is True
