"""
Unit tests for Counterfactual Narrative Reasoning Engine.
"""

import pytest
from src.reasoning.engine import NarrativeReasoningEngine
from src.metrics.narrative_reasoning import NarrativeReasoningMetrics


def test_counterfactual_entity_removal():
    engine = NarrativeReasoningEngine()
    text = "The United Nations Security Council convened an emergency meeting."
    res = engine.run_counterfactual_intervention(text, "en", "ukraine_russia", "remove_entity", "United Nations Security Council")
    
    assert "delta_confidence" in res
    assert "cns_metrics" in res
    assert res["cns_metrics"]["metric_name"] == "CNS"
    assert res["intervened_target"] == "United Nations Security Council"


def test_cns_metric_sanity():
    res = NarrativeReasoningMetrics.compute_cns(0.88, 0.42, "remove_entity", "decrease")
    assert res["cns_score"] > 0.0
    assert res["directional_correctness"] is True
