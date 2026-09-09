"""
Unit tests for Narrative Conflict Detection Engine.
"""

import pytest
from src.reasoning.engine import NarrativeReasoningEngine
from src.metrics.narrative_reasoning import NarrativeReasoningMetrics


def test_narrative_conflict_detection():
    engine = NarrativeReasoningEngine()
    text = "The United Nations Security Council met today. Western envoys condemned actions while opposing delegates defended them."
    res = engine.detect_narrative_conflict(text, "en", "ukraine_russia")

    assert "dominant_narrative" in res
    assert "competing_narrative" in res
    assert "ncs_score" in res["conflict_metrics"]


def test_ncs_metric_sanity():
    res = NarrativeReasoningMetrics.compute_ncs(0.85, 0.45, 0.40, 0.60)
    assert 0.0 <= res["ncs_score"] <= 1.0
    assert "has_conflict" in res
