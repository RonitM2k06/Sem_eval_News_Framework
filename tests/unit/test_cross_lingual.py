"""
Unit tests for Cross-Lingual Structural Consistency.
"""

import pytest
from src.reasoning.engine import NarrativeReasoningEngine
from src.metrics.narrative_reasoning import NarrativeReasoningMetrics


def test_cross_lingual_consistency():
    engine = NarrativeReasoningEngine()
    text_a = "The United Nations Security Council met in New York."
    text_b = "Российская делегация провела переговоры в Женеве."
    res = engine.compare_cross_lingual_consistency(text_a, "en", text_b, "ru")

    assert "clsc_metrics" in res
    assert res["clsc_metrics"]["clsc_score"] > 0.0


def test_clsc_metric_sanity():
    g1 = {"nodes": [{"id": "1", "label": "UN"}, {"id": "2", "label": "Russia"}], "edges": [{"source": "1", "target": "2"}]}
    g2 = {"nodes": [{"id": "1", "label": "UN"}, {"id": "2", "label": "Russia"}], "edges": [{"source": "1", "target": "2"}]}
    res = NarrativeReasoningMetrics.compute_clsc(g1, g2, "en", "ru")
    assert res["clsc_score"] == 1.0
