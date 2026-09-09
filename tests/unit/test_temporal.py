"""
Unit tests for Temporal Narrative Evolution Engine.
"""

import pytest
from src.reasoning.engine import NarrativeReasoningEngine
from src.metrics.narrative_reasoning import NarrativeReasoningMetrics


def test_temporal_narrative_evolution():
    engine = NarrativeReasoningEngine()
    res = engine.get_temporal_evolution_trajectory("United Nations Security Council", "ukraine_russia")

    assert "temporal_metrics" in res
    assert "nps_score" in res["temporal_metrics"]
    assert "nvs_score" in res["temporal_metrics"]
    assert len(res["timesteps"]) == 4


def test_temporal_metrics_sanity():
    steps = [
        {"week": "W1", "role": "Protagonist", "narrative": "N1"},
        {"week": "W2", "role": "Protagonist", "narrative": "N1"},
        {"week": "W3", "role": "Victim", "narrative": "N2"}
    ]
    res = NarrativeReasoningMetrics.compute_temporal_evolution(steps)
    assert res["nps_score"] < 1.0
    assert res["role_transitions"] == 1
