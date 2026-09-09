"""
Unit tests for Evidence Necessity, Sufficiency, and Minimality.
"""

import pytest
from src.reasoning.engine import NarrativeReasoningEngine
from src.metrics.narrative_reasoning import NarrativeReasoningMetrics


def test_evidence_grounding_pipeline():
    engine = NarrativeReasoningEngine()
    text = "The United Nations Security Council convened an emergency meeting. Zelensky requested urgent military assistance."
    res = engine.evaluate_evidence_necessity_sufficiency(text, "en", "ukraine_russia")

    assert "ens_score" in res["evidence_metrics"]
    assert "ess_score" in res["evidence_metrics"]
    assert "ems_score" in res["evidence_metrics"]
    assert len(res["preservation_curve"]) >= 1


def test_ens_ess_metric_sanity():
    res = NarrativeReasoningMetrics.compute_ens_ess_ems(0.90, 0.40, 0.85, [0.75, 0.88, 0.90])
    assert res["ens_score"] > 0.0
    assert res["ess_score"] > 0.0
    assert res["ems_score"] > 0.0
