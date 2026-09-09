"""
API Endpoint Test Suite for NarrativeGraph.
Tests all 8 major endpoints with valid requests, empty inputs, malformed bodies, and verifies response schemas.
"""

import pytest
import urllib.request
import urllib.error
import json

BASE_URL = "http://127.0.0.1:8000"


def make_request(path: str, method: str = "GET", data: dict = None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            res_data = json.loads(response.read().decode("utf-8"))
            return status, res_data
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            parsed = json.loads(error_body)
        except Exception:
            parsed = {"raw": error_body}
        return e.code, parsed


class TestApiEndpoints:

    def test_project_summary(self):
        status, data = make_request("/api/project")
        assert status == 200
        assert data.get("title") == "NarrativeGraph"
        assert data.get("overall_macro_f1") == 0.7680

    def test_paper_source(self):
        status, data = make_request("/api/paper-source")
        assert status == 200
        content = data.get("content") or data.get("source")
        assert content is not None
        assert "NarrativeGraph" in content

    def test_analyze_valid(self):
        payload = {
            "text": "The United Nations Security Council met in New York to address rising tensions.",
            "language": "en",
            "domain": "ukraine_russia"
        }
        status, data = make_request("/api/analyze", method="POST", data=payload)
        assert status == 200
        assert "subtask1_entity_framing" in data or "entities" in data
        assert "subtask2_narrative" in data or "narrative" in data

    def test_analyze_empty_text(self):
        payload = {"text": "", "language": "en", "domain": "ukraine_russia"}
        status, data = make_request("/api/analyze", method="POST", data=payload)
        # Should return 200 with fallback or 422
        assert status in [200, 422]

    def test_counterfactual_valid(self):
        payload = {
            "text": "The United Nations Security Council convened an emergency session in New York.",
            "language": "en",
            "domain": "ukraine_russia",
            "intervention_type": "remove_entity",
            "target_name": "United Nations Security Council"
        }
        status, data = make_request("/api/counterfactual", method="POST", data=payload)
        assert status == 200
        assert "cns_metrics" in data or "delta" in data

    def test_evidence_analyze_valid(self):
        payload = {
            "text": "The United Nations Security Council convened an emergency session in New York.",
            "language": "en",
            "domain": "ukraine_russia"
        }
        status, data = make_request("/api/evidence/analyze", method="POST", data=payload)
        assert status == 200
        assert "evidence_metrics" in data or "ens_score" in str(data)

    def test_conflict_analyze_valid(self):
        payload = {
            "text": "The United Nations Security Council convened an emergency session in New York.",
            "language": "en",
            "domain": "ukraine_russia"
        }
        status, data = make_request("/api/conflict/analyze", method="POST", data=payload)
        assert status == 200
        assert "conflict_metrics" in data or "ncs_score" in str(data)

    def test_cross_lingual_compare_valid(self):
        payload = {
            "text_a": "The United Nations Security Council convened an emergency session in New York.",
            "lang_a": "en",
            "text_b": "Российская делегация на переговорах в Женеве сделала заявление.",
            "lang_b": "ru",
            "domain": "ukraine_russia"
        }
        status, data = make_request("/api/cross-lingual/compare", method="POST", data=payload)
        assert status == 200
        assert "clsc_metrics" in data or "is_structurally_consistent" in data

    def test_adversarial_run_valid(self):
        payload = {
            "text": "The United Nations Security Council convened an emergency session in New York.",
            "language": "en",
            "domain": "ukraine_russia",
            "perturbation_type": "irrelevant_insertion"
        }
        status, data = make_request("/api/adversarial/run", method="POST", data=payload)
        assert status == 200
        assert "prs_metrics" in data or "delta" in data

    def test_temporal_evolution_valid(self):
        status, data = make_request("/api/temporal/evolution?entity=United+Nations&domain=ukraine_russia")
        assert status == 200
        assert "timesteps" in data or "temporal_metrics" in data

    def test_malformed_input(self):
        # Missing required field 'text'
        payload = {"invalid_field": 123}
        status, data = make_request("/api/analyze", method="POST", data=payload)
        assert status == 422  # Unprocessable Entity
        assert "detail" in data
