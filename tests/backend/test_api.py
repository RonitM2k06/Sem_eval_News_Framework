"""
Unit tests for FastAPI Backend API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_project_summary_endpoint():
    response = client.get("/api/project")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "NarrativeGraph"


def test_results_endpoints():
    res_main = client.get("/api/results/main")
    assert res_main.status_code == 200
    assert isinstance(res_main.json(), list)

    res_abl = client.get("/api/results/ablations")
    assert res_abl.status_code == 200

    res_xl = client.get("/api/results/cross-lingual")
    assert res_xl.status_code == 200


def test_analyze_endpoint():
    req_body = {
        "text": "NATO members discussed eastern security expansion while Russian diplomats urged peace negotiations in Geneva.",
        "language": "en",
        "domain": "ukraine_russia"
    }
    response = client.post("/api/analyze", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert "subtask1_entity_framing" in data
    assert "subtask2_narrative" in data
    assert "subtask3_explanation" in data
    assert "narrative_graph" in data
