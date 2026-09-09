"""
Unit tests for Research Provenance & Validator script.
"""

import pytest
from scripts.validate_research_results import validate_provenance


def test_provenance_validation():
    success = validate_provenance()
    assert success is True
