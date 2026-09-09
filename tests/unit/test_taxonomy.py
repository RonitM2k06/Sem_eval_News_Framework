"""
Unit tests for Taxonomy Parser.
"""

import pytest
from src.taxonomy.parser import TaxonomyParser


def test_taxonomy_roles():
    parser = TaxonomyParser()
    assert parser.num_roles > 0
    assert "Protagonist" in parser.main_roles
    assert "Antagonist" in parser.main_roles

    encoded = parser.encode_roles(["Protagonist", "Hero"])
    decoded = parser.decode_roles(encoded)
    assert "Protagonist" in decoded
    assert "Hero" in decoded


def test_taxonomy_narratives():
    parser = TaxonomyParser()
    assert parser.num_narratives > 0
    assert parser.num_subnarratives > 0

    nid = parser.encode_narrative(parser.narratives[0])
    decoded_n = parser.decode_narrative(nid)
    assert decoded_n == parser.narratives[0]
