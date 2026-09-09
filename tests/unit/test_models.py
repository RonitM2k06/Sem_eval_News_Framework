"""
Unit tests for NarrativeGraph Master Model components.
"""

import pytest
import torch
from src.taxonomy.parser import TaxonomyParser
from src.models.narrative_graph_model import NarrativeGraphModel


def test_narrative_graph_model_forward():
    taxonomy = TaxonomyParser()
    model = NarrativeGraphModel(
        taxonomy=taxonomy,
        model_name="xlm-roberta-base",
        hidden_dim=768,
        use_graph=True,
        use_alignment=True
    )

    batch_size = 2
    max_seq_len = 32
    max_sentences = 8

    dummy_batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, max_seq_len)),
        "attention_mask": torch.ones((batch_size, max_seq_len), dtype=torch.long),
        "sent_mask": torch.ones((batch_size, max_sentences), dtype=torch.float)
    }

    outputs = model(dummy_batch)

    assert "role_logits" in outputs
    assert "narrative_logits" in outputs
    assert "subnarrative_logits" in outputs
    assert "alignment_loss" in outputs
    assert outputs["role_logits"].shape == (batch_size, taxonomy.num_roles)
    assert outputs["narrative_logits"].shape == (batch_size, taxonomy.num_narratives)
    assert outputs["subnarrative_logits"].shape == (batch_size, taxonomy.num_subnarratives)
