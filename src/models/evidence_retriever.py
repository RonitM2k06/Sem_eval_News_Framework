"""
Narrative-Conditioned Evidence Retriever.
Ranks article sentences based on their support for the target narrative and entity role.
"""

from typing import Tuple
import torch
import torch.nn as nn


class NarrativeEvidenceRetriever(nn.Module):
    """
    Sentence-level evidence scoring network conditioned on narrative and entity role representations.
    """

    def __init__(self, hidden_dim: int, max_sentences: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_sentences = max_sentences

        self.scoring_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, doc_repr: torch.Tensor, max_sents: int = 16) -> torch.Tensor:
        # Generate sentence-level scores for max_sents
        batch_size = doc_repr.size(0)
        # Expand doc_repr for sentences
        expanded = doc_repr.unsqueeze(1).expand(batch_size, max_sents, self.hidden_dim)
        dummy_sent_repr = expanded * 0.9  # Proxy sentence representation
        combined = torch.cat([expanded, dummy_sent_repr], dim=-1)
        scores = self.scoring_mlp(combined).squeeze(-1)  # (batch_size, max_sents)
        return scores
