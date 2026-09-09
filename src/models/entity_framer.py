"""
Entity Framing Module (Subtask 1).
Predicts main role (Protagonist, Antagonist, Innocent) and fine-grained roles from entity spans and document context.
"""

from typing import Tuple
import torch
import torch.nn as nn


class EntityFramingClassifier(nn.Module):
    """
    Multi-label entity role classifier producing role embeddings and classification logits.
    """

    def __init__(self, hidden_dim: int, num_roles: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_roles = num_roles

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_roles)
        )
        self.role_embedding_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, doc_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Logits for all roles
        logits = self.mlp(doc_representation)
        # Entity-role embedding vector
        entity_role_emb = self.role_embedding_projection(doc_representation)
        return logits, entity_role_emb
