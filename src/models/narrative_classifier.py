"""
Hierarchical Narrative Classifier (Subtask 2).
Predicts parent narrative and fine-grained subnarratives using hierarchical tree loss.
"""

from typing import Tuple
import torch
import torch.nn as nn


class HierarchicalNarrativeClassifier(nn.Module):
    """
    Two-level narrative & subnarrative classifier.
    """

    def __init__(self, hidden_dim: int, num_narratives: int, num_subnarratives: int, dropout: float = 0.1):
        super().__init__()
        self.num_narratives = num_narratives
        self.num_subnarratives = num_subnarratives

        self.parent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_narratives)
        )

        self.child_head = nn.Sequential(
            nn.Linear(hidden_dim + num_narratives, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_subnarratives)
        )

    def forward(self, feature_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        narrative_logits = self.parent_head(feature_repr)
        # Condition subnarrative prediction on parent narrative logits
        combined = torch.cat([feature_repr, narrative_logits], dim=-1)
        subnarrative_logits = self.child_head(combined)
        return narrative_logits, subnarrative_logits
