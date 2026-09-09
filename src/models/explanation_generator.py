"""
Explanation Generator (Subtask 3).
Generates concise, grounded textual explanations (<= 80 words) conditioned on predicted narrative and retrieved evidence.
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn


class RuleBasedGroundedGenerator(nn.Module):
    """
    Template- and evidence-conditioned explanation generator ensuring strict <= 80 words constraint.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, 128)

    def forward(self, doc_repr: torch.Tensor) -> torch.Tensor:
        return self.proj(doc_repr)

    def generate_explanation(
        self,
        article_id: str,
        narrative_name: str,
        subnarrative_name: str,
        entity_role: str,
        evidence_snippet: str
    ) -> str:
        """
        Generates evidence-grounded explanation within 80 words.
        """
        explanation = (
            f"The article frames the key entity as a {entity_role.lower()} within the overarching narrative of "
            f"'{narrative_name}', specifically advancing the subnarrative '{subnarrative_name}'. "
            f"This is supported by textual evidence stating: '{evidence_snippet[:150]}'."
        )
        words = explanation.split()
        if len(words) > 80:
            explanation = " ".join(words[:80])
        return explanation
