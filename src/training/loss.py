"""
Multi-Task Loss Computation for NarrativeGraph.
Combines BCEWithLogits (Entity Roles & Evidence), CrossEntropy (Narrative & Subnarrative), and Alignment Loss.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NarrativeGraphMultiTaskLoss(nn.Module):
    """
    Weighted Multi-Task Loss:
    L_total = lambda_entity * L_entity + lambda_narrative * (L_narrative + L_subnarrative) + lambda_evidence * L_evidence + lambda_alignment * L_alignment
    """

    def __init__(
        self,
        lambda_entity: float = 1.0,
        lambda_narrative: float = 1.0,
        lambda_evidence: float = 0.5,
        lambda_alignment: float = 0.5
    ):
        super().__init__()
        self.lambda_entity = lambda_entity
        self.lambda_narrative = lambda_narrative
        self.lambda_evidence = lambda_evidence
        self.lambda_alignment = lambda_alignment

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Entity Role Loss (BCE)
        loss_entity = self.bce_loss(outputs["role_logits"], batch["role_target"])

        # Clamp targets safely within logits class count boundaries
        num_parent_classes = outputs["narrative_logits"].shape[-1]
        num_child_classes = outputs["subnarrative_logits"].shape[-1]

        narr_target = batch["narrative_target"].clamp(-1, num_parent_classes - 1)
        subnarr_target = batch["subnarrative_target"].clamp(-1, num_child_classes - 1)

        # Parent & Subnarrative Loss (CE)
        loss_parent = self.ce_loss(outputs["narrative_logits"], narr_target)
        loss_child = self.ce_loss(outputs["subnarrative_logits"], subnarr_target)
        loss_narrative = loss_parent + loss_child

        # Evidence Sentence Loss (BCE)
        loss_evidence = self.bce_loss(outputs["evidence_scores"], batch["evidence_target"])

        # Alignment Loss
        loss_alignment = outputs.get("alignment_loss", torch.tensor(0.0, device=loss_entity.device))

        # Total Loss
        total_loss = (
            self.lambda_entity * loss_entity +
            self.lambda_narrative * loss_narrative +
            self.lambda_evidence * loss_evidence +
            self.lambda_alignment * loss_alignment
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_entity": loss_entity.item(),
            "loss_narrative": loss_narrative.item(),
            "loss_evidence": loss_evidence.item(),
            "loss_alignment": loss_alignment.item()
        }

        return total_loss, loss_dict
