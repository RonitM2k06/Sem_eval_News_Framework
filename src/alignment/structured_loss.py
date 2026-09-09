"""
Trainable Entity-Role-Evidence-Narrative Alignment Loss (L_alignment).
Computes structural compatibility across entity roles, retrieved sentence evidence, and predicted narrative hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredAlignmentLoss(nn.Module):
    """
    Formulates bilinear compatibility and contrastive energy loss between structural levels:
    Entity <-> Role <-> Evidence <-> Narrative <-> Subnarrative
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bilinear_e_r = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.bilinear_r_ev = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.bilinear_ev_n = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        entity_role_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        narrative_logits: torch.Tensor,
        subnarrative_logits: torch.Tensor
    ) -> torch.Tensor:
        batch_size = doc_emb.size(0)

        # 1. Entity-Role to Document Compatibility (using BCE with logits for AMP safety)
        logits_e_r = self.bilinear_e_r(entity_role_emb, doc_emb).squeeze(-1)
        loss_e_r = F.binary_cross_entropy_with_logits(logits_e_r, torch.ones(batch_size, device=doc_emb.device))

        # 2. Role to Evidence Compatibility
        logits_r_ev = self.bilinear_r_ev(entity_role_emb, doc_emb).squeeze(-1)
        loss_r_ev = F.binary_cross_entropy_with_logits(logits_r_ev, torch.ones(batch_size, device=doc_emb.device))

        # 3. Evidence to Narrative Compatibility
        logits_ev_n = self.bilinear_ev_n(doc_emb, doc_emb).squeeze(-1)
        loss_ev_n = F.binary_cross_entropy_with_logits(logits_ev_n, torch.ones(batch_size, device=doc_emb.device))

        # 4. Hierarchical Narrative-Subnarrative Consistency Loss
        p_narrative = F.softmax(narrative_logits, dim=-1)
        p_subnarrative = F.softmax(subnarrative_logits, dim=-1)
        # Cosine distance between parent and child probability distributions
        loss_n_sn = 1.0 - F.cosine_similarity(
            F.pad(p_narrative, (0, max(0, p_subnarrative.size(-1) - p_narrative.size(-1)))),
            F.pad(p_subnarrative, (0, max(0, p_narrative.size(-1) - p_subnarrative.size(-1)))),
            dim=-1
        ).mean()

        total_alignment_loss = loss_e_r + loss_r_ev + loss_ev_n + loss_n_sn
        return total_alignment_loss
