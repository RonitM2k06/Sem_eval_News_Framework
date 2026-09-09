"""
Master NarrativeGraph Model.
Unifies Multilingual Encoder, Entity Framing Head, Hierarchical Narrative Classifier, Narrative Evidence Retriever,
Heterogeneous GNN Message Passing, Structured Alignment Loss, and Grounded Explanation Generator.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from src.models.encoders import MultilingualTextEncoder
from src.models.entity_framer import EntityFramingClassifier
from src.models.narrative_classifier import HierarchicalNarrativeClassifier
from src.models.evidence_retriever import NarrativeEvidenceRetriever
from src.models.explanation_generator import RuleBasedGroundedGenerator
from src.alignment.structured_loss import StructuredAlignmentLoss
from src.graph.builder import NarrativeGraphBuilder
from src.graph.gnn_layer import HeteroGraphAttentionLayer
from src.taxonomy.parser import TaxonomyParser


class NarrativeGraphModel(nn.Module):
    """
    Unified end-to-end framework for SemEval 2025 Task 10.
    """

    def __init__(
        self,
        taxonomy: TaxonomyParser,
        model_name: str = "xlm-roberta-base",
        hidden_dim: int = 768,
        gnn_dim: int = 256,
        gnn_heads: int = 4,
        max_sentences: int = 16,
        max_entities: int = 8,
        use_graph: bool = True,
        use_alignment: bool = True,
        use_evidence_conditioning: bool = True
    ):
        super().__init__()
        self.taxonomy = taxonomy
        self.hidden_dim = hidden_dim
        self.use_graph = use_graph
        self.use_alignment = use_alignment
        self.use_evidence_conditioning = use_evidence_conditioning

        # Backbone encoder
        self.encoder = MultilingualTextEncoder(model_name=model_name)

        # Entity Framing Head
        self.entity_framer = EntityFramingClassifier(
            hidden_dim=hidden_dim,
            num_roles=taxonomy.num_roles
        )

        # Graph Building & Message Passing
        if use_graph:
            self.graph_builder = NarrativeGraphBuilder(
                hidden_dim=hidden_dim,
                max_sentences=max_sentences,
                max_entities=max_entities
            )
            self.gnn_layer = HeteroGraphAttentionLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=gnn_heads
            )

        # Narrative Classifier
        self.narrative_classifier = HierarchicalNarrativeClassifier(
            hidden_dim=hidden_dim,
            num_narratives=taxonomy.num_narratives,
            num_subnarratives=taxonomy.num_subnarratives
        )

        # Evidence Retriever
        self.evidence_retriever = NarrativeEvidenceRetriever(
            hidden_dim=hidden_dim,
            max_sentences=max_sentences
        )

        # Structured Alignment Loss
        if use_alignment:
            self.alignment_loss_fn = StructuredAlignmentLoss(hidden_dim=hidden_dim)

        # Grounded Generator
        self.generator = RuleBasedGroundedGenerator(hidden_dim=hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        sent_mask = batch["sent_mask"]

        # 1. Encode text
        doc_emb, token_embeddings = self.encoder(input_ids, attention_mask)

        # 2. Entity Framing Prediction
        role_logits, entity_role_emb = self.entity_framer(doc_emb)

        # 3. Heterogeneous Graph Reasoning (optional via use_graph)
        refined_doc_emb = doc_emb
        if self.use_graph:
            nodes, adj = self.graph_builder(doc_emb, entity_role_emb, sent_mask)
            updated_nodes = self.gnn_layer(nodes, adj)
            refined_doc_emb = updated_nodes[:, 0, :]  # Extract updated Document Node representation

        # 4. Narrative & Subnarrative Prediction
        narrative_logits, subnarrative_logits = self.narrative_classifier(refined_doc_emb)

        # 5. Evidence Retrieval Scoring
        evidence_scores = self.evidence_retriever(refined_doc_emb, max_sents=sent_mask.size(1))

        # 6. Compute Alignment Loss if enabled
        alignment_loss = torch.tensor(0.0, device=doc_emb.device)
        if self.use_alignment:
            alignment_loss = self.alignment_loss_fn(
                entity_role_emb=entity_role_emb,
                doc_emb=refined_doc_emb,
                narrative_logits=narrative_logits,
                subnarrative_logits=subnarrative_logits
            )

        return {
            "doc_emb": refined_doc_emb,
            "role_logits": role_logits,
            "narrative_logits": narrative_logits,
            "subnarrative_logits": subnarrative_logits,
            "evidence_scores": evidence_scores,
            "alignment_loss": alignment_loss
        }
