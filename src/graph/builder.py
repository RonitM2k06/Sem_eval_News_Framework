"""
Heterogeneous NarrativeGraph Builder.
Constructs node feature tensors and adjacency matrices connecting Document, Sentences, Entities, Roles, and Narratives.
"""

from typing import Tuple
import torch
import torch.nn as nn


class NarrativeGraphBuilder(nn.Module):
    """
    Assembles initial node features and builds adjacency graphs per batch element.
    Nodes order per sample:
    0: Document Node
    1..S: Sentence Nodes (S = max_sentences)
    S+1..S+E: Entity Nodes (E = max_entities)
    S+E+1: Role Node
    S+E+2: Narrative Node
    S+E+3: Subnarrative Node
    Total Nodes: 1 + S + E + 3
    """

    def __init__(self, hidden_dim: int, max_sentences: int = 16, max_entities: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_sentences = max_sentences
        self.max_entities = max_entities
        self.total_nodes = 1 + max_sentences + max_entities + 3

        # Type embeddings for nodes
        self.type_embed = nn.Embedding(6, hidden_dim)

    def forward(
        self,
        doc_emb: torch.Tensor,
        entity_role_emb: torch.Tensor,
        sent_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = doc_emb.size(0)
        num_sents = min(self.max_sentences, sent_mask.size(1))

        # Initialize node feature matrix: (batch_size, total_nodes, hidden_dim)
        nodes = torch.zeros(batch_size, self.total_nodes, self.hidden_dim, device=doc_emb.device)

        # Node 0: Document
        nodes[:, 0, :] = doc_emb + self.type_embed(torch.tensor(0, device=doc_emb.device))

        # Nodes 1..S: Sentences
        for s in range(self.max_sentences):
            nodes[:, 1 + s, :] = doc_emb * 0.8 + self.type_embed(torch.tensor(1, device=doc_emb.device))

        # Nodes S+1..S+E: Entities
        offset = 1 + self.max_sentences
        for e in range(self.max_entities):
            nodes[:, offset + e, :] = entity_role_emb * 0.9 + self.type_embed(torch.tensor(2, device=doc_emb.device))

        # Node S+E+1: Role Node
        nodes[:, offset + self.max_entities, :] = entity_role_emb + self.type_embed(torch.tensor(3, device=doc_emb.device))

        # Node S+E+2: Narrative Node
        nodes[:, offset + self.max_entities + 1, :] = doc_emb * 0.95 + self.type_embed(torch.tensor(4, device=doc_emb.device))

        # Node S+E+3: Subnarrative Node
        nodes[:, offset + self.max_entities + 2, :] = doc_emb * 0.85 + self.type_embed(torch.tensor(5, device=doc_emb.device))

        # Build Adjacency Matrix (batch_size, total_nodes, total_nodes)
        adj = torch.eye(self.total_nodes, device=doc_emb.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Edges: Document <-> Sentences
        for s in range(num_sents):
            adj[:, 0, 1 + s] = sent_mask[:, s]
            adj[:, 1 + s, 0] = sent_mask[:, s]

        # Edges: Document <-> Entity, Entity <-> Role
        role_idx = offset + self.max_entities
        for e in range(self.max_entities):
            adj[:, 0, offset + e] = 1.0
            adj[:, offset + e, 0] = 1.0
            adj[:, offset + e, role_idx] = 1.0
            adj[:, role_idx, offset + e] = 1.0

        # Edges: Role <-> Narrative, Narrative <-> Subnarrative
        narr_idx = role_idx + 1
        subnarr_idx = role_idx + 2
        adj[:, role_idx, narr_idx] = 1.0
        adj[:, narr_idx, role_idx] = 1.0
        adj[:, narr_idx, subnarr_idx] = 1.0
        adj[:, subnarr_idx, narr_idx] = 1.0

        return nodes, adj
