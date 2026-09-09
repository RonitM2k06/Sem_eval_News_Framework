"""
Heterogeneous Graph Neural Network Layer using multi-head Graph Attention (GATv2).
Operates over heterogeneous nodes: Document, Sentence, Entity, Role, Narrative, Subnarrative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroGraphAttentionLayer(nn.Module):
    """
    Multi-head heterogeneous graph attention layer updating node representations through message passing.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.attn)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        node_features: (batch_size, num_nodes, in_dim)
        adj_matrix: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = node_features.size()

        h_src = self.lin_src(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        h_dst = self.lin_dst(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Self-attention over nodes
        h_src_exp = h_src.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        h_dst_exp = h_dst.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)

        cat_feat = torch.cat([h_src_exp, h_dst_exp], dim=-1)  # (batch, num_nodes, num_nodes, heads, 2*head_dim)
        attn_scores = (cat_feat * self.attn).sum(dim=-1)  # (batch, num_nodes, num_nodes, heads)
        attn_scores = F.leaky_relu(attn_scores, 0.2)

        # Mask non-adjacent nodes
        mask = adj_matrix.unsqueeze(-1).expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=2)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of neighbor features
        out = (attn_weights.unsqueeze(-1) * h_dst_exp).sum(dim=2)  # (batch, num_nodes, heads, head_dim)
        out = out.reshape(batch_size, num_nodes, self.out_dim)

        # Residual connection + Norm
        out = F.layer_norm(out + self.lin_src(node_features), [self.out_dim])
        return out
