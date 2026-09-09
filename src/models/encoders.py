"""
Encoder Module wrapping Multilingual Transformers (XLM-RoBERTa, mBERT) for text and sentence representations.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MultilingualTextEncoder(nn.Module):
    """
    Multilingual transformer encoder providing token-level, sentence-level, and document-level embeddings.
    """

    def __init__(self, model_name: str = "xlm-roberta-base", freeze_layers: int = 0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.config.hidden_size

        if freeze_layers > 0 and hasattr(self.transformer, "encoder"):
            for layer in self.transformer.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling over token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        doc_embedding = sum_embeddings / sum_mask
        return doc_embedding, outputs.last_hidden_state
