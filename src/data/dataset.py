"""
PyTorch Dataset for SemEval 2025 Task 10 NarrativeGraph framework.
Handles multilingual text tokenization, sentence splitting, entity offset alignment, and target encoding.
"""

import json
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.taxonomy.parser import TaxonomyParser


class NarrativeDataset(Dataset):
    """
    Multilingual Dataset for Entity Framing, Narrative Classification, Evidence Selection, and Explanation Generation.
    """

    def __init__(
        self,
        filepath: str,
        tokenizer: AutoTokenizer,
        taxonomy: TaxonomyParser,
        max_seq_len: int = 512,
        max_sentences: int = 16,
        max_entities: int = 8,
        is_test: bool = False
    ):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.max_seq_len = max_seq_len
        self.max_sentences = max_sentences
        self.max_entities = max_entities
        self.is_test = is_test

        with open(filepath, "r", encoding="utf-8") as f:
            self.data: List[Dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item["text"]
        sentences = item.get("sentences", [text])[:self.max_sentences]
        entities = item.get("entities", [])[:self.max_entities]

        # Tokenize article
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        narr_val = item.get("narrative", "")
        if isinstance(narr_val, dict):
            narr_str = narr_val.get("parent_narrative", "")
            sub_str = narr_val.get("subnarrative", "")
        else:
            narr_str = str(narr_val)
            sub_str = str(item.get("subnarrative", ""))

        narrative_target = torch.tensor(
            self.taxonomy.encode_narrative(narr_str),
            dtype=torch.long
        )
        subnarrative_target = torch.tensor(
            self.taxonomy.encode_subnarrative(sub_str),
            dtype=torch.long
        )

        # Role targets (multi-hot)
        role_target = torch.zeros(self.taxonomy.num_roles, dtype=torch.float)
        for ent in entities:
            roles = ent.get("roles", [])
            for r_id in self.taxonomy.encode_roles(roles):
                role_target[r_id] = 1.0

        # Evidence sentence target (binary vector over sentences)
        evidence_target = torch.zeros(self.max_sentences, dtype=torch.float)
        for ev_id in item.get("evidence_sentence_ids", []):
            if ev_id < self.max_sentences:
                evidence_target[ev_id] = 1.0

        # Sentence start/end token approximations
        sent_mask = torch.zeros(self.max_sentences, dtype=torch.float)
        for i in range(min(len(sentences), self.max_sentences)):
            sent_mask[i] = 1.0

        return {
            "article_id": item["article_id"],
            "language": item.get("language", "en"),
            "domain": item.get("domain", "ukraine_russia"),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "narrative_target": narrative_target,
            "subnarrative_target": subnarrative_target,
            "role_target": role_target,
            "evidence_target": evidence_target,
            "sent_mask": sent_mask,
            "num_sentences": torch.tensor(len(sentences), dtype=torch.long),
            "explanation_text": item.get("explanation", "")
        }
