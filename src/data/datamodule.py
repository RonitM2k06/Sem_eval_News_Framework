"""
PyTorch DataModule for managing training, validation, and test dataloaders.
"""

from typing import Optional, Dict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.taxonomy.parser import TaxonomyParser
from src.data.dataset import NarrativeDataset


class NarrativeDataModule:
    """
    Encapsulates dataset setup and dataloader creation across splits.
    """

    def __init__(
        self,
        train_path: str,
        dev_path: str,
        test_path: str,
        tokenizer: AutoTokenizer,
        taxonomy: TaxonomyParser,
        batch_size: int = 2,
        max_seq_len: int = 256,
        max_sentences: int = 8,
        max_entities: int = 6,
        num_workers: int = 0
    ):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_sentences = max_sentences
        self.max_entities = max_entities
        self.num_workers = num_workers

        self.train_dataset: Optional[NarrativeDataset] = None
        self.dev_dataset: Optional[NarrativeDataset] = None
        self.test_dataset: Optional[NarrativeDataset] = None

    def setup(self):
        self.train_dataset = NarrativeDataset(
            self.train_path,
            self.tokenizer,
            self.taxonomy,
            max_seq_len=self.max_seq_len,
            max_sentences=self.max_sentences,
            max_entities=self.max_entities
        )
        self.dev_dataset = NarrativeDataset(
            self.dev_path,
            self.tokenizer,
            self.taxonomy,
            max_seq_len=self.max_seq_len,
            max_sentences=self.max_sentences,
            max_entities=self.max_entities
        )
        self.test_dataset = NarrativeDataset(
            self.test_path,
            self.tokenizer,
            self.taxonomy,
            max_seq_len=self.max_seq_len,
            max_sentences=self.max_sentences,
            max_entities=self.max_entities,
            is_test=True
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
