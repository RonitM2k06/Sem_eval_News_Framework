"""
Training Script for NarrativeGraph and Baselines.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import argparse
import torch
from transformers import AutoTokenizer
from src.taxonomy.parser import TaxonomyParser
from src.data.datamodule import NarrativeDataModule
from src.models.narrative_graph_model import NarrativeGraphModel
from src.training.trainer import NarrativeGraphTrainer


def run_training(config_path: str, seed: int = 42):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[Train] Running experiment: {config['experiment_name']} (Seed: {seed})")
    torch.manual_seed(seed)

    taxonomy = TaxonomyParser()
    model_name = config["model"].get("encoder_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    processed_train = os.path.join(config["data"]["data_dir"], "processed/train.json")
    processed_dev = os.path.join(config["data"]["data_dir"], "processed/dev.json")
    processed_test = os.path.join(config["data"]["data_dir"], "processed/test.json")

    if os.path.exists(processed_train):
        train_path = processed_train
        dev_path = processed_dev if os.path.exists(processed_dev) else processed_train
        test_path = processed_test if os.path.exists(processed_test) else dev_path
        print(f"[Train] Using REAL SemEval 2025 Task 10 dataset: {train_path}")
    else:
        train_path = os.path.join(config["data"]["data_dir"], "synthetic/synthetic_train.json")
        dev_path = os.path.join(config["data"]["data_dir"], "synthetic/synthetic_dev.json")
        test_path = os.path.join(config["data"]["data_dir"], "synthetic/synthetic_test.json")
        print(f"[Train] Using SYNTHETIC dataset fallback: {train_path}")

    datamodule = NarrativeDataModule(
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        tokenizer=tokenizer,
        taxonomy=taxonomy,
        batch_size=config["data"].get("batch_size", 2),
        max_seq_len=config["data"].get("max_seq_len", 256),
        max_sentences=config["data"].get("max_sentences", 8),
        max_entities=config["data"].get("max_entities", 6)
    )
    datamodule.setup()

    model = NarrativeGraphModel(
        taxonomy=taxonomy,
        model_name=model_name,
        hidden_dim=config["model"].get("hidden_dim", 768),
        gnn_dim=config["model"].get("gnn_dim", 256),
        use_graph=config["model"].get("use_graph", True),
        use_alignment=config["model"].get("use_alignment", True)
    )

    save_dir = f"experiments/{config['experiment_name']}_seed{seed}"
    trainer = NarrativeGraphTrainer(
        model=model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader(),
        epochs=config["training"].get("epochs", 3),
        lr=float(config["training"].get("lr", 2e-5)),
        save_dir=save_dir
    )

    best_metrics = trainer.fit()
    print(f"[Train] Finished {config['experiment_name']} | Best Val Macro F1: {best_metrics.get('macro_f1', 0.0):.4f}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/narrativegraph.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(args.config, args.seed)
