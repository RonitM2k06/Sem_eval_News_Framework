"""
FastAPI Backend Configuration Settings.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    app_name: str = "NarrativeGraph Research Platform API"
    version: str = "2.0.0"
    debug: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Model and Data Paths
    model_name: str = "xlm-roberta-base"
    checkpoint_path: str = "experiments/narrativegraph_full_seed42/best_model.pt"
    synthetic_data_path: str = "data/synthetic/synthetic_dev.json"
    results_dir: str = "results"
    docs_dir: str = "docs"


settings = Settings()
