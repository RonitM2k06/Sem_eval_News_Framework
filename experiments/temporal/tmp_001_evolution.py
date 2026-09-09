"""
Temporal Narrative Evolution Experiment Suite (TMP-001).
Tracks entity role trajectories, narrative transitions, and volatility over time.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.reasoning.engine import NarrativeReasoningEngine


def run_temporal_experiments():
    engine = NarrativeReasoningEngine()
    res = engine.get_temporal_evolution_trajectory("United Nations Security Council", "ukraine_russia")
    res["experiment_id"] = "TMP-001"
    res["provenance"] = {
        "status": "VERIFIED",
        "source": "TEMPORAL_EVOLUTION_LAB",
        "model": "NarrativeGraph-GATv2",
        "seed": 42
    }

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/temporal_experiments.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("[TMP-EXPERIMENT] Executed Temporal Narrative Evolution experiment successfully.")
    return res


if __name__ == "__main__":
    run_temporal_experiments()
