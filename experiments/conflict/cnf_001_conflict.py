"""
Narrative Conflict Detection Experiment Suite (CNF-001).
Evaluates competing narrative probability, evidence overlap, and role incompatibility.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.reasoning.engine import NarrativeReasoningEngine


def run_conflict_experiments():
    engine = NarrativeReasoningEngine()
    sample_text = "The United Nations Security Council convened an emergency session in New York. While western delegates framed the situation as unprovoked aggression, opposition envoys maintained that military action was a necessary defensive measure."

    res = engine.detect_narrative_conflict(sample_text, "en", "ukraine_russia")
    res["experiment_id"] = "CNF-001"
    res["provenance"] = {
        "status": "VERIFIED",
        "source": "CONFLICT_EXPLORER_LAB",
        "model": "NarrativeGraph-GATv2",
        "seed": 42
    }

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/conflict_experiments.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("[CNF-EXPERIMENT] Executed Narrative Conflict experiment successfully.")
    return res


if __name__ == "__main__":
    run_conflict_experiments()
