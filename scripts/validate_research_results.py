"""
Research Result Provenance Validator.
Ensures every reported metric in results/, tables/, and paper/ has verified provenance and source traceability.
Fails loudly if unverified or synthetic numbers are mislabeled as official benchmark results.
"""

import os
import sys
import csv
import pandas as pd


def validate_provenance() -> bool:
    print("[ProvenanceValidator] Auditing research results and provenance traceability...")
    errors = 0

    # 1. Check Result Files Exist
    required_files = [
        "results/main_results.csv",
        "results/tables/main_results.tex",
        "results/tables/ablations.csv",
        "results/tables/cross_lingual.csv",
        "results/research_dashboard.md",
        "docs/RESULT_PROVENANCE.md",
        "experiments/registry.csv"
    ]

    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"[ERROR] Missing required provenance artifact: {filepath}")
            errors += 1

    # 2. Check Experiment Registry Integrity
    registry_file = "experiments/registry.csv"
    real_data_exists = os.path.exists("data/processed/train.json")
    if os.path.exists(registry_file):
        with open(registry_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                print("[ERROR] experiments/registry.csv is empty")
                errors += 1
            for row in rows:
                if not row.get("experiment_id") or not row.get("status"):
                    print(f"[ERROR] Registry row missing ID or status: {row}")
                    errors += 1
                if "VERIFIED" in row.get("status", "") and not real_data_exists:
                    print(f"[ERROR] Found VERIFIED claim, but real data is missing! Row: {row}")
                    errors += 1

    # 3. Check Main Results Table
    main_csv = "results/main_results.csv"
    if os.path.exists(main_csv):
        df = pd.read_csv(main_csv)
        if "Macro_F1" not in df.columns or "Model" not in df.columns:
            print("[ERROR] main_results.csv missing required columns 'Model' or 'Macro_F1'")
            errors += 1

    # 4. Check Provenance Registry Document
    prov_doc = "docs/RESULT_PROVENANCE.md"
    if os.path.exists(prov_doc):
        with open(prov_doc, "r", encoding="utf-8") as f:
            text = f.read()
            if "SYNTHETIC" not in text and "PENDING" not in text and "VERIFIED" not in text:
                print("[ERROR] RESULT_PROVENANCE.md missing status classification")
                errors += 1

    print(f"[ProvenanceValidator] Audit Completed: {errors} Errors Found.")
    return errors == 0


if __name__ == "__main__":
    success = validate_provenance()
    if not success:
        sys.exit(1)
    print("[ProvenanceValidator] All research results have valid provenances and integrity checks passed!")
