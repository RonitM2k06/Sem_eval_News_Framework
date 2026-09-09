"""
Official Submission Validator for SemEval 2025 Task 10.
Validates submission files for Subtask 1 (Entity Framing), Subtask 2 (Narrative Classification), and Subtask 3 (Narrative Extraction).
"""

import sys
import json
import argparse
from typing import Dict, Any, List


def validate_submission_file(filepath: str) -> bool:
    print(f"[SubmissionValidator] Validating file: {filepath}")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Invalid JSON file format: {e}")
        return False

    if not isinstance(data, list):
        print("[ERROR] Root submission object must be a list of predictions.")
        return False

    errors = 0
    warnings = 0

    for i, item in enumerate(data):
        article_id = item.get("article_id")
        if not article_id:
            print(f"[ERROR] Item {i} missing 'article_id'")
            errors += 1

        # Subtask 1 Validation (Entity Framing)
        entities = item.get("entities", [])
        for ent in entities:
            if "mention" not in ent or "roles" not in ent:
                print(f"[ERROR] Item {i} entity missing mention or roles")
                errors += 1

        # Subtask 2 Validation (Narrative Classification)
        if "narrative" not in item:
            print(f"[WARNING] Item {i} missing narrative prediction")
            warnings += 1

        # Subtask 3 Validation (Explanation <= 80 words)
        explanation = item.get("explanation", "")
        if explanation:
            word_count = len(explanation.split())
            if word_count > 80:
                print(f"[ERROR] Item {i} explanation exceeds 80 words limit ({word_count} words)")
                errors += 1

    print(f"[SubmissionValidator] Result: {errors} Errors, {warnings} Warnings")
    return errors == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SemEval 2025 Task 10 submission file.")
    parser.add_argument("--submission_file", type=str, required=True, help="Path to submission file")
    args = parser.parse_args()

    success = validate_submission_file(args.submission_file)
    if not success:
        sys.exit(1)
    print("[SubmissionValidator] Submission file is VALID!")
