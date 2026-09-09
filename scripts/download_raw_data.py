"""
Raw Data Ingestion & Downloader Helper for SemEval 2025 Task 10.

Note: The official dataset is distributed via the SemEval Task 10 portal (https://propaganda.math.unipd.it/semeval2025task10/)
and requires participant registration / login credentials.

This script guides the raw data ingestion process into data/raw/.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_and_ingest_raw_data(data_dir: str = "data/raw"):
    os.makedirs(data_dir, exist_ok=True)
    raw_files = [f for f in os.listdir(data_dir) if f.endswith(".json") or f.endswith(".tsv") or f.endswith(".csv")]
    
    print("=================================================================")
    print("        SEMEVAL 2025 TASK 10: RAW DATA INGESTION AUDITOR        ")
    print("=================================================================")
    print(f"Target Directory: {os.path.abspath(data_dir)}")
    
    if not raw_files:
        print("\n[WARNING] No raw competition files (.json, .tsv, .csv) found in data/raw/!")
        print("\nHow to obtain the official dataset:")
        print("1. Visit the official task portal: https://propaganda.math.unipd.it/semeval2025task10/")
        print("2. Log in with your registered SemEval 2025 participant account.")
        print("3. Download the official data packages (e.g., train.json, dev.json, test.json).")
        print("4. Copy/move the downloaded files directly into: data/raw/\n")
        print("Once files are placed in data/raw/, run:")
        print("   python scripts/prepare_data.py")
        print("   python scripts/generate_paper_results.py")
        print("=================================================================")
        return False

    print(f"\n[SUCCESS] Detected {len(raw_files)} raw data files in data/raw/:")
    for f in raw_files:
        path = os.path.join(data_dir, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")
        
    print("\nNext step: Execute 'python scripts/prepare_data.py' to process raw splits.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemEval Task 10 Raw Data Auditor")
    parser.add_argument("--dir", type=str, default="data/raw", help="Directory containing raw competition files")
    args = parser.parse_args()
    check_and_ingest_raw_data(args.dir)
