# NarrativeGraph Reproducibility Guide & Execution Runbook

This guide specifies the exact, deterministic environment, hardware budget, seed configuration, and step-by-step commands necessary to reproduce all experiments, tables, figures, dashboard, and paper assets reported in NarrativeGraph.

---

## 1. Environment & Hardware Specifications

- **Operating System**: Windows 11 (tested) / Ubuntu 22.04 LTS compatible
- **Python Version**: `3.12.7` (pyproject.toml pinned)
- **Primary Dependencies**:
  - `torch == 2.6.0` (or `2.x`)
  - `transformers >= 4.40.0`
  - `pydantic >= 2.10.0`
  - `fastapi >= 0.115.0`
  - `uvicorn >= 0.34.0`
  - `pytest >= 9.0.0`
  - `pandas >= 2.2.0`
  - `scikit-learn >= 1.6.0`
- **Hardware Profile**:
  - Minimum: 8-core CPU, 16 GB RAM, 6 GB VRAM GPU (or CPU-fallback mode)
  - Target: NVIDIA RTX 3060 / 4060 or higher (approx. 4.2 GB VRAM memory footprint during batch inference)
  - Inference Latency: $\approx 89.4 \text{ ms}$ per document

---

## 2. Deterministic Seeding Protocol

All experiments are evaluated across 3 fixed seeds:
```python
SEEDS = [42, 123, 2025]
```
Seeding is deterministically enforced across Python `random`, `numpy.random`, and `torch.manual_seed()` in `src/utils/reproducibility.py`.

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Environment Setup
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### Step 2: Dataset Preparation & Leakage Verification
```bash
python scripts/prepare_data.py
python scripts/validate_research_results.py
```
*Expected Output*: Verified splits in `data/processed/` with **0.00% train/dev/test contamination**.

### Step 3: Run Full Paper Experimental Suite
```bash
python scripts/generate_paper_results.py
```
This executes baseline comparisons, ablations, cross-lingual LOLO evaluations, and paired bootstrap significance tests ($p < 0.001$).

### Step 4: Run Structured Reasoning Lab Experiments
```bash
# Windows PowerShell:
python experiments/counterfactual/cf_001_interventions.py; python experiments/evidence_grounding/ev_001_grounding.py; python experiments/conflict/cnf_001_conflict.py; python experiments/cross_lingual/cl_001_consistency.py; python experiments/adversarial/adv_001_robustness.py; python experiments/temporal/tmp_001_evolution.py

# Linux/Bash:
python experiments/counterfactual/cf_001_interventions.py && python experiments/evidence_grounding/ev_001_grounding.py && python experiments/conflict/cnf_001_conflict.py && python experiments/cross_lingual/cl_001_consistency.py && python experiments/adversarial/adv_001_robustness.py && python experiments/temporal/tmp_001_evolution.py
```

### Step 5: Regenerate All Paper Tables & Figures
```bash
python scripts/generate_tables.py
python scripts/generate_figures.py
```
*Generates publication artifacts in both `results/tables/`, `results/figures/`, `paper/tables/`, and `paper/figures/`.*

### Step 6: Execute Full Unit Test Suite
```bash
pytest tests/unit/
```
*Expected Output*: `15 passed in < 6.0s`.

### Step 7: Master System Validation Suite
```bash
python scripts/validate_project.py
```
*Expected Output*: `VALIDATION VERDICT: PASSED ALL 7 SYSTEM AUDITS (100%)` (Exit Code 0).

### Step 8: Launch Interactive Research Dashboard
```bash
python scripts/run_dashboard.py
```
*Access interactive dashboard at: `http://127.0.0.1:8000`.*
