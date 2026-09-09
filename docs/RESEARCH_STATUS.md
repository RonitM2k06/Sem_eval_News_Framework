# Authoritative Research Component Status Page

This document provides the definitive verification status of all theoretical, experimental, software, and evaluation components within Project NarrativeGraph.

---

## 1. Research Status Matrix

| Component | Category | Status | Verification Evidence | Notes / Constraints |
|---|---|:---:|---|---|
| **Heterogeneous Graph Formulation** | Core Method | **COMPLETE** | `src/models/narrative_graph_model.py` | Full PyTorch GATv2 forward pass operational |
| **Structured Alignment Loss ($\mathcal{L}_{\text{align}}$)** | Core Method | **COMPLETE** | `src/models/alignment_loss.py` | Cross-level compatibility penalty validated |
| **Entity Framing Classifier (ST1)** | Core Subtask | **COMPLETE** | `src/models/` & `backend/services/` | Multi-word phrase grouping across 5 languages |
| **Hierarchical Narrative Classifier (ST2)** | Core Subtask | **COMPLETE** | `src/taxonomy/parser.py` | 2-level parent & subnarrative tree parsing |
| **Evidence RAG Grounding (ST3)** | Core Subtask | **COMPLETE** | `src/evaluation/faithfulness.py` | Reduces hallucination from 18.2% to 4.8% |
| **Main Benchmark Evaluation** | Experiments | **COMPLETE** | `results/main_results.csv` | Macro F1 = 0.7680 across 3 seeds |
| **Component Ablation Suite** | Experiments | **COMPLETE** | `results/tables/ablations.csv` | Isolates GNN (-0.060) and L_align (-0.052) |
| **Cross-Lingual LOLO Transfer** | Experiments | **COMPLETE** | `results/tables/cross_lingual.csv` | Evaluates bg, en, hi, pt, ru transfer |
| **Low-Resource Data Regimes (5%-25%)** | Experiments | **COMPLETE** | `results/figures/low_resource_curves.png` | 82.4% retention at 25% data budget |
| **Statistical Significance Testing** | Statistics | **COMPLETE** | `src/analysis/significance.py` | Paired bootstrap $p < 0.001$ verified |
| **Counterfactual Sensitivity (CNS)** | Reasoning | **COMPLETE** | `experiments/counterfactual/` | CNS = 0.8420 on entity removals |
| **Evidence Necessity & Sufficiency** | Reasoning | **COMPLETE** | `experiments/evidence_grounding/` | ENS = 0.7810, ESS = 0.8650, EMS top-2 |
| **Heuristic Narrative Conflict (NCS)** | Reasoning | **COMPLETE** | `experiments/conflict/` | NCS = 0.6140 multi-frame scoring |
| **Cross-Lingual Consistency (CLSC)** | Reasoning | **COMPLETE** | `experiments/cross_lingual/` | CLSC = 0.8340 across 5 language pairs |
| **Adversarial Robustness (PRS)** | Reasoning | **COMPLETE** | `experiments/adversarial/` | PRS = 0.8920 across 10 perturbations |
| **Temporal Trajectory Tracking** | Reasoning | **PARTIALLY COMPLETE** | `experiments/temporal/` | Framework operational; empirical multi-year evaluation marked PENDING REAL TEMPORAL DATA |
| **Human Expert Evaluation** | Evaluation | **PENDING** | `docs/FINAL_LIMITATIONS.md` | Automated metrics (NLI, BERTScore) utilized; human domain expert panel not conducted |
| **Interactive Research Dashboard** | Engineering | **COMPLETE** | `frontend/`, `backend/` | 15-column Baseline Ladder & 6 Reasoning Labs |
| **LaTeX Conference Paper** | Publication | **COMPLETE** | `paper/main.tex` | ACL camera-ready formatted, 22 sections |
| **End-to-End Master Validation** | Validation | **COMPLETE** | `scripts/validate_project.py` | All 7 system audits pass with exit code 0 |

---

## 2. Status Classification Summary
- **COMPLETE**: 18 components
- **PARTIALLY COMPLETE**: 1 component (Temporal trajectory tracking framework operational, awaiting multi-year dataset timestamps)
- **PENDING**: 1 component (Human domain expert panel)
- **BLOCKED**: 0 components
