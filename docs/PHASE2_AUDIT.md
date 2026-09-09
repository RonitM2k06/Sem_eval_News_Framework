# Phase 2 Codebase & Research Audit: Project NarrativeGraph

This audit evaluates the current state of Project NarrativeGraph following Phase 1 execution, identifying genuine implementations, synthetic/placeholder components, verified metrics, and required repairs for Phase 2.

---

## 1. Inventory of Existing Components

| Component | File Path | Status | Execution Verdict | Action Required |
| --------- | --------- | ------ | ----------------- | --------------- |
| **Multilingual Text Encoder** | `src/models/encoders.py` | Genuine | Working (`xlm-roberta-base` wrapper with mean pooling) | Retain & optimize |
| **Entity Framing Classifier** | `src/models/entity_framer.py` | Genuine | Working (Multi-label span role classification) | Retain |
| **Hierarchical Narrative Head** | `src/models/narrative_classifier.py` | Genuine | Working (Two-level narrative & subnarrative tree prediction) | Retain |
| **Evidence Retriever** | `src/models/evidence_retriever.py` | Genuine | Working (Sentence-level narrative conditioning network) | Retain |
| **Heterogeneous Graph Builder** | `src/graph/builder.py` | Genuine | Working (Document, Sentence, Entity, Role, Narrative, Subnarrative nodes) | Retain |
| **GATv2 Graph Attention Layer** | `src/graph/gnn_layer.py` | Genuine | Working (Multi-head heterogeneous graph message passing) | Retain |
| **Structured Alignment Loss** | `src/alignment/structured_loss.py` | Genuine | Working (`F.binary_cross_entropy_with_logits` energy loss) | Retain |
| **Explanation Generator** | `src/models/explanation_generator.py` | Demonstration | Working (`RuleBasedGroundedGenerator` template engine) | Label explicitly as `RULE-BASED DEMONSTRATION GENERATOR` |
| **Synthetic Dataset Generator** | `src/data/synthetic.py` | Synthetic | Working (Generates bg, en, hi, pt, ru articles across 2 domains) | Keep as synthetic offline fixture |
| **Data Leakage Detector** | `src/data/leakage.py` | Genuine | Working (Report generated in `docs/DATA_LEAKAGE_REPORT.md`) | Retain |
| **PyTorch DataModule & Dataset** | `src/data/dataset.py`, `src/data/datamodule.py` | Genuine | Working (Tokenization & batching) | Retain |
| **Trainer Pipeline** | `src/training/trainer.py` | Genuine | Working (AdamW, linear warmup, `torch.amp.autocast('cuda')`) | Retain |
| **Metrics & Faithfulness Evaluators** | `src/evaluation/metrics.py`, `src/evaluation/faithfulness.py` | Genuine | Working (Macro F1, Accuracy, BERTScore, Entailment, Hallucination) | Retain |
| **Statistical Significance** | `src/analysis/significance.py` | Genuine | Working (Paired Bootstrap Resampling test) | Retain |

---

## 2. Research Integrity Audit: Data & Metrics Truthfulness

1. **Synthetic vs Benchmark Data**: The codebase includes `src/data/synthetic.py` to generate multilingual test fixtures. Training passes run via `scripts/generate_paper_results.py` execute on synthetic data (`data/synthetic/`).
2. **Metric Provenance**: In `results/tables/main_results.tex`, baseline paper scores (e.g. `0.7680 F1`) represent target published values, whereas actual synthetic training runs achieve `~0.0245 - 0.0361 F1`.
3. **Scientific Directive**: The platform must explicitly label synthetic runs as `SYNTHETIC / SMOKE TEST` or `DEMO DATA` and mark official benchmark data as `PENDING REAL-DATA VALIDATION` unless raw SemEval 2025 Task 10 data is loaded into `data/raw/`. Zero fabricated claims are permitted.

---

## 3. Required Phase 2 Redesigns & Additions

1. **Authoritative Result Provenance (`docs/RESULT_PROVENANCE.md`)**: A single source of truth mapping every displayed metric to raw output, script, model checkpoint, and verified status.
2. **Programmatic Provenance Validator (`scripts/validate_research_results.py`)**: Automated script enforcing provenance rules.
3. **FastAPI Backend Server (`backend/`)**: REST APIs providing real model inference, demo samples, experiment registry, research trace, and paper metadata.
4. **Polished Interactive Next.js/React Dashboard (`frontend/`)**: Academic presentation dashboard with Professor View vs Researcher View, live article analysis, interactive NarrativeGraph explorer, evidence explorer, and research trace modal.
