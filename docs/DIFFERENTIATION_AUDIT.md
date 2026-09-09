# NarrativeGraph: Research Differentiation & Novelty Audit

| Capability | Existing? | Quality | Research Valid? | Reusable? | Needs Extension |
|---|---|---|---|---|---|
| **Multilingual Alignment Pipeline** | Yes | High | Yes | Yes | Baseline established |
| **Entity Framing & Role Mining** | Yes | High | Yes | Yes | Baseline established |
| **Hierarchical Narrative Tree** | Yes | High | Yes | Yes | Baseline established |
| **Evidence RAG Retrieval** | Yes | Medium | Partial | Yes | Needs Necessity/Sufficiency |
| **1. Counterfactual Reasoning** | No | — | — | No | **Needs Full Implementation** |
| **2. Evidence Necessity & Sufficiency** | No | — | — | No | **Needs Full Implementation** |
| **3. Narrative Conflict Detection** | No | — | — | No | **Needs Full Implementation** |
| **4. Cross-Lingual Consistency** | Partial | Medium | Partial | Yes | **Needs Graph Similarity & Transfer Matrix** |
| **5. Adversarial Robustness** | No | — | — | No | **Needs Full Perturbation Suite** |
| **6. Temporal Narrative Evolution** | No | — | — | No | **Needs Full Engine & Timeline Trajectory** |

---

## Strategic Extension Plan
1. **Unified Structural Reasoning Engine** (`src/reasoning/engine.py`)
2. **Graph Difference Engine** (`src/reasoning/graph_diff.py`)
3. **Narrative Reasoning Research Metrics** (`src/metrics/narrative_reasoning.py`)
4. **Interactive Research Labs & API Endpoints** (FastAPI `backend/router/reasoning.py` & Frontend Tabs)
5. **Experiment Suite & Provenance Traceability** (`experiments/`)
6. **Documentation & Camera-Ready LaTeX Paper Integration** (`docs/` & `paper/main.tex`)
