# Dashboard & Research Platform Documentation: Project NarrativeGraph

This document describes the interactive research dashboard, backend FastAPI endpoints, data architecture, Professor/Researcher views, and live presentation walkthrough for **NarrativeGraph**.

---

## 1. Architecture & Component Overview

- **FastAPI Backend Engine (`backend/`)**: REST APIs delivering live article analysis (`/api/analyze`), model inference, result metrics (`/api/results/*`), experiment registry (`/api/experiments`), research trace (`/api/experiments/trace/{metric_key}`), paper draft metadata (`/api/paper`), and reproducibility guides (`/api/reproducibility`).
- **Interactive Academic Dashboard (`frontend/`)**: Modern single-page web dashboard with responsive layout, SVG/canvas graph renderer, Professor vs Researcher view mode switcher, and interactive research trace modal.

---

## 2. Main Dashboard Navigation & Views

1. **Professor View (5-Minute Executive Mode)**:
   - Tailored for high-level academic presentations.
   - Flow: Research Overview $\to$ Live Article Analysis $\to$ NarrativeGraph Canvas $\to$ Benchmark Comparison $\to$ Paper Verification.
2. **Researcher View (Deep Scientific Inspection Mode)**:
   - Exposes raw seeds ($42, 123, 2025$), YAML configurations, loss components ($L_{\text{alignment}}$), statistical significance ($p < 0.001$), ECE calibration curves, model checkpoints, and interactive research trace.
3. **Interactive NarrativeGraph Explorer**:
   - SVG canvas visualizing Document, Sentence, Entity, Role, Narrative, and Subnarrative nodes with GATv2 message-passing edges and evidence highlighting.
4. **Interactive Research Trace**:
   - Clicking any metric in tables or charts opens a modal displaying the exact raw JSON output, script command, model checkpoint, random seed, and LaTeX paper reference.

---

## 3. Backend API Endpoint Reference

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/api/health` | GET | Health check status & PyTorch GPU availability |
| `/api/project` | GET | High-level project summary and hypothesis |
| `/api/research/questions` | GET | Research Questions RQ1--RQ10 status and empirical findings |
| `/api/results/main` | GET | Main benchmark model comparison (B0--B5 vs NarrativeGraph) |
| `/api/results/ablations` | GET | 10-way ablation delta scores |
| `/api/results/cross-lingual` | GET | Cross-lingual transfer & LOLO scores across 5 languages |
| `/api/results/faithfulness` | GET | Entailment, coverage, and hallucination reduction metrics |
| `/api/results/alignment` | GET | Bilinear compatibility energy scores |
| `/api/results/low-resource` | GET | Learning curves at 5%, 10%, 25%, 50%, 100% data budgets |
| `/api/experiments` | GET | Searchable experiment registry list |
| `/api/experiments/trace/{metric_key}` | GET | Provenance trace object for any metric |
| `/api/analyze` | POST | Live PyTorch inference over custom article text input |
| `/api/analyze/demo` | GET | Curated multilingual demo sample articles |
| `/api/paper` | GET | LaTeX paper draft metadata and section breakdown |
| `/api/reproducibility` | GET | Step-by-step reproduction command protocol |
| `/api/scorecard` | GET | Phase 2 Research Scorecard metrics |

---

## 4. How to Launch & Run the Dashboard

```bash
# Launch unified dashboard server (FastAPI + Frontend)
python scripts/run_dashboard.py
```

Access the dashboard at `http://127.0.0.1:8000` and API documentation at `http://127.0.0.1:8000/docs`.
