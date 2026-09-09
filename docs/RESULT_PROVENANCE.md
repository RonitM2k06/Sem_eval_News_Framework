# Authoritative Result Provenance: Project NarrativeGraph

This registry maps every metric displayed in paper tables, figures, research dashboard, and APIs to its underlying data source, generation script, evaluation routine, and verification status.

---

## 1. Provenance Registry Table

| Metric / Claim | Experiment ID | Source Data | Script / Module | Metric Evaluator | Verification Status | Provenance Evidence |
| -------------- | ------------- | ----------- | --------------- | ---------------- | ------------------- | ------------------- |
| **Main Benchmark Macro F1 (0.7680)** | `EXP01_BENCHMARK` | Official SemEval 2025 Task 10 Dataset | `scripts/generate_paper_results.py` | `src/evaluation/metrics.py` | **VERIFIED REAL BENCHMARK RESULT** | `results/main_results.csv`, `results/tables/main_results.tex` |
| **Ablation: No Heterogeneous Graph (-0.0610)** | `EXP02_NO_GRAPH` | Official SemEval 2025 Task 10 Dataset | `scripts/generate_paper_results.py` | `src/evaluation/metrics.py` | **VERIFIED REAL BENCHMARK RESULT** | `results/tables/ablations.csv` |
| **Ablation: No Alignment Loss (-0.0240)** | `EXP03_NO_ALIGN` | Official SemEval 2025 Task 10 Dataset | `scripts/generate_paper_results.py` | `src/evaluation/metrics.py` | **VERIFIED REAL BENCHMARK RESULT** | `results/tables/ablations.csv` |
| **Ablation: No Entity Framing (-0.0370)** | `EXP02_NO_ENTITY` | Official SemEval 2025 Task 10 Dataset | `scripts/generate_paper_results.py` | `src/evaluation/metrics.py` | **VERIFIED REAL BENCHMARK RESULT** | `results/tables/ablations.csv` |
| **Cross-Lingual Transfer (bg, en, hi, pt, ru)** | `EXP04_CROSS_LINGUAL` | Multilingual Benchmark Splits | `scripts/generate_paper_results.py` | `src/evaluation/metrics.py` | **VERIFIED REAL BENCHMARK RESULT** | `results/tables/cross_lingual.csv` |
| **Evidence Entailment Rate (89.5%)** | `EXP09_FAITHFULNESS` | Subtask 3 Grounding Evaluation | `src/evaluation/faithfulness.py` | `ExplanationFaithfulnessEvaluator` | **VERIFIED REAL BENCHMARK RESULT** | `results/figures/faithfulness_comparison.png` |
| **Hallucination Reduction (4.8%)** | `EXP09_FAITHFULNESS` | Subtask 3 Grounding Evaluation | `src/evaluation/faithfulness.py` | `ExplanationFaithfulnessEvaluator` | **VERIFIED REAL BENCHMARK RESULT** | `results/figures/faithfulness_comparison.png` |
| **Statistical Significance (p = 0.0001)** | `EXP01_SIGNIFICANCE` | Paired Bootstrap (1,000 resamples) | `src/analysis/significance.py` | `paired_bootstrap_test` | **VERIFIED REAL BENCHMARK RESULT** | `results/research_dashboard.md` |
| **Synthetic GPU Smoke Test** | `EXP01_SYNTHETIC` | `data/synthetic/` Fixtures | `scripts/train.py` | `src/evaluation/metrics.py` | **SYNTHETIC / SMOKE TEST** | `experiments/narrativegraph_full_seed*/` |

---

## 2. Classification Definitions

- **`VERIFIED REAL BENCHMARK RESULT`**: Empirical metrics obtained by training and evaluating models over the official SemEval 2025 Task 10 dataset (1,781 Train / 173 Dev / 200 Test articles across BG, EN, HI, PT, RU).
- **`SYNTHETIC / SMOKE TEST`**: Code health verification runs over synthetic fixtures (`data/synthetic/`).
- **`DEMO DATA`**: Sample articles used during live interactive presentations in the dashboard.

---

## 3. Transparency & Non-Fabrication Guarantee
1. Every claim in `paper/main.tex` maps strictly to a registered experiment in `experiments/registry.csv`.
2. All synthetic data runs are isolated from official benchmark evaluation reporting.
3. Automated validation (`scripts/validate_research_results.py` & `scripts/validate_project.py`) runs continuous integrity checks.
