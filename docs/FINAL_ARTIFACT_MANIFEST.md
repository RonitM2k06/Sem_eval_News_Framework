# NarrativeGraph Final Artifact Manifest

This manifest catalogs all verified digital assets, source modules, experiments, documentation, paper assets, and reproduction tools comprising Project NarrativeGraph.

---

## 1. Core Source Code Modules
- `src/models/narrative_graph_model.py`: Heterogeneous GATv2 PyTorch model implementation.
- `src/models/alignment_loss.py`: Trainable Entity--Role--Evidence--Narrative Alignment loss objective.
- `src/reasoning/engine.py`: Unified Structured Narrative Reasoning Engine.
- `src/reasoning/graph_diff.py`: Structural graph comparison and modification scoring engine.
- `src/metrics/narrative_reasoning.py`: Scientific implementations of CNS, ENS, ESS, EMS, NCS, CLSC, PRS, NPS, NVS.
- `src/taxonomy/parser.py`: Authoritative two-level parent/subnarrative taxonomy parser.
- `src/evaluation/faithfulness.py`: NLI entailment, BERTScore, and hallucination reduction evaluator.
- `src/analysis/significance.py`: Paired bootstrap hypothesis testing module (1,000 resamples).
- `src/utils/reproducibility.py`: Deterministic seeding protocol (seeds 42, 123, 2025).

---

## 2. Web API & Dashboard Assets
- `backend/app.py`: FastAPI server application entrypoint.
- `backend/services/inference.py`: Production PyTorch inference pipeline service.
- `backend/router/reasoning.py`: Research API endpoints for all 6 Structured Reasoning Labs.
- `frontend/index.html`: Complete research dashboard with 15-column Baseline Ladder and 6 Reasoning Labs.
- `frontend/app.js`: Interactive client controller with dynamic multi-metric switching and Cytoscape graph renderer.
- `frontend/styles.css`: Visual presentation system with dark/light themes.

---

## 3. Experimentation Suites (`experiments/`)
- `experiments/registry.csv`: Machine-readable provenance registry tracking all experiment IDs and seeds.
- `experiments/counterfactual/cf_001_interventions.py`: Counterfactual intervention test suite.
- `experiments/evidence_grounding/ev_001_grounding.py`: Evidence necessity, sufficiency, and minimality suite.
- `experiments/conflict/cnf_001_conflict.py`: Competing narrative frame and role conflict suite.
- `experiments/cross_lingual/cl_001_consistency.py`: Multilingual graph structural alignment suite.
- `experiments/adversarial/adv_001_robustness.py`: 10-type adversarial perturbation suite.
- `experiments/temporal/tmp_001_evolution.py`: Longitudinal narrative persistence and volatility suite.

---

## 4. Verification & Validation Scripts (`scripts/`)
- `scripts/validate_project.py`: Master 7-point validation suite (returns Exit Code 0).
- `scripts/validate_research_results.py`: Provenance and data leakage audit script.
- `scripts/generate_paper_results.py`: End-to-end multi-seed baseline and ablation runner.
- `scripts/generate_tables.py`: Automated generator for CSV and publication LaTeX tables.
- `scripts/generate_figures.py`: Publication figure rendering script.
- `scripts/run_dashboard.py`: Lightweight background dashboard daemon launcher.

---

## 5. Publication Paper Assets (`paper/`)
- `paper/main.tex`: 22-section ACL proceedings formatted LaTeX paper.
- `paper/references.bib`: BibTeX bibliography covering all cited works.
- `paper/tables/main_results.tex`: Main benchmark multi-metric table.
- `paper/tables/ablations.tex`: Component contribution ablation table.
- `paper/tables/cross_lingual.tex`: Cross-lingual transfer evaluation table.
- `paper/figures/main_performance.png`: Benchmark F1 ladder plot.
- `paper/figures/faithfulness_comparison.png`: Explanation hallucination reduction plot.
- `paper/figures/low_resource_curves.png`: Low-resource data regime retention curves.

---

## 6. Authoritative Research Documentation (`docs/`)
- `docs/FINAL_PUBLICATION_AUDIT.md`: Multidisciplinary evaluation matrix.
- `docs/FINAL_LEAKAGE_AUDIT.md`: 12-point data contamination verification.
- `docs/FINAL_CLAIM_EVIDENCE_MATRIX.md`: End-to-end claim-to-artifact mapping.
- `docs/FINAL_NOVELTY_POSITION.md`: Conservative novelty classification.
- `docs/FINAL_REVIEW.md`: Hostile ACL/EMNLP peer review.
- `docs/LIKELY_REVIEWER_QUESTIONS.md`: Comprehensive rebuttal Q&A.
- `docs/REPRODUCIBILITY.md`: Deterministic environment and hardware runbook.
- `docs/RESEARCH_STATUS.md`: Authoritative component status catalog.
- `docs/FINAL_PUBLICATION_SCORECARD.md`: Evidence-based research scorecard.
- `docs/FINAL_STATUS.md`: Executive research platform summary.
- `docs/COUNTERFACTUAL_REASONING.md`: Counterfactual sensitivity architecture.
- `docs/EVIDENCE_GROUNDING.md`: Necessity and sufficiency formulations.
- `docs/NARRATIVE_CONFLICT.md`: Competing narrative detection framework.
- `docs/CROSS_LINGUAL_CONSISTENCY.md`: Multilingual graph consistency framework.
- `docs/ADVERSARIAL_ROBUSTNESS.md`: 10-type perturbation framework.
- `docs/TEMPORAL_EVOLUTION.md`: Longitudinal trajectory tracking framework.

---

## 7. Automated Test Suite (`tests/`)
- `tests/unit/test_counterfactual.py`: Counterfactual intervention unit tests.
- `tests/unit/test_evidence_grounding.py`: Evidence necessity and sufficiency tests.
- `tests/unit/test_conflict.py`: Competing narrative conflict tests.
- `tests/unit/test_cross_lingual.py`: Cross-lingual graph comparison tests.
- `tests/unit/test_adversarial.py`: Adversarial perturbation tests.
- `tests/unit/test_temporal.py`: Temporal trajectory tracking tests.
- `tests/unit/test_models.py`: PyTorch GATv2 forward pass and alignment loss tests.
- `tests/unit/test_taxonomy.py`: Taxonomy parser and hierarchy traversal tests.
