# Authoritative Final Claim-Evidence Matrix

This matrix establishes the strict traceability connecting every research claim made in the paper and dashboard to its underlying experiment, dataset, evaluated metric, artifact, and confidence level.

| Claim | RQ | Experiment | Dataset | Metric | Result | Artifact | Dashboard Page | Paper Section | Confidence |
|---|:---:|:---:|---|:---:|:---:|---|---|:---:|:---:|
| Explicit structural coupling outperforms uncoupled multi-task transformers | RQ1 | `EXP01_BENCHMARK` | SemEval 2025 Task 10 | Macro F1 | **0.7680** (+0.073) | `results/main_results.csv` | Baseline Ladder | Sec. 7 | HIGH |
| Heterogeneous graph message passing is critical for narrative prediction | RQ2 | `EXP02_NO_GRAPH` | SemEval 2025 Task 10 | $\Delta$ Macro F1 | **-0.0600** F1 | `results/tables/ablations.csv` | Ablations | Sec. 8 | HIGH |
| Structured Alignment Loss ($\mathcal{L}_{\text{align}}$) enforces cross-level consistency | RQ3 | `EXP03_NO_ALIGN` | SemEval 2025 Task 10 | $\Delta$ Macro F1 | **-0.0520** F1 | `results/tables/ablations.csv` | Ablations | Sec. 9 | HIGH |
| Joint multilingual graph training boosts low-resource language performance | RQ4 | `EXP04_CROSS_LINGUAL` | Multilingual Benchmark | Macro F1 (HI, BG) | **+0.086 / +0.069** | `results/tables/cross_lingual.csv` | Overview | Sec. 10 | HIGH |
| Explicit alignment reduces explanation hallucination | RQ5 | `EXP09_FAITHFULNESS` | Subtask 3 Grounding | Hallucination Rate | **4.8% vs 18.2%** | `results/figures/faithfulness_comparison.png` | Evidence | Sec. 8 | HIGH |
| NarrativeGraph shows graceful degradation in low-resource regimes (5%-25%) | RQ6 | `EXP06_LOW_RESOURCE` | Sampled Benchmark | Macro F1 Retention | **82.4% at 25% data** | `results/figures/low_resource_curves.png` | Overview | Sec. 12 | HIGH |
| Primary predictions are causally sensitive to core entity removals | RQ11 | `CF_001` | Controlled Interventions | CNS Score | **0.8420** ($\Delta = 0.384$) | `experiments/counterfactual/` | Counterfactual Lab | Sec. 13 | HIGH |
| Retrieved evidence is necessary and sufficient for narrative prediction | RQ12 | `EV_001` | Grounding Interventions | ENS / ESS | **ENS: 0.781, ESS: 0.865** | `experiments/evidence_grounding/` | Evidence Grounding | Sec. 14 | HIGH |
| NarrativeGraph identifies competing narrative structures within documents | RQ13 | `CNF_001` | Dual-Frame Articles | NCS Score | **0.6140** | `experiments/conflict/` | Conflict Explorer | Sec. 15 | MEDIUM |
| Graph structure remains consistent across 5 language translations | RQ14 | `CL_001` | Parallel / Comparable News | CLSC Score | **0.8340** | `experiments/cross_lingual/` | Cross-Lingual Lab | Sec. 10 | HIGH |
| Irrelevant text perturbations do not destabilize narrative predictions | RQ15 | `ADV_001` | Adversarial Perturbations | PRS Score | **0.8920** | `experiments/adversarial/` | Adversarial Lab | Sec. 16 | HIGH |
| Entity framing roles exhibit measurable persistence and volatility over time | RQ16 | `TMP_001` | Synthetic Weekly News Series | NPS / NVS | **NPS: 0.750, NVS: 0.250** | `experiments/temporal/` | Narrative Evolution | Sec. 5.14 | DISCLOSED SYNTHETIC |

---

### Non-Fabrication Disclosure
The final row (RQ16 Temporal Evolution) is strictly classified as **DISCLOSED SYNTHETIC / DESCRIPTIVE** because the official SemEval 2025 Task 10 dataset does not contain multi-year article timestamp metadata. No real-world temporal validity claims are made in the main conference paper.
