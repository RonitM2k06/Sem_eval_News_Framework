# Phase 3 Scientific Verdict: Project NarrativeGraph

## Research Hypothesis
**Hypothesis**: Explicit Entity–Role–Evidence–Narrative alignment improves multilingual narrative understanding compared with independent or loosely coupled models.

**Verdict**: **PENDING REAL DATA / INCONCLUSIVE**

---

## Evidence
The experimental validation pipeline successfully executed tests for baseline models (Independent tasks), NarrativeGraph, and critical ablations (No GNN, No Alignment Loss) across 3 distinct random seeds. 

However, because the official SemEval 2025 Task 10 dataset is not currently present in the execution environment, all empirical evaluation was performed over the `data/synthetic/` smoke-test split. 
As expected, models operating on randomly generated synthetic text and labels achieved near-random Macro F1 scores (~0.0245) and displayed no statistically significant variance across ablations. 

We can confirm the *code* works, the *loss* computes gradients, and the *graph* updates node embeddings—but we cannot confirm the *hypothesis*.

## Strongest Contribution
**Automated Research Infrastructure**: We proved that the pipeline can train, ablate, and evaluate heterogeneous graph models end-to-end without manual intervention, while strictly tracking provenance across configurations and seeds.

## Weakest Contribution
**Empirical Validation**: Until real data is provided, the mathematical soundness of $L_{\text{alignment}}$ on true linguistic distributions remains unmeasured.

## Most Important Limitation
**Lack of Real Benchmark Data**: The current findings are purely a reflection of the software architecture's stability, not the method's NLP capabilities. The synthetic dataset does not contain actual semantic structures for the GNN to learn.

## Publication Readiness
**NOT YET ASSESSED**
The platform and dashboard are technically ready for a publication artifact, but the research claims cannot be submitted until the real SemEval dataset is acquired and processed through the pipeline.
