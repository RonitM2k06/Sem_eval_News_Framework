# Hostile ACL / EMNLP Peer Review (Meta-Review & Official Reviews)

## Meta-Review (Area Chair)
**Recommendation**: Accept (Poster / Findings).  
**Summary**: The paper presents NarrativeGraph, a heterogeneous Graph Attention Network framework that jointly models entity framing, textual evidence, and hierarchical narratives for the SemEval 2025 Task 10 benchmark. The reviewers appreciate the thorough ablation studies, multi-seed reporting, and counterfactual sensitivity experiments. However, reviewers raise important concerns regarding whether the conflict detection and temporal components are truly supervised vs heuristic, and whether the graph overhead is justified relative to strong LLM prompting. The authors' empirical ablations and statistical tests address the core technical questions well, provided that claims are toned down from "causal" to "interventional sensitivity" and temporal evaluations are honestly framed.

---

## Official Reviewer 1 (Expert in Graph NLP & Multilingual Modeling)
**Score**: 3.5 / 5.0 (Borderline Accept)  
**Confidence**: 4 / 5 (High)

### Strengths:
1. Solid motivation: Coupling entity framing roles with document-level narrative prediction makes linguistic sense.
2. Strong empirical results: Outperforming uncoupled multi-task transformers by +0.073 F1 on SemEval 2025 Task 10 is non-trivial.
3. Excellent ablation study: Demonstrates the necessity of both GNN message passing (-0.060 F1) and alignment loss (-0.052 F1).

### Weaknesses:
1. *Novelty of GNN Architecture*: The graph attention mechanism relies on standard GATv2 layers (Brody et al., 2022). The novelty lies in the schema formulation and alignment loss rather than a new graph neural operator.
2. *Language Family Bias*: Gains are strongest in high-resource English and Russian, but lower in Hindi and Bulgarian.

### Questions for Authors:
- Did you explore cross-lingual link prediction directly on the graph across languages?

---

## Official Reviewer 2 (Expert in Faithfulness & Explainability)
**Score**: 4.0 / 5.0 (Accept)  
**Confidence**: 5 / 5 (Expert)

### Strengths:
1. Excellent explanation evaluation: Assessing faithfulness through both necessity (ENS) and sufficiency (ESS) provides much more insight than simple ROUGE or BLEU scores.
2. Reducing hallucination rate from 18.2% to 4.8% is a major practical improvement.
3. 3-seed evaluation with paired bootstrap significance ($p < 0.001$) shows rigorous experimental standards.

### Weaknesses:
1. *Causal Language*: The paper initially uses terms like "causal inference" in Section 1. True causal inference requires unverifiable DAG assumptions; "counterfactual sensitivity" or "interventional ablation" is more accurate.
2. *Human Evaluation*: Explanation quality is evaluated using BERTScore and NLI models rather than human domain experts. The authors must disclose this limitation.

---

## Official Reviewer 3 (Expert in Information Extraction & Shared Tasks)
**Score**: 3.5 / 5.0 (Borderline Accept)  
**Confidence**: 4 / 5 (High)

### Strengths:
1. Directly targets a real shared task (SemEval 2025 Task 10) covering 5 languages and 2 complex domains.
2. Latency and parameter efficiency table is very welcome; showing only +7M parameters over baseline demonstrates practical deployability.

### Weaknesses:
1. *Temporal Dataset Metadata*: The paper mentions temporal narrative evolution, but SemEval 2025 Task 10 articles do not provide verified multi-year timestamps. The authors must make sure synthetic timelines are not conflated with real benchmark data.
2. *Baseline Tuning*: Were baselines B0--B5 tuned with equal computational budget?
