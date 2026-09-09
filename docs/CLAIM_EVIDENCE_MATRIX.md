# Claim–Evidence Matrix: Project NarrativeGraph

This document tracks all primary research claims, mapping each claim to its supporting experiment, target metric, empirical evidence, and confidence assessment.

---

| Paper Claim | Target Research Question | Supporting Experiment | Metric | Empirical Evidence / Status | Confidence |
| ----------- | ----------------------- | --------------------- | ------ | --------------------------- | ---------- |
| **Claim 1**: Explicit modeling of Entity–Role–Evidence–Narrative interactions improves document-level narrative and subnarrative classification over uncoupled multi-task baselines. | **RQ1** & **RQ4** | Exp 1 (Main Benchmark) & Exp 2 (Ablation) | Macro F1 (Subtask 2), Accuracy | Pending Experiment Run | High (Hypothesized) |
| **Claim 2**: Entity framing representations provide informative signals for document-level narrative prediction. | **RQ2** | Exp 2 (Ablation: No Entity Info) | Macro F1, Δ F1 | Pending Experiment Run | High (Hypothesized) |
| **Claim 3**: Narrative-conditioned sentence retrieval improves narrative prediction and evidence grounding. | **RQ3** & **RQ9** | Exp 8 (Distractor Robustness) & Exp 9 (Evidence Grounding) | Top-$k$ Precision, BERTScore | Pending Experiment Run | High (Hypothesized) |
| **Claim 4**: Evidence-conditioned explanation generation produces significantly lower hallucination rates and higher entailment. | **RQ5** | Exp 9 (Evidence Grounding) | Entailment Rate, Unsupported Claim Rate, BERTScore | Pending Experiment Run | Very High (Hypothesized) |
| **Claim 5**: Joint multilingual training with NarrativeGraph improves low-resource language performance. | **RQ6** | Exp 4 (Cross-Lingual Transfer & LOLO) | Macro F1 (bg, hi, pt, ru) | Pending Experiment Run | High (Hypothesized) |
| **Claim 6**: NarrativeGraph's structural reasoning generalizes across distinct domains (Ukraine-Russia War $\leftrightarrow$ Climate Change). | **RQ7** | Exp 5 (Cross-Domain Generalization) | Out-of-Domain Macro F1 drop (Δ) | Pending Experiment Run | Moderate (Hypothesized) |
| **Claim 7**: NarrativeGraph degrades more gracefully than standard transformers when training data is severely restricted ($5\% - 25\%$). | **RQ8** | Exp 6 (Low-Resource Data Efficiency) | F1 vs Training Data % Learning Curves | Pending Experiment Run | High (Hypothesized) |
