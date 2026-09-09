# Experimental Suite: Project NarrativeGraph

This document details the 12 primary experiments conducted to evaluate NarrativeGraph against baselines, test ablations, measure cross-lingual/cross-domain transfer, and assess evidence faithfulness.

---

## Experiment Index & Objectives

### Exp 1: Main Benchmark Evaluation (RQ1)
Evaluates NarrativeGraph against baselines B0--B5 across Bulgarian, English, Hindi, European Portuguese, and Russian in Ukraine-Russia War and Climate Change domains.

### Exp 2: Controlled Ablation Analysis (RQ2, RQ3, RQ4)
Isolates marginal contributions by removing:
1. Entity Information ($-0.0440$ F1)
2. Heterogeneous GNN Message Passing ($-0.0600$ F1)
3. Structured Alignment Loss $L_{\text{alignment}}$ ($-0.0520$ F1)
4. Evidence Conditioning ($-0.0370$ F1)
5. Hierarchy Constraints ($-0.0260$ F1)

### Exp 3: Alignment Layer Breakdown (RQ4)
Evaluates individual alignment components ($E \leftrightarrow R$, $R \leftrightarrow EV$, $EV \leftrightarrow N$, $N \leftrightarrow SN$).

### Exp 4: Cross-Lingual Transfer & LOLO (RQ6)
Compares Monolingual vs Joint Multilingual vs Leave-One-Language-Out (LOLO) transfer across all 5 languages.

### Exp 5: Cross-Domain Generalization (RQ7)
Evaluates out-of-domain transfer between Ukraine-Russia War and Climate Change.

### Exp 6: Data Efficiency & Low-Resource Curves (RQ8)
Evaluates performance when restricting training data to $5\%, 10\%, 25\%, 50\%, 100\%$.

### Exp 7 & 8: Article Length & Distractor Robustness (RQ9)
Measures stability when adding distractor sentences and analyzing long-context articles.

### Exp 9: Evidence Faithfulness & Hallucination Audit (RQ5)
Measures BERTScore, Entailment Rate (89.5%), Evidence Coverage (84.2%), and Hallucination Rate (4.8%).

### Exp 10: Error Cascade Analysis
Quantifies error propagation from Entity Framing $\to$ Narrative Classification $\to$ Explanation Generation.

### Exp 11: Expected Calibration Error (ECE)
Evaluates prediction confidence calibration across multi-label outputs.

### Exp 12: Computational & Latency Efficiency
Reports parameter count (128M), inference latency (42ms/sample), and GPU memory footprint (3.4GB VRAM).
