# NarrativeGraph Research Dashboard (Gold Development Set)

## Executive Performance Summary
- **Best Model**: NarrativeGraph (XLM-RoBERTa-base + GATv2 + L_alignment)
- **Overall Benchmark Macro F1**: **0.7680 ± 0.004** (+0.0730 over uncoupled multi-task transformer baseline B5)
- **Subtask 1 (Entity Framing) Macro F1**: **0.7320**
- **Subtask 2 (Narrative & Subnarrative) Macro F1**: **0.8040**
- **Subtask 3 (Explanation BERTScore)**: **0.8910**
- **Statistical Significance**: p < 0.0001 against baseline (Paired Bootstrap 10,000 resamples; 95% CI: [+0.0612, +0.0848])

---

## Component Contribution (Ablation Summary)
1. **Heterogeneous Graph Message Passing**: +0.0610 Macro F1 contribution (ablation: 0.7070).
2. **Trainable Alignment Loss (L_alignment)**: +0.0240 Macro F1 contribution (ablation: 0.7440).
3. **Entity Framing Signals**: +0.0370 Macro F1 contribution (ablation: 0.7310).
4. **Hierarchical Decoder**: +0.0300 Macro F1 contribution (ablation: 0.7380).
5. **Evidence Conditioning**: +0.0190 Macro F1 contribution (ablation: 0.7490).

---

## Evidence Faithfulness Metrics (Automated Evaluation)
- **Evidence Coverage Rate**: **81.2%**
- **Entailment Rate**: **89.5%**
- **Token Overlap with Source**: **74.3%**
- **Factual Precision**: **83.7%**
- **Hallucination Rate**: **4.8%** (Reduced from 12.3% without evidence RAG)
