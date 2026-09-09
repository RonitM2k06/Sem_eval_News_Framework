# Novelty Audit: Project NarrativeGraph

## Executive Summary
This document provides a systematic review of existing work in multilingual narrative understanding, entity framing, evidence retrieval, and structured prediction, explicitly establishing the novelty of **NarrativeGraph**.

---

## 1. Literature Matrix & Overlap Audit

| Existing Method | Paper / Source | Year | What It Does | Overlap With NarrativeGraph | Our Difference | Novelty Assessment |
| --------------- | -------------- | ---: | ------------ | --------------------------- | -------------- | ------------------ |
| **SemEval-2025 Task 10 System Papers** | Piskorski et al. (SemEval 2025) | 2025 | Standard fine-tuning of XLM-R / Llama on Subtasks 1, 2, 3 independently or sequentially. | Uses transformer backbones and standard cross-entropy loss. | NarrativeGraph explicitly models inter-subtask structural dependencies via a heterogeneous graph and trainable alignment loss. | **Genuinely Novel** (First joint entity-role-evidence alignment graph for narrative extraction) |
| **Hierarchical Narrative Classifier** | Chen et al. (ACL) | 2024 | Multi-label hierarchical document classification with parent-child tree loss. | Taxonomy-aware hierarchical loss formulation. | Integrates entity framing and sentence-level evidence into the hierarchy via GNN message passing. | **Potentially Novel** (Novel integration of entity roles into hierarchy) |
| **Target-Oriented Entity Framing** | Nakov et al. (EMNLP) | 2023 | Predicts entity roles (Protagonist / Antagonist / Victim) from news spans. | Entity offset extraction and span-level classification. | Feeds entity-role representations directly into document-level narrative prediction and evidence grounding. | **Moderately Novel** (Entity role representation fed upstream to narrative prediction) |
| **Sentence-Level RAG for Explanations** | Lewis et al. (NeurIPS) | 2020 | Retrieves top-$k$ sentences to condition generator for summarization. | Conditioned generation for explanation subtask (Subtask 3). | NarrativeGraph conditions retrieval on *both* predicted narrative and entity roles, explicitly filtering non-role evidence. | **Moderately Novel** (Narrative- and Role-conditioned evidence selection) |
| **Heterogeneous Graph Reasoning for NLP** | Lin et al. (NAACL) | 2022 | Builds entity-sentence-document graphs for multi-document summarization. | Node-edge graph layout with Graph Attention Networks (GAT). | Extends graph to include explicit *Entity Role*, *Narrative*, and *Subnarrative* nodes with custom alignment loss. | **Genuinely Novel** (Domain-specific structured graph for narrative reasoning) |

---

## 2. Novelty Gate Verification

### Question A: Does the proposed contribution already exist almost exactly?
**Answer**: NO. Standard participant systems for SemEval 2025 Task 10 either fine-tune XLM-RoBERTa on isolated subtasks or prompt LLMs (e.g., Llama-3, Flan-T5) with standard RAG. No existing work unifies Entity Framing, Narrative Classification, Evidence Retrieval, and Explanation Generation into a single trainable heterogeneous alignment graph.

### Question B: Does existing work combine some of the components?
**Answer**: YES. Multi-task transformers and RAG pipeline approaches exist. However, existing multi-task approaches simply concatenate representations or share backbones without enforcing explicit compatibility constraints ($L_{\text{alignment}}$) between entity roles and document narratives.

### Question C: Is our proposed graph merely a visualization of existing relationships?
**Answer**: NO. NarrativeGraph's heterogeneous graph is a **trainable message-passing mechanism** utilizing Graph Attention Networks (GATv2) and contrastive energy-based compatibility losses. Node embeddings for entities, sentences, and narratives mutually refine each other during gradient updates.

### Question D: Does the component produce a measurable scientific hypothesis?
**Answer**: YES. Hypothesis: *Explicit structural alignment between entity roles and supporting sentence evidence reduces hallucination in explanation generation and improves subnarrative classification accuracy in cross-lingual/cross-domain settings.*

### Question E: Can an ablation actually test the component?
**Answer**: YES. Controlled ablations (Removing Graph, Removing Alignment Loss, Removing Evidence Conditioning, Removing Entity Roles) directly isolate the marginal gain of each component.

---

## 3. Explicit Novelty Claim

We claim two core scientific contributions:
1. **The NarrativeGraph Architecture**: A heterogeneous graph neural network that unifies Document, Sentence, Entity, Role, Narrative, and Subnarrative nodes into a joint reasoning framework.
2. **Trainable Entity–Role–Evidence–Narrative Alignment Loss ($\mathcal{L}_{\text{alignment}}$)**: A structured loss formulation enforcing cross-level consistency across framing, evidence retrieval, and narrative prediction.
