# Methodology: Project NarrativeGraph

This document provides a comprehensive technical reference for the architecture, loss functions, graph construction, and alignment mechanisms of **NarrativeGraph**.

---

## 1. System Overview
NarrativeGraph processes multilingual news articles to simultaneously extract entity framing roles (Subtask 1), classify document-level hierarchical narratives and subnarratives (Subtask 2), and generate evidence-grounded textual explanations $\le 80$ words (Subtask 3).

---

## 2. Encoder Backbone
We employ `xlm-roberta-base` as the multilingual encoder backbone. Given an input token sequence $X = (x_1, \dots, x_T)$:
$$\mathbf{H} = \text{XLM-R}(X) \in \mathbb{R}^{T \times d_h}$$
$$\mathbf{h}_{\text{doc}} = \text{MeanPool}(\mathbf{H}) \in \mathbb{R}^{d_h}$$

---

## 3. Entity Framing Head (Subtask 1)
Given entity mention spans and local offsets, the entity framing classifier predicts probabilities across 11 main and fine-grained roles (Protagonist, Antagonist, Victim, Hero, etc.):
$$\hat{\mathbf{y}}_{\text{role}} = \sigma(\text{MLP}(\mathbf{h}_{\text{doc}}))$$

---

## 4. Heterogeneous Graph Construction & GATv2 Message Passing
We construct a heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ comprising:
- **Node Types**: Document ($D$), Sentence ($S_i$), Entity ($E_j$), Role ($R$), Narrative ($N$), Subnarrative ($SN$).
- **Edge Types**: Mentioned-In, Has-Role, Supports, Compatible-With.

Nodes are updated via Graph Attention Networks (GATv2):
$$\mathbf{h}_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W} \mathbf{h}_j^{(l)} \right)$$

---

## 5. Structured Alignment Loss ($L_{\text{alignment}}$)
To enforce cross-level consistency between structural components, we optimize:
$$\mathcal{L}_{\text{alignment}} = \mathcal{L}_{E \leftrightarrow R} + \mathcal{L}_{R \leftrightarrow EV} + \mathcal{L}_{EV \leftrightarrow N} + \mathcal{L}_{N \leftrightarrow SN}$$
where bilinear compatibility matrices evaluate pair matching energy.
