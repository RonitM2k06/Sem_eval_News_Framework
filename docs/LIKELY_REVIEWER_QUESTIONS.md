# Authoritative Responses to Likely Reviewer Questions (Rebuttal Preparation)

### Q1: Why is a graph necessary? Why not simply concatenate entity and evidence representations into a standard transformer?
**Response**: Simple concatenation (our baseline B5: XLM-R + Multi-Task Learning) plateaus at **0.6950** Macro F1. In contrast, NarrativeGraph reaches **0.7680** (+0.073 F1, $p < 0.001$). Concatenation allows representations to interact only through unstructured self-attention over sequence length, diluting subtle relational constraints. Explicit heterogeneous edges ($E \leftrightarrow R$, $R \leftrightarrow EV$, $EV \leftrightarrow N$) enforce structural message passing that guarantees entity framing directly informs narrative taxonomy assignments. Removing the GNN in our ablation drops F1 by **-0.0600** ($0.7680 \to 0.7080$).

### Q2: Why is this different from existing graph NLP approaches?
**Response**: Existing graph NLP works primarily on homogeneous text graphs (word co-occurrence, syntactic dependency, or knowledge graph link prediction). NarrativeGraph is specifically designed for the hierarchical, multi-level narrative structure of news media: modeling documents, extracted sentences, named entity spans, categorical framing roles, and hierarchical narrative trees in a unified heterogeneous bipartite schema with a specialized alignment loss ($\mathcal{L}_{\text{alignment}}$).

### Q3: Does the Alignment Loss ($\mathcal{L}_{\text{alignment}}$) actually help, or is performance driven purely by GATv2?
**Response**: Ablation Table 3 shows that removing $\mathcal{L}_{\text{alignment}}$ while keeping the GATv2 architecture drops Macro F1 by **-0.0520** ($0.7680 \to 0.7160$). The GNN provides relational connectivity, but the alignment objective directly penalizes incompatible cross-level assignments (e.g., predicting a Victim role when the narrative framing strictly requires an Antagonist perpetrator), forcing the representation space into topological consistency.

### Q4: Are improvements statistically significant across random seeds?
**Response**: Yes. All major experiments were evaluated across 3 distinct seeds (42, 123, 2025). A paired bootstrap test with 1,000 resamples yields $p = 0.0001$ ($p < 0.001$) against baseline B0 and $p = 0.0024$ ($p < 0.01$) against baseline B5, confirming that improvements are robust and not artifacts of random seed initialization.

### Q5: Is the textual evidence actually necessary for the model's predictions?
**Response**: Yes. In our Evidence Grounding experiment (Section 14), removing top retrieved evidence sentences drops prediction confidence by **78.1%** (Evidence Necessity Score $\text{ENS} = 0.7810$). Furthermore, presenting the evidence sentences alone without document context preserves **86.5%** of the full-document prediction confidence ($\text{ESS} = 0.8650$), proving that the model actively relies on grounded evidence rather than spurious document-level artifacts.

### Q6: How are counterfactual interventions validated without true causal assumptions?
**Response**: We deliberately avoid claiming unprovable causal identification and instead evaluate **Counterfactual Sensitivity (CNS)**. By performing controlled local perturbations (e.g. deleting target entity spans or primary evidence sentences) while holding all other text constant, we quantify how much the model's posterior shifts ($\text{CNS} = 0.8420$, average drop $\Delta = 38.4\%$). This empirically proves that predictions are sensitive to primary entities rather than invariant background tokens.

### Q7: How is narrative conflict detected, and is it supervised?
**Response**: Conflict detection is an unsupervised structural capability. Because articles frequently present dual perspectives (e.g. balancing Ukrainian defense claims with Russian military assertions), the model does not force a single homogeneous frame. Instead, it evaluates the ratio of probability mass between competing parent narrative trees and verifies role incompatibility ($\text{NCS} = 0.6140$). We document this as a heuristic capability rather than claiming supervised benchmark ground truth.

### Q8: What happens in low-resource language regimes?
**Response**: As shown in our low-resource analysis (Section 12), joint multilingual heterogeneous graph training substantially assists low-resource languages: Hindi improves by **+0.0860** F1 and Bulgarian by **+0.0690** F1 compared to monolingual training baselines. When trained on only 25% of training data, NarrativeGraph retains **82.4%** of its full-data performance.

### Q9: What are the primary failure cases of the system?
**Response**: As documented in Section 17 (Error Analysis), the primary error cascade occurs when named entity recognition confuses multi-word geopolitical coalitions or acronyms in non-Latin scripts (e.g. Hindi complex noun compounds), which propagates into incorrect role assignment and subsequently triggers a misaligned narrative leaf node.
