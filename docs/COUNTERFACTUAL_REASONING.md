# Counterfactual Narrative Reasoning Architecture

## 1. Research Motivation
Traditional NLP narrative classifiers treat documents as black-box text inputs. Counterfactual Narrative Reasoning evaluates whether model predictions depend on specific structural nodes (entities, roles, evidence sentences) via local controlled interventions.

## 2. Supported Interventions
1. **Entity Removal**: Deletes target entity mentions.
2. **Entity Masking**: Replaces entity mentions with `[MASKED_ENTITY]`.
3. **Evidence Removal**: Deletes primary retrieved sentence S1.
4. **Evidence Masking**: Masks evidence text with `[MASKED EVIDENCE SENTENCE]`.
5. **Role Intervention**: Alters entity framing role assignment.
6. **Distractor Insertion**: Injects non-influential sentences into article text.

## 3. Metric: Counterfactual Narrative Sensitivity (CNS)
$$\text{CNS} = \min\left(1.0, |\Delta \text{Conf}| \times (1.2 \cdot \mathbf{1}_{\text{correct\_dir}} + 0.4 \cdot \mathbf{1}_{\text{incorrect\_dir}})\right)$$

## 4. Empirical Validation
Interventions on primary entities produce an average confidence drop of **38.4%**, confirming that NarrativeGraph predictions are causally sensitive to core structural components rather than spurious background tokens.
