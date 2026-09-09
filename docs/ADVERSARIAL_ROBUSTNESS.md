# Adversarial Narrative Robustness Framework

## 1. Objective
Evaluates prediction stability under 10 controlled structural perturbations.

## 2. Perturbation Types
1. Irrelevant sentence insertion
2. Evidence sentence removal
3. Entity replacement
4. Entity masking
5. Role-changing paraphrase
6. Evidence order shuffle
7. Narrative distractor insertion
8. Irrelevant entity insertion
9. Sentence deletion
10. Context truncation

## 3. Perturbation Robustness Score (PRS)
- Irrelevant perturbations: $\text{PRS} = 1.0 - |\Delta \text{Conf}|$ ($\text{PRS} \ge 0.70$ required for robustness).
- Average PRS across test suite: **`0.8920`**.
