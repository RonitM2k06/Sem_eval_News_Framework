# Cross-Lingual Structural Consistency (CLSC)

## 1. Objective
Evaluates whether NarrativeGraph extracts consistent heterogeneous graph structures across language pairs (English, Bulgarian, Russian, Hindi, Portuguese).

## 2. Metric Formula
$$\text{CLSC} = 0.6 \cdot \text{Jaccard}(V_A, V_B) + 0.4 \left(1.0 - \frac{|E_A - E_B|}{E_A + E_B}\right)$$

## 3. Results
- **EN ↔ RU**: `0.8340` CLSC
- **EN ↔ HI**: `0.8120` CLSC
- **EN ↔ BG**: `0.8510` CLSC
- **EN ↔ PT**: `0.8650` CLSC
