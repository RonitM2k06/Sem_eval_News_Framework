# Narrative Conflict Detection Framework

## 1. Objective
Real-world news articles frequently present competing perspectives. NarrativeGraph models conflict without forcing articles into single homogeneous frames.

## 2. Narrative Conflict Score (NCS)
$$\text{NCS} = 0.5 \left(\frac{P_{\text{competing}}}{P_{\text{dominant}}}\right) + 0.3 \, \text{RoleIncomp} + 0.2 (1.0 - \text{EvOverlap})$$

## 3. Conflict Categories
1. **Entity-Role Conflict**: Entities assigned opposing roles across sentences.
2. **Evidence-Narrative Conflict**: Retrieved sentences support divergent parent trees.
3. **Narrative Competition**: High probability mass shared between top-2 taxonomy nodes.
