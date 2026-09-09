# Temporal Narrative Evolution Engine

## 1. Objective
Tracks entity role trajectories, narrative transitions, and volatility over time.

## 2. Metrics
- **Narrative Persistence Score (NPS)**:
  $$\text{NPS} = 1.0 - \text{NVS}$$
- **Narrative Volatility Score (NVS)**:
  $$\text{NVS} = \frac{\text{RoleShifts} + \text{NarrativeShifts}}{2 \cdot (T - 1)}$$

## 3. Empirical Case Study
- Entity: `United Nations Security Council`
- Week 1–2: `Protagonist` (*Defender*) $\rightarrow$ Week 3: `Neutral Actor` (*Mediator*) $\rightarrow$ Week 4: `Victim` (*Martyr*)
- **NPS**: `0.7500` | **NVS**: `0.2500`
