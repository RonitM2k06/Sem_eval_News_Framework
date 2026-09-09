# Evidence Necessity, Sufficiency, and Minimality

## 1. Metric Formulations
- **Evidence Necessity Score (ENS)**:
  $$\text{ENS} = \frac{\text{Conf}(\text{Full}) - \text{Conf}(\text{NoEv})}{\text{Conf}(\text{Full})}$$
- **Evidence Sufficiency Score (ESS)**:
  $$\text{ESS} = \frac{\text{Conf}(\text{EvOnly})}{\text{Conf}(\text{Full})}$$
- **Evidence Minimality Score (EMS)**:
  $$\text{EMS} = 1.0 - \frac{k_{\min} - 1}{K_{\text{total}}}$$

## 2. Experimental Results
- Average **ENS**: `0.7810` (High necessity)
- Average **ESS**: `0.8650` (High sufficiency)
- **Minimality Threshold**: Top-2 retrieved sentences preserve $\ge 90.0\%$ of full-document narrative prediction.
