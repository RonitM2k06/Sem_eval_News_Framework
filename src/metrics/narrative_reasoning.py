"""
NarrativeGraph Research Metrics Module.
Implements scientific metric calculations for structured narrative reasoning:
1. Counterfactual Narrative Sensitivity (CNS)
2. Evidence Necessity Score (ENS)
3. Evidence Sufficiency Score (ESS)
4. Evidence Minimality Score (EMS)
5. Narrative Conflict Score (NCS)
6. Cross-Lingual Structural Consistency (CLSC)
7. Perturbation Robustness Score (PRS)
8. Narrative Persistence Score (NPS)
9. Narrative Volatility Score (NVS)
"""

import math
from typing import Dict, List, Any, Optional


class NarrativeReasoningMetrics:
    """
    Centralized research metrics suite for NarrativeGraph structural reasoning.
    """

    @staticmethod
    def compute_cns(
        orig_confidence: float,
        cf_confidence: float,
        intervention_type: str,
        expected_direction: str = "decrease"
    ) -> Dict[str, Any]:
        """
        Counterfactual Narrative Sensitivity (CNS).
        Measures prediction sensitivity to structural interventions.
        """
        delta = orig_confidence - cf_confidence
        abs_delta = abs(delta)

        # Directional correctness check
        if expected_direction == "decrease":
            correct_direction = delta > 0.01
        elif expected_direction == "increase":
            correct_direction = delta < -0.01
        else:
            correct_direction = abs_delta < 0.05

        cns_score = min(1.0, max(0.0, abs_delta * (1.2 if correct_direction else 0.4)))

        return {
            "metric_name": "CNS",
            "cns_score": float(round(cns_score, 4)),
            "delta_confidence": float(round(delta, 4)),
            "directional_correctness": bool(correct_direction),
            "intervention_type": intervention_type,
            "interpretation": "High sensitivity indicates target prediction depends strongly on intervened structural node."
        }

    @staticmethod
    def compute_ens_ess_ems(
        full_confidence: float,
        no_evidence_confidence: float,
        evidence_only_confidence: float,
        curve_confidences: List[float]
    ) -> Dict[str, Any]:
        """
        Evidence Grounding Metrics:
        - ENS: Evidence Necessity Score
        - ESS: Evidence Sufficiency Score
        - EMS: Evidence Minimality Score
        """
        # Necessity: degradation when evidence is removed
        ens = max(0.0, min(1.0, (full_confidence - no_evidence_confidence) / max(1e-5, full_confidence)))

        # Sufficiency: preservation when only evidence is provided
        ess = max(0.0, min(1.0, evidence_only_confidence / max(1e-5, full_confidence)))

        # Minimality: smallest index achieving >= 90% of full confidence
        target_conf = 0.90 * full_confidence
        minimal_idx = len(curve_confidences)
        for idx, conf in enumerate(curve_confidences):
            if conf >= target_conf:
                minimal_idx = idx + 1
                break
        ems = max(0.0, min(1.0, 1.0 - (minimal_idx - 1) / max(1, len(curve_confidences))))

        return {
            "ens_score": float(round(ens, 4)),
            "ess_score": float(round(ess, 4)),
            "ems_score": float(round(ems, 4)),
            "minimal_sentence_count": int(minimal_idx),
            "preservation_curve": [float(round(c, 4)) for c in curve_confidences]
        }

    @staticmethod
    def compute_ncs(
        dominant_prob: float,
        competing_prob: float,
        evidence_overlap: float,
        role_incompatibility: float
    ) -> Dict[str, Any]:
        """
        Narrative Conflict Score (NCS).
        Quantifies competition between incompatible narrative frames.
        """
        # Entropy ratio of top-2 narratives
        prob_ratio = competing_prob / max(1e-5, dominant_prob)
        conflict_score = (0.5 * prob_ratio) + (0.3 * role_incompatibility) + (0.2 * (1.0 - evidence_overlap))
        ncs = max(0.0, min(1.0, conflict_score))

        return {
            "ncs_score": float(round(ncs, 4)),
            "competing_probability_ratio": float(round(prob_ratio, 4)),
            "role_incompatibility": float(round(role_incompatibility, 4)),
            "evidence_overlap": float(round(evidence_overlap, 4)),
            "has_conflict": bool(ncs >= 0.45)
        }

    @staticmethod
    def compute_clsc(
        graph_a: Dict[str, Any],
        graph_b: Dict[str, Any],
        lang_a: str,
        lang_b: str
    ) -> Dict[str, Any]:
        """
        Cross-Lingual Structural Consistency (CLSC).
        Measures structural similarity between graphs across languages.
        """
        nodes_a = set([n.get("label", n.get("id")) for n in graph_a.get("nodes", [])])
        nodes_b = set([n.get("label", n.get("id")) for n in graph_b.get("nodes", [])])

        if not nodes_a or not nodes_b:
            node_sim = 0.5
        else:
            intersection = len(nodes_a.intersection(nodes_b))
            union = len(nodes_a.union(nodes_b))
            node_sim = intersection / max(1, union)

        edges_a = len(graph_a.get("edges", []))
        edges_b = len(graph_b.get("edges", []))
        edge_sim = 1.0 - abs(edges_a - edges_b) / max(1, edges_a + edges_b)

        clsc_score = max(0.0, min(1.0, 0.6 * node_sim + 0.4 * edge_sim))

        return {
            "clsc_score": float(round(clsc_score, 4)),
            "node_jaccard_similarity": float(round(node_sim, 4)),
            "edge_density_similarity": float(round(edge_sim, 4)),
            "language_pair": f"{lang_a} -> {lang_b}"
        }

    @staticmethod
    def compute_prs(
        original_conf: float,
        perturbed_conf: float,
        perturbation_type: str,
        is_adversarial: bool = True
    ) -> Dict[str, Any]:
        """
        Perturbation Robustness Score (PRS).
        Evaluates stability under irrelevant/adversarial perturbations.
        """
        delta = abs(original_conf - perturbed_conf)
        if not is_adversarial:
            prs = max(0.0, min(1.0, 1.0 - delta))
        else:
            prs = max(0.0, min(1.0, delta * 1.5))

        return {
            "prs_score": float(round(prs, 4)),
            "delta_confidence": float(round(delta, 4)),
            "perturbation_type": perturbation_type,
            "is_adversarial": is_adversarial,
            "is_robust": bool(prs >= 0.70)
        }

    @staticmethod
    def compute_temporal_evolution(
        timesteps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Temporal Evolution Metrics:
        - NPS: Narrative Persistence Score
        - NVS: Narrative Volatility Score
        """
        if not timesteps:
            return {"nps_score": 1.0, "nvs_score": 0.0, "role_transitions": 0}

        narratives = [t.get("narrative") for t in timesteps]
        roles = [t.get("role") for t in timesteps]

        # Role transition count
        role_shifts = sum(1 for i in range(1, len(roles)) if roles[i] != roles[i - 1])
        narrative_shifts = sum(1 for i in range(1, len(narratives)) if narratives[i] != narratives[i - 1])

        total_steps = max(1, len(timesteps) - 1)
        volatility = (role_shifts + narrative_shifts) / (2.0 * total_steps)
        persistence = 1.0 - volatility

        return {
            "nps_score": float(round(persistence, 4)),
            "nvs_score": float(round(volatility, 4)),
            "role_transitions": int(role_shifts),
            "narrative_shifts": int(narrative_shifts),
            "total_timesteps": len(timesteps)
        }
