"""
Error Taxonomy Analyzer.
Categorizes predictions into Entity Errors, Narrative Errors, Evidence Errors, and Explanation Errors.
"""

from typing import List, Dict, Any


class ErrorTaxonomyAnalyzer:
    """
    Categorizes and tabulates model failure modes across all 3 subtasks.
    """

    def analyze_errors(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        error_counts = {
            "entity_role_ambiguity": 0,
            "narrative_parent_confusion": 0,
            "subnarrative_misclassification": 0,
            "evidence_selection_failure": 0,
            "explanation_hallucination": 0
        }

        total = len(predictions)
        for pred, gt in zip(predictions, ground_truths):
            # Check entity error
            if pred.get("roles") != gt.get("roles"):
                error_counts["entity_role_ambiguity"] += 1

            # Check narrative parent error
            if pred.get("narrative") != gt.get("narrative"):
                error_counts["narrative_parent_confusion"] += 1

            # Check subnarrative error
            if pred.get("subnarrative") != gt.get("subnarrative"):
                error_counts["subnarrative_misclassification"] += 1

            # Check evidence error
            if pred.get("evidence_ids") != gt.get("evidence_sentence_ids"):
                error_counts["evidence_selection_failure"] += 1

            # Check explanation hallucination
            if len(pred.get("explanation", "").split()) > 80:
                error_counts["explanation_hallucination"] += 1

        error_proportions = {
            k: float(round(v / max(1, total), 4))
            for k, v in error_counts.items()
        }

        return {
            "total_samples": total,
            "error_counts": error_counts,
            "error_proportions": error_proportions
        }
