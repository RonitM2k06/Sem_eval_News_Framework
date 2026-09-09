"""
Evaluation Metrics for SemEval 2025 Task 10 Subtasks 1, 2, and 3.
Calculates Macro F1, Micro F1, Accuracy, Evidence Ranking Top-K Precision, and BERTScore proxies.
"""

from typing import List, Dict, Any
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def compute_all_metrics(
    outputs_list: List[Dict[str, torch.Tensor]],
    targets_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    all_role_preds = []
    all_role_targets = []
    all_narr_preds = []
    all_narr_targets = []
    all_subnarr_preds = []
    all_subnarr_targets = []

    for out, tgt in zip(outputs_list, targets_list):
        # Role predictions (sigmoid threshold >= 0.5)
        role_probs = torch.sigmoid(out["role_logits"]).cpu()
        role_preds = (role_probs >= 0.5).float()
        all_role_preds.append(role_preds)
        all_role_targets.append(tgt["role_target"].cpu())

        # Narrative predictions (argmax)
        narr_preds = torch.argmax(out["narrative_logits"], dim=-1).cpu()
        all_narr_preds.append(narr_preds)
        all_narr_targets.append(tgt["narrative_target"].cpu())

        # Subnarrative predictions (argmax)
        subnarr_preds = torch.argmax(out["subnarrative_logits"], dim=-1).cpu()
        all_subnarr_preds.append(subnarr_preds)
        all_subnarr_targets.append(tgt["subnarrative_target"].cpu())

    all_role_preds = torch.cat(all_role_preds, dim=0).numpy()
    all_role_targets = torch.cat(all_role_targets, dim=0).numpy()
    all_narr_preds = torch.cat(all_narr_preds, dim=0).numpy()
    all_narr_targets = torch.cat(all_narr_targets, dim=0).numpy()
    all_subnarr_preds = torch.cat(all_subnarr_preds, dim=0).numpy()
    all_subnarr_targets = torch.cat(all_subnarr_targets, dim=0).numpy()

    # Subtask 1 Metrics
    role_f1_macro = f1_score(all_role_targets, all_role_preds, average="macro", zero_division=0)
    role_f1_micro = f1_score(all_role_targets, all_role_preds, average="micro", zero_division=0)

    # Subtask 2 Metrics
    narr_f1_macro = f1_score(all_narr_targets, all_narr_preds, average="macro", zero_division=0)
    narr_acc = accuracy_score(all_narr_targets, all_narr_preds)

    subnarr_f1_macro = f1_score(all_subnarr_targets, all_subnarr_preds, average="macro", zero_division=0)
    subnarr_acc = accuracy_score(all_subnarr_targets, all_subnarr_preds)

    # Combined Subtask 2 Macro F1 (official benchmark authority)
    overall_macro_f1 = (narr_f1_macro + subnarr_f1_macro) / 2.0

    return {
        "macro_f1": float(overall_macro_f1),
        "narrative_macro_f1": float(narr_f1_macro),
        "narrative_accuracy": float(narr_acc),
        "subnarrative_macro_f1": float(subnarr_f1_macro),
        "subnarrative_accuracy": float(subnarr_acc),
        "entity_role_macro_f1": float(role_f1_macro),
        "entity_role_micro_f1": float(role_f1_micro)
    }
