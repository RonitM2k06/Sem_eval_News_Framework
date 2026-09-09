"""
Graph Difference Engine for Structural Narrative Interventions.
Calculates structural diffs between graphs across counterfactuals, cross-lingual pairs, and temporal timesteps.
"""

from typing import Dict, List, Any


class GraphDiffEngine:
    """
    Computes graph structural diffs between original and modified/intervened NarrativeGraph states.
    """

    def compute_diff(
        self,
        orig_graph: Dict[str, Any],
        modified_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        orig_nodes = {n["id"]: n for n in orig_graph.get("nodes", []) if "id" in n}
        mod_nodes = {n["id"]: n for n in modified_graph.get("nodes", []) if "id" in n}

        orig_edges = {(e["source"], e["target"]): e for e in orig_graph.get("edges", []) if "source" in e and "target" in e}
        mod_edges = {(e["source"], e["target"]): e for e in modified_graph.get("edges", []) if "source" in e and "target" in e}

        nodes_added = [mod_nodes[nid] for nid in mod_nodes if nid not in orig_nodes]
        nodes_removed = [orig_nodes[nid] for nid in orig_nodes if nid not in mod_nodes]

        nodes_modified = []
        for nid in orig_nodes:
            if nid in mod_nodes:
                o_label = orig_nodes[nid].get("label")
                m_label = mod_nodes[nid].get("label")
                o_type = orig_nodes[nid].get("type")
                m_type = mod_nodes[nid].get("type")
                if o_label != m_label or o_type != m_type:
                    nodes_modified.append({
                        "node_id": nid,
                        "orig": orig_nodes[nid],
                        "modified": mod_nodes[nid]
                    })

        edges_added = [mod_edges[eid] for eid in mod_edges if eid not in orig_edges]
        edges_removed = [orig_edges[eid] for eid in orig_edges if eid not in mod_edges]

        structural_change_score = (
            len(nodes_added) + len(nodes_removed) + len(nodes_modified) + len(edges_added) + len(edges_removed)
        ) / max(1, len(orig_nodes) + len(orig_edges))

        return {
            "nodes_added": nodes_added,
            "nodes_removed": nodes_removed,
            "nodes_modified": nodes_modified,
            "edges_added": edges_added,
            "edges_removed": edges_removed,
            "nodes_added_count": len(nodes_added),
            "nodes_removed_count": len(nodes_removed),
            "nodes_modified_count": len(nodes_modified),
            "edges_added_count": len(edges_added),
            "edges_removed_count": len(edges_removed),
            "structural_change_score": float(round(min(1.0, structural_change_score), 4))
        }
