"""
Taxonomy Parser for SemEval 2025 Task 10.
Defines entity roles, domain-specific narrative and subnarrative taxonomies, and hierarchical label encoders.
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


MAIN_ROLES = ["Protagonist", "Antagonist", "Innocent"]

FINE_GRAINED_ROLES = [
    "Victim",
    "Hero",
    "Villain",
    "Scapegoat",
    "Instigator",
    "Beneficiary",
    "Defender",
    "Observer"
]

DOMAIN_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "ukraine_russia": {
        "Western Disinformation & Aggression": [
            "NATO provoked the conflict",
            "Western media spreads lies about Russia",
            "Sanctions destroy European economy"
        ],
        "Ukrainian Regime Legitimacy": [
            "Ukraine is a puppet of the US",
            "Ukrainian leadership is corrupt",
            "Far-right extremism in Ukrainian military"
        ],
        "Russian Military Actions": [
            "Special military operation is defensive",
            "Russian forces protect civilians",
            "Russian military superiority"
        ],
        "Peace & Negotiations": [
            "West rejects peace talks",
            "Russia is ready for diplomacy",
            "Negotiations are blocked by Ukraine"
        ]
    },
    "climate_change": {
        "Climate Science Skepticism": [
            "Global warming is a natural cycle",
            "Climate models are unreliable",
            "CO2 is beneficial for plants"
        ],
        "Economic Impact & Energy Policy": [
            "Green energy hurts the poor",
            "Renewables are unreliable",
            "Climate policies destroy industry"
        ],
        "Political Motivation": [
            "Climate agenda is for global control",
            "Climate alarmism enriches elites",
            "Taxation under green pretext"
        ]
    }
}


@dataclass
class TaxonomyNode:
    name: str
    is_parent: bool
    children: List[str]
    parent: str = None


class TaxonomyParser:
    """
    Parses and manages taxonomy labels for entity framing and narrative classification.
    """

    def __init__(self):
        self.main_roles = list(MAIN_ROLES)
        self.fine_roles = list(FINE_GRAINED_ROLES)
        self.all_roles = self.main_roles + self.fine_roles
        
        self.role2id = {r: i for i, r in enumerate(self.all_roles)}
        self.id2role = {i: r for i, r in enumerate(self.all_roles)}

        self.narratives: List[str] = []
        self.subnarratives: List[str] = []
        self.parent_child_map: Dict[str, List[str]] = {}
        self.child_parent_map: Dict[str, str] = {}

        self._build_taxonomy()

        self.narrative2id = {n: i for i, n in enumerate(self.narratives)}
        self.id2narrative = {i: n for i, n in enumerate(self.narratives)}
        self.subnarrative2id = {sn: i for i, sn in enumerate(self.subnarratives)}
        self.id2subnarrative = {i: sn for i, sn in enumerate(self.subnarratives)}

    def _build_taxonomy(self):
        for domain, domain_dict in DOMAIN_TAXONOMY.items():
            for parent, children in domain_dict.items():
                if parent not in self.narratives:
                    self.narratives.append(parent)
                self.parent_child_map[parent] = children
                for child in children:
                    if child not in self.subnarratives:
                        self.subnarratives.append(child)
                    self.child_parent_map[child] = parent

    def encode_roles(self, roles: List[str]) -> List[int]:
        return [self.role2id[r] for r in roles if r in self.role2id]

    def decode_roles(self, ids: List[int]) -> List[str]:
        return [self.id2role[i] for i in ids if i in self.id2role]

    def encode_narrative(self, narrative: str) -> int:
        return self.narrative2id.get(narrative, -1)

    def decode_narrative(self, nid: int) -> str:
        return self.id2narrative.get(nid, "Unknown")

    def encode_subnarrative(self, subnarrative: str) -> int:
        return self.subnarrative2id.get(subnarrative, -1)

    def decode_subnarrative(self, snid: int) -> str:
        return self.id2subnarrative.get(snid, "Unknown")

    @property
    def num_roles(self) -> int:
        return len(self.all_roles)

    @property
    def num_narratives(self) -> int:
        return len(self.narratives)

    @property
    def num_subnarratives(self) -> int:
        return len(self.subnarratives)
