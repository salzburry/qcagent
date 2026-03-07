"""
TA pack loader.
Provides synonym expansion and ambiguity hotspot warnings per therapeutic area.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


PACK_DIR = Path(__file__).parent


@dataclass
class AmbiguityHotspot:
    concept: str
    severity: str       # high | medium | low
    warning: str


@dataclass
class TAPack:
    name: str
    concept_synonyms: dict[str, list[str]]      # concept → list of sponsor terms
    ambiguity_hotspots: list[AmbiguityHotspot]
    expected_concepts: list[str]
    section_priority: dict[str, list[str]]       # concept → preferred sections


def load_ta_pack(ta_name: str) -> Optional[TAPack]:
    """
    Load a TA pack by name.
    ta_name: "oncology" | "cardiovascular" | ...
    Returns None if pack not found — pipeline continues without TA hints.
    """
    path = PACK_DIR / f"{ta_name.lower()}.yaml"
    if not path.exists():
        print(f"[TAPack] No pack found for '{ta_name}'. Running without TA hints.")
        return None

    with open(path) as f:
        data = yaml.safe_load(f)

    hotspots = [
        AmbiguityHotspot(
            concept=h["concept"],
            severity=h["severity"],
            warning=h["warning"].strip(),
        )
        for h in data.get("ambiguity_hotspots", [])
    ]

    return TAPack(
        name=ta_name,
        concept_synonyms=data.get("concept_synonyms", {}),
        ambiguity_hotspots=hotspots,
        expected_concepts=data.get("expected_concepts", []),
        section_priority=data.get("section_priority", {}),
    )


def get_synonyms(pack: Optional[TAPack], concept: str) -> list[str]:
    """Return synonym list for a concept, empty list if no pack."""
    if pack is None:
        return []
    return pack.concept_synonyms.get(concept, [])


def get_hotspot_warning(pack: Optional[TAPack], concept: str) -> Optional[str]:
    """Return ambiguity warning for a concept if it exists."""
    if pack is None:
        return None
    for h in pack.ambiguity_hotspots:
        if h.concept == concept:
            return f"[{h.severity.upper()}] {h.warning}"
    return None


def get_section_priority(pack: Optional[TAPack], concept: str) -> list[str]:
    """Return preferred section names to prioritize in retrieval."""
    if pack is None:
        return []
    return pack.section_priority.get(concept, [])


def build_query_bank(base_query: str, pack: Optional[TAPack], concept: str) -> list[str]:
    """
    Build a multi-query list for retrieval.
    Base query + synonym expansions from TA pack.
    """
    synonyms = get_synonyms(pack, concept)
    queries = [base_query]
    # Add synonym-based queries
    for syn in synonyms[:4]:     # cap at 4 extra queries
        queries.append(f"{syn} definition protocol")
    return queries
