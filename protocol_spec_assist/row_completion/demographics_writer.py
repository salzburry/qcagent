"""
Demographics row writer — deterministic row-family expansion.

For each demographic variable family (AGE, SEX, RACE, ETHNICITY, etc.),
generates the full variable family (base + numeric + date + value variants).
Definitions come from data_sources/registry.py (source-specific).
"""

from __future__ import annotations
from typing import Optional

from .base import RowWriter
from ..schemas.evidence import EvidencePack
from ..spec_output.spec_schema import VariableRow
from ..data_sources.registry import get_definition


class DemographicsWriter(RowWriter):
    concept = "demographics"
    target_field = "demographics"

    # Variable families: base_var → (list of variables, default label)
    FAMILIES = {
        "AGE": {
            "variables": ["AGE", "AGEN", "AGEGR", "AGEGRN"],
            "label": "Age at index date",
            "time_period": "INDEX",
        },
        "SEX": {
            "variables": ["SEX", "SEXN"],
            "label": "Sex",
            "time_period": "INDEX",
        },
        "RACE": {
            "variables": ["RACE", "RACEN"],
            "label": "Race",
            "time_period": "INDEX",
        },
        "ETHNICITY": {
            "variables": ["ETH", "ETHN"],
            "label": "Ethnicity",
            "time_period": "INDEX",
        },
        "REGION": {
            "variables": ["REGION", "REGIONN"],
            "label": "Geographic region",
            "time_period": "INDEX",
        },
        "BMI": {
            "variables": ["BMI", "BMIN", "BMIDT", "BMIV"],
            "label": "Body mass index",
            "time_period": "BASELINE",
        },
        "WEIGHT": {
            "variables": ["WEIGHT", "WEIGHTN", "WEIGHTDT", "WEIGHTV"],
            "label": "Weight",
            "time_period": "BASELINE",
        },
        "HEIGHT": {
            "variables": ["HEIGHT", "HEIGHTN", "HEIGHTDT", "HEIGHTV"],
            "label": "Height",
            "time_period": "BASELINE",
        },
        "SMOKING": {
            "variables": ["SMOKING", "SMOKINGN"],
            "label": "Smoking status",
            "time_period": "BASELINE",
        },
    }

    # Variable suffixes and their meaning
    SUFFIX_LABELS = {
        "": "Character value",
        "N": "Numeric coded value",
        "GR": "Grouped/categorized",
        "GRN": "Grouped numeric code",
        "DT": "Date of measurement",
        "V": "Raw/observed value",
    }

    def expand(
        self,
        pack: EvidencePack,
        data_source: str = "generic",
    ) -> list[VariableRow]:
        rows = []
        gov_candidate = self._get_governing_candidate(pack)
        meta = (pack.concept_metadata or {}).get("per_candidate", {})

        # Check which families are mentioned in the evidence
        mentioned_families = self._detect_mentioned_families(pack, meta)

        for family_key, family_def in self.FAMILIES.items():
            # Always include core demographics (AGE, SEX, RACE, ETHNICITY)
            # Only include optional demographics if mentioned in evidence
            is_core = family_key in {"AGE", "SEX", "RACE", "ETHNICITY"}
            if not is_core and family_key not in mentioned_families:
                continue

            for var in family_def["variables"]:
                # Get source-specific definition
                definition = get_definition(data_source, var, "")
                if not definition:
                    # Construct default definition from base
                    base = family_def["variables"][0]
                    suffix = var[len(base):]
                    suffix_desc = self.SUFFIX_LABELS.get(suffix, "")
                    definition = f"{family_def['label']} — {suffix_desc}" if suffix else family_def["label"]

                # Determine label for this specific variable
                base = family_def["variables"][0]
                suffix = var[len(base):]
                if suffix:
                    suffix_desc = self.SUFFIX_LABELS.get(suffix, suffix)
                    label = f"{family_def['label']} ({suffix_desc})"
                else:
                    label = family_def["label"]

                rows.append(VariableRow(
                    time_period=family_def["time_period"],
                    variable=var,
                    label=label,
                    values="",
                    definition=definition,
                    code_lists_group="",
                    additional_notes="",
                    source_page=gov_candidate.page if gov_candidate else None,
                    confidence=gov_candidate.llm_confidence if gov_candidate else None,
                    explicit="explicit" if is_core else "inferred",
                ))

        return rows

    def _detect_mentioned_families(
        self,
        pack: EvidencePack,
        meta: dict,
    ) -> set[str]:
        """Detect which variable families are mentioned in the evidence text."""
        mentioned = set()
        all_text = " ".join(c.snippet.lower() for c in pack.candidates)

        family_keywords = {
            "BMI": ["bmi", "body mass index"],
            "WEIGHT": ["weight", "kg", "lbs"],
            "HEIGHT": ["height", "cm", "inches"],
            "SMOKING": ["smoking", "tobacco", "cigarette", "nicotine"],
            "REGION": ["region", "geography", "geographic", "state", "country"],
        }

        for family_key, keywords in family_keywords.items():
            if any(kw in all_text for kw in keywords):
                mentioned.add(family_key)

        return mentioned
