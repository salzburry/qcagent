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

        # Determine whether we have real (non-static) evidence.
        # A pack with no candidates, or only candidates that lack both a
        # source page and LLM confidence, is "static-only" — its rows
        # must NOT look like evidence-backed rows.
        has_real_evidence = self._has_real_evidence(pack)

        for family_key, family_def in self.FAMILIES.items():
            # Always include core demographics (AGE, SEX, RACE, ETHNICITY)
            # Only include optional demographics if mentioned in evidence
            is_core = family_key in {"AGE", "SEX", "RACE", "ETHNICITY"}
            if not is_core and family_key not in mentioned_families:
                continue

            # Determine whether this specific family has evidence backing
            family_has_evidence = has_real_evidence and (
                is_core or family_key in mentioned_families
            )

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

                # Static-only evidence: mark as unresolved, no source page,
                # no confidence, and add review-required note
                if family_has_evidence:
                    source_page = gov_candidate.page if gov_candidate else None
                    confidence = gov_candidate.llm_confidence if gov_candidate else None
                    explicit_val = "explicit" if is_core else "inferred"
                    additional_notes = ""
                else:
                    source_page = None
                    confidence = None
                    explicit_val = "inferred"
                    additional_notes = (
                        "[UNRESOLVED] Auto-generated from standard template — "
                        "no protocol evidence found. Review required."
                    )

                rows.append(VariableRow(
                    time_period=family_def["time_period"],
                    variable=var,
                    label=label,
                    values="",
                    definition=definition,
                    code_lists_group="",
                    additional_notes=additional_notes,
                    source_page=source_page,
                    confidence=confidence,
                    explicit=explicit_val,
                ))

        return rows

    def _has_real_evidence(self, pack: EvidencePack) -> bool:
        """Check if the pack contains any candidates with real provenance.

        A candidate counts as real evidence only if it has a source page
        (indicating it was traced back to a specific protocol page).
        Static-template candidates have llm_confidence=0.5 but no page —
        those must NOT be treated as real evidence.

        Also rejects packs built entirely from static templates
        (model_used="static_template").
        """
        # Packs built entirely from static templates are never real evidence
        if getattr(pack, "model_used", None) == "static_template":
            return False
        if not pack.candidates:
            return False
        # Require at least one candidate with an actual source page
        return any(c.page is not None for c in pack.candidates)

    def _detect_mentioned_families(
        self,
        pack: EvidencePack,
        meta: dict,
    ) -> set[str]:
        """Detect which variable families are mentioned in the evidence text.

        Only scans candidates that have real provenance (a source page).
        Static-template candidates contain keywords like 'weight', 'height',
        'smoking' in their snippets by construction, so scanning them would
        falsely trigger optional families.
        """
        mentioned = set()
        # Only consider candidates with actual page provenance
        real_snippets = [c.snippet for c in pack.candidates if c.page is not None]
        if not real_snippets:
            return mentioned
        all_text = " ".join(s.lower() for s in real_snippets)

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
