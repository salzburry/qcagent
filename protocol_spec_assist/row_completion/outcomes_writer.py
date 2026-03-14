"""
Outcomes row writer — expands endpoints and censoring rules into
proper variable families for the Outcomes (7.Outcomes) tab.

Endpoint families:
  - Primary endpoint → EVENTDT, EVENTFL, TTOEVENT
  - Censoring → CENSFL, CENSDT, CENSREAS

Each censoring rule gets its own variable family row.
"""

from __future__ import annotations
from typing import Optional

from .base import RowWriter
from ..schemas.evidence import EvidencePack
from ..spec_output.spec_schema import VariableRow


class EndpointWriter(RowWriter):
    """Expand primary_endpoint pack into outcome variable families."""
    concept = "primary_endpoint"
    target_field = "outcome_variables"

    # Standard endpoint variable family
    ENDPOINT_FAMILY = [
        ("EVENTFL", "Event flag", "1=Event occurred, 0=No event"),
        ("EVENTDT", "Event date", "Date"),
        ("TTOEVENT", "Time to event", "Days from INDEX to event or censor"),
    ]

    def expand(
        self,
        pack: EvidencePack,
        data_source: str = "generic",
    ) -> list[VariableRow]:
        rows = []
        gov = self._get_governing_candidate(pack)
        if not gov:
            return rows

        snippet = gov.snippet
        sponsor_term = gov.sponsor_term or "PRIMARY_EP"

        # Primary endpoint definition row
        rows.append(VariableRow(
            time_period="FU",
            variable="PRIMARY_EP",
            label="Primary Endpoint",
            values="",
            definition=snippet,
            additional_notes=f"sponsor_term: {sponsor_term}",
            source_page=gov.page,
            confidence=gov.llm_confidence,
            explicit=gov.explicit,
        ))

        # Expand into variable family
        for var_suffix, label, values in self.ENDPOINT_FAMILY:
            var_name = f"PRIMARY_{var_suffix}"
            rows.append(VariableRow(
                time_period="FU",
                variable=var_name,
                label=f"Primary endpoint — {label}",
                values=values,
                definition=f"Derived from PRIMARY_EP: {label.lower()}",
                source_page=gov.page,
                confidence=gov.llm_confidence,
                explicit="inferred",
            ))

        return rows


class CensoringWriter(RowWriter):
    """Expand censoring_rules pack into censoring variable rows."""
    concept = "censoring_rules"
    target_field = "outcome_variables"

    def expand(
        self,
        pack: EvidencePack,
        data_source: str = "generic",
    ) -> list[VariableRow]:
        rows = []
        meta = (pack.concept_metadata or {}).get("per_candidate", {})
        candidates = self._get_all_candidates(pack)

        for i, cand in enumerate(candidates, 1):
            cm = meta.get(cand.candidate_id, {})
            rule_type = cm.get("rule_type", "event_based")
            applies_to = cm.get("applies_to", "all")
            label = cand.sponsor_term or f"Censoring Rule {i}"

            # Censoring rule definition row
            rows.append(VariableRow(
                time_period="FU",
                variable=f"CENS{i:02d}",
                label=label,
                values="",
                definition=cand.snippet,
                additional_notes=f"type: {rule_type}; applies_to: {applies_to}",
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
            ))

            # Censoring flag and date
            rows.append(VariableRow(
                time_period="FU",
                variable=f"CENS{i:02d}FL",
                label=f"{label} — Flag",
                values="1=Censored by this rule, 0=Not censored",
                definition=f"Derived from CENS{i:02d}: censoring flag",
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit="inferred",
            ))
            rows.append(VariableRow(
                time_period="FU",
                variable=f"CENS{i:02d}DT",
                label=f"{label} — Date",
                values="Date",
                definition=f"Derived from CENS{i:02d}: date of censoring",
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit="inferred",
            ))

        return rows
