"""
Outcomes row writer — expands endpoints and censoring rules into
proper variable families for the Outcomes (7.Outcomes) tab.

Naming convention follows org program-spec style:
  - Endpoint definition row uses sponsor_term or a normalized short name
  - Derived family: <EP_PREFIX>_EVENTFL, <EP_PREFIX>_EVENTDT, <EP_PREFIX>_TTOEVENT
  - Composite endpoints expand each component into its own sub-family
  - Censoring: <CENS_LABEL>_CENSFL, <CENS_LABEL>_CENSDT, <CENS_LABEL>_CENSREAS

Variable naming derives from the sponsor's endpoint term when available,
falling back to positional names (EP01, CENS01) only when no term exists.
"""

from __future__ import annotations
import re
from typing import Optional

from .base import RowWriter
from ..schemas.evidence import EvidencePack
from ..spec_output.spec_schema import VariableRow


def _normalize_var_name(term: str, max_len: int = 16) -> str:
    """Normalize a sponsor term into a valid variable name prefix.

    Examples:
        "Overall Survival" → "OS"
        "Progression-Free Survival" → "PFS"
        "Major Adverse Cardiovascular Events" → "MACE"
        "Time to Treatment Discontinuation" → "TTD"
        "Complete Response" → "CR"
    """
    if not term:
        return ""

    # Known abbreviation mappings (common RWE endpoint terms)
    _KNOWN_ABBREVS = {
        "overall survival": "OS",
        "progression-free survival": "PFS",
        "progression free survival": "PFS",
        "event-free survival": "EFS",
        "event free survival": "EFS",
        "disease-free survival": "DFS",
        "disease free survival": "DFS",
        "relapse-free survival": "RFS",
        "relapse free survival": "RFS",
        "major adverse cardiovascular event": "MACE",
        "major adverse cardiovascular events": "MACE",
        "time to treatment discontinuation": "TTD",
        "time to next treatment": "TTNT",
        "time to progression": "TTP",
        "time to response": "TTR",
        "time to event": "TTE",
        "complete response": "CR",
        "partial response": "PR",
        "objective response": "ORR",
        "objective response rate": "ORR",
        "overall response rate": "ORR",
        "duration of response": "DOR",
        "duration of treatment": "DOT",
        "myocardial infarction": "MI",
        "cardiovascular death": "CVDEATH",
        "all-cause mortality": "ACM",
        "all cause mortality": "ACM",
        "hospitalization": "HOSP",
    }

    lower = term.lower().strip()

    # Check known abbreviations
    for phrase, abbrev in _KNOWN_ABBREVS.items():
        if phrase in lower:
            return abbrev

    # If the term is already short and uppercase-ish, use it directly
    cleaned = re.sub(r'[^A-Za-z0-9_]', '', term.replace(' ', '_').replace('-', '_'))
    if len(cleaned) <= max_len:
        return cleaned.upper()

    # Build acronym from first letters of significant words
    words = re.findall(r'[A-Za-z]+', term)
    stop_words = {"of", "the", "to", "in", "for", "and", "or", "a", "an", "from", "by", "with"}
    significant = [w for w in words if w.lower() not in stop_words]
    if significant:
        acronym = "".join(w[0].upper() for w in significant)
        if 2 <= len(acronym) <= max_len:
            return acronym

    # Fallback: truncate
    return cleaned[:max_len].upper()


class EndpointWriter(RowWriter):
    """Expand primary_endpoint pack into outcome variable families.

    Uses endpoint metadata (is_composite, components, time_to_event) to
    generate more specific variable families that match org naming conventions.
    """
    concept = "primary_endpoint"
    target_field = "outcome_variables"

    def expand(
        self,
        pack: EvidencePack,
        data_source: str = "generic",
    ) -> list[VariableRow]:
        rows = []
        meta = (pack.concept_metadata or {}).get("per_candidate", {})
        gov = self._get_governing_candidate(pack)
        if not gov:
            return rows

        snippet = gov.snippet
        cm = meta.get(gov.candidate_id, {})

        # Derive variable prefix from sponsor term
        sponsor_term = gov.sponsor_term or ""
        ep_prefix = _normalize_var_name(sponsor_term) or "EP01"
        is_composite = cm.get("is_composite", False)
        components = cm.get("components", [])
        is_tte = cm.get("time_to_event", False)

        # Primary endpoint definition row
        rows.append(VariableRow(
            time_period="FU",
            variable=ep_prefix,
            label=sponsor_term or "Primary Endpoint",
            values="",
            definition=snippet,
            additional_notes=f"sponsor_term: {sponsor_term}" if sponsor_term else "",
            source_page=gov.page,
            confidence=gov.llm_confidence,
            explicit=gov.explicit,
        ))

        # Event flag
        rows.append(VariableRow(
            time_period="FU",
            variable=f"{ep_prefix}_EVENTFL",
            label=f"{sponsor_term or 'Primary endpoint'} — Event flag",
            values="1=Event occurred, 0=No event",
            definition=f"1 if {ep_prefix} event observed during follow-up, 0 otherwise",
            source_page=gov.page,
            confidence=gov.llm_confidence,
            explicit="inferred",
        ))

        # Event date
        rows.append(VariableRow(
            time_period="FU",
            variable=f"{ep_prefix}_EVENTDT",
            label=f"{sponsor_term or 'Primary endpoint'} — Event date",
            values="Date",
            definition=f"Date of {ep_prefix} event, if occurred",
            source_page=gov.page,
            confidence=gov.llm_confidence,
            explicit="inferred",
        ))

        # Time to event (only if endpoint is TTE)
        if is_tte:
            rows.append(VariableRow(
                time_period="FU",
                variable=f"{ep_prefix}_TTOEVENT",
                label=f"{sponsor_term or 'Primary endpoint'} — Time to event",
                values="Days from INDEX to event or censor",
                definition=f"Number of days from INDEX date to {ep_prefix} event or censoring date",
                source_page=gov.page,
                confidence=gov.llm_confidence,
                explicit="inferred",
            ))

        # Composite components: each gets its own sub-family
        if is_composite and components:
            for comp in components:
                comp_prefix = _normalize_var_name(comp) or comp.upper()[:8]
                comp_prefix = f"{ep_prefix}_{comp_prefix}"

                rows.append(VariableRow(
                    time_period="FU",
                    variable=f"{comp_prefix}_FL",
                    label=f"{comp} — Component flag",
                    values="1=Component event occurred, 0=No",
                    definition=f"Component of {ep_prefix}: {comp}",
                    additional_notes=f"Composite component of {sponsor_term or ep_prefix}",
                    source_page=gov.page,
                    confidence=gov.llm_confidence,
                    explicit="inferred",
                ))

                rows.append(VariableRow(
                    time_period="FU",
                    variable=f"{comp_prefix}_DT",
                    label=f"{comp} — Component date",
                    values="Date",
                    definition=f"Date of {comp} component event",
                    additional_notes=f"Composite component of {sponsor_term or ep_prefix}",
                    source_page=gov.page,
                    confidence=gov.llm_confidence,
                    explicit="inferred",
                ))

        return rows


class CensoringWriter(RowWriter):
    """Expand censoring_rules pack into censoring variable rows.

    Uses sponsor term and rule type to derive meaningful variable names
    instead of positional CENS01, CENS02.
    """
    concept = "censoring_rules"
    target_field = "outcome_variables"

    # Known censoring rule → variable name mappings
    _CENS_PREFIXES = {
        "death": "DEATH",
        "disenrollment": "DISENRL",
        "disenroll": "DISENRL",
        "data cutoff": "DTACUT",
        "data cut": "DTACUT",
        "end of data": "DTACUT",
        "loss to follow-up": "LTFU",
        "lost to follow-up": "LTFU",
        "treatment switch": "TRTSWITCH",
        "treatment discontinuation": "TRTDISC",
        "administrative": "ADMCENS",
    }

    def _derive_cens_prefix(self, label: str, index: int) -> str:
        """Derive a censoring variable prefix from the rule label."""
        lower = label.lower()
        for phrase, prefix in self._CENS_PREFIXES.items():
            if phrase in lower:
                return prefix

        # Try normalizing the sponsor term
        normalized = _normalize_var_name(label)
        if normalized and len(normalized) >= 2:
            return f"CENS_{normalized}"

        return f"CENS{index:02d}"

    def expand(
        self,
        pack: EvidencePack,
        data_source: str = "generic",
    ) -> list[VariableRow]:
        rows = []
        meta = (pack.concept_metadata or {}).get("per_candidate", {})
        candidates = self._get_all_candidates(pack)
        used_prefixes: set[str] = set()

        for i, cand in enumerate(candidates, 1):
            cm = meta.get(cand.candidate_id, {})
            rule_type = cm.get("rule_type", "event_based")
            applies_to = cm.get("applies_to", "all")
            label = cand.sponsor_term or f"Censoring Rule {i}"

            # Derive meaningful prefix
            prefix = self._derive_cens_prefix(label, i)
            # Dedupe: if prefix already used, add numeric suffix
            base_prefix = prefix
            suffix_num = 2
            while prefix in used_prefixes:
                prefix = f"{base_prefix}{suffix_num}"
                suffix_num += 1
            used_prefixes.add(prefix)

            # Censoring rule definition row
            rows.append(VariableRow(
                time_period="FU",
                variable=prefix,
                label=label,
                values="",
                definition=cand.snippet,
                additional_notes=f"type: {rule_type}; applies_to: {applies_to}",
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
            ))

            # Censoring flag
            rows.append(VariableRow(
                time_period="FU",
                variable=f"{prefix}_FL",
                label=f"{label} — Flag",
                values="1=Censored by this rule, 0=Not censored",
                definition=f"1 if patient censored due to {label.lower()}, 0 otherwise",
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit="inferred",
            ))

            # Censoring date
            rows.append(VariableRow(
                time_period="FU",
                variable=f"{prefix}_DT",
                label=f"{label} — Date",
                values="Date",
                definition=f"Date of censoring due to {label.lower()}",
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit="inferred",
            ))

            # Censoring reason (for composite/multiple rules)
            if rule_type in ("composite", "competing_risk") or len(candidates) > 1:
                rows.append(VariableRow(
                    time_period="FU",
                    variable=f"{prefix}_REAS",
                    label=f"{label} — Reason",
                    values="Character",
                    definition=f"Reason for censoring: {label}",
                    source_page=cand.page,
                    confidence=cand.llm_confidence,
                    explicit="inferred",
                ))

        return rows
