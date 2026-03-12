"""
Program spec schema — the structured output that maps to the Excel program spec.
Built from evidence packs after extraction. This is the auto-translation layer.

Each section matches a tab or block in the typical RWE program spec Excel workbook.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class SpecCriterion(BaseModel):
    """One inclusion or exclusion criterion row."""
    criterion_id: str
    type: str                           # inclusion | exclusion
    criterion_label: str                # short label
    domain: str                         # demographic | clinical | treatment | enrollment | other
    description: str                    # full quoted text from protocol
    operational_definition: Optional[str] = None
    lookback_window: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    confidence: float = 0.0
    explicit: str = "explicit"          # explicit | inferred | assumed


class SpecEndpoint(BaseModel):
    """One endpoint row."""
    endpoint_id: str
    type: str                           # primary | secondary
    label: str
    description: str
    is_composite: bool = False
    components: list[str] = Field(default_factory=list)
    time_to_event: bool = False
    page: Optional[int] = None
    section: Optional[str] = None
    confidence: float = 0.0


class SpecCensoringRule(BaseModel):
    """One censoring rule row."""
    rule_id: str
    rule_label: str
    rule_type: str                      # event_based | date_based | administrative | competing_risk
    description: str
    applies_to: Optional[str] = None    # which endpoint(s)
    page: Optional[int] = None
    section: Optional[str] = None
    confidence: float = 0.0


class ProgramSpec(BaseModel):
    """
    The complete auto-generated program spec.
    One per protocol. Built from evidence packs.
    """
    # Header
    protocol_id: str
    protocol_title: Optional[str] = None
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    generator_version: str = "0.3.0"

    # Study Design
    design_type: Optional[str] = None
    study_period_start: Optional[str] = None
    study_period_end: Optional[str] = None
    data_source: Optional[str] = None
    data_source_version: Optional[str] = None

    # Index Date
    index_date_definition: Optional[str] = None
    index_date_sponsor_term: Optional[str] = None
    index_date_confidence: float = 0.0
    index_date_page: Optional[int] = None

    # Follow-up
    follow_up_end_definition: Optional[str] = None
    follow_up_end_confidence: float = 0.0

    # Eligibility
    inclusion_criteria: list[SpecCriterion] = Field(default_factory=list)
    exclusion_criteria: list[SpecCriterion] = Field(default_factory=list)

    # Endpoints
    endpoints: list[SpecEndpoint] = Field(default_factory=list)

    # Censoring
    censoring_rules: list[SpecCensoringRule] = Field(default_factory=list)

    # QC metadata
    concepts_extracted: list[str] = Field(default_factory=list)
    concepts_with_low_signal: list[str] = Field(default_factory=list)
    concepts_with_contradictions: list[str] = Field(default_factory=list)


def build_program_spec(
    protocol_id: str,
    evidence_packs: dict[str, dict],
    protocol_title: Optional[str] = None,
) -> ProgramSpec:
    """
    Build a ProgramSpec from evidence packs.
    This is the auto-translation: evidence packs → structured spec.
    """
    spec = ProgramSpec(
        protocol_id=protocol_id,
        protocol_title=protocol_title,
    )

    for concept, pack_data in evidence_packs.items():
        spec.concepts_extracted.append(concept)

        if pack_data.get("low_retrieval_signal"):
            spec.concepts_with_low_signal.append(concept)
        if pack_data.get("contradictions_found"):
            spec.concepts_with_contradictions.append(concept)

    # ── Index Date ────────────────────────────────────────────────────────
    if "index_date" in evidence_packs:
        pack = evidence_packs["index_date"]
        candidates = pack.get("candidates", [])
        if candidates:
            top = candidates[0]
            spec.index_date_definition = top.get("snippet", "")
            spec.index_date_sponsor_term = top.get("sponsor_term")
            spec.index_date_confidence = top.get("llm_confidence", 0.0) or 0.0
            spec.index_date_page = top.get("page")

    # ── Follow-up End ─────────────────────────────────────────────────────
    if "follow_up_end" in evidence_packs:
        pack = evidence_packs["follow_up_end"]
        candidates = pack.get("candidates", [])
        if candidates:
            top = candidates[0]
            spec.follow_up_end_definition = top.get("snippet", "")
            spec.follow_up_end_confidence = top.get("llm_confidence", 0.0) or 0.0

    # ── Study Period / Data Source ────────────────────────────────────────
    if "study_period" in evidence_packs:
        pack = evidence_packs["study_period"]
        meta = pack.get("concept_metadata") or {}
        spec.study_period_start = meta.get("study_period_start")
        spec.study_period_end = meta.get("study_period_end")
        spec.data_source = meta.get("data_source")
        spec.data_source_version = meta.get("data_source_version")
        spec.design_type = meta.get("design_type")

    # ── Inclusion Criteria ────────────────────────────────────────────────
    if "eligibility_inclusion" in evidence_packs:
        pack = evidence_packs["eligibility_inclusion"]
        per_candidate = (pack.get("concept_metadata") or {}).get("per_candidate", {})
        for i, c in enumerate(pack.get("candidates", []), 1):
            cid = c.get("candidate_id", f"inc_{i}")
            meta = per_candidate.get(cid, {})
            spec.inclusion_criteria.append(SpecCriterion(
                criterion_id=f"INC-{i:02d}",
                type="inclusion",
                criterion_label=meta.get("criterion_label", c.get("sponsor_term", "")),
                domain=meta.get("domain", "other"),
                description=c.get("snippet", ""),
                operational_definition=meta.get("operational_detail"),
                lookback_window=meta.get("lookback_window"),
                page=c.get("page"),
                section=c.get("section_title"),
                confidence=c.get("llm_confidence", 0.0) or 0.0,
                explicit=c.get("explicit", "explicit"),
            ))

    # ── Exclusion Criteria ────────────────────────────────────────────────
    if "eligibility_exclusion" in evidence_packs:
        pack = evidence_packs["eligibility_exclusion"]
        per_candidate = (pack.get("concept_metadata") or {}).get("per_candidate", {})
        for i, c in enumerate(pack.get("candidates", []), 1):
            cid = c.get("candidate_id", f"exc_{i}")
            meta = per_candidate.get(cid, {})
            spec.exclusion_criteria.append(SpecCriterion(
                criterion_id=f"EXC-{i:02d}",
                type="exclusion",
                criterion_label=meta.get("criterion_label", c.get("sponsor_term", "")),
                domain=meta.get("domain", "other"),
                description=c.get("snippet", ""),
                operational_definition=meta.get("operational_detail"),
                lookback_window=meta.get("lookback_window"),
                page=c.get("page"),
                section=c.get("section_title"),
                confidence=c.get("llm_confidence", 0.0) or 0.0,
                explicit=c.get("explicit", "explicit"),
            ))

    # ── Primary Endpoint ──────────────────────────────────────────────────
    if "primary_endpoint" in evidence_packs:
        pack = evidence_packs["primary_endpoint"]
        per_candidate = (pack.get("concept_metadata") or {}).get("per_candidate", {})
        for i, c in enumerate(pack.get("candidates", []), 1):
            cid = c.get("candidate_id", f"ep_{i}")
            meta = per_candidate.get(cid, {})
            spec.endpoints.append(SpecEndpoint(
                endpoint_id=f"EP-{i:02d}",
                type="primary",
                label=c.get("sponsor_term", ""),
                description=c.get("snippet", ""),
                is_composite=meta.get("is_composite", False),
                components=meta.get("components", []),
                time_to_event=meta.get("time_to_event", False),
                page=c.get("page"),
                section=c.get("section_title"),
                confidence=c.get("llm_confidence", 0.0) or 0.0,
            ))

    # ── Censoring Rules ───────────────────────────────────────────────────
    if "censoring_rules" in evidence_packs:
        pack = evidence_packs["censoring_rules"]
        per_candidate = (pack.get("concept_metadata") or {}).get("per_candidate", {})
        for i, c in enumerate(pack.get("candidates", []), 1):
            cid = c.get("candidate_id", f"cr_{i}")
            meta = per_candidate.get(cid, {})
            spec.censoring_rules.append(SpecCensoringRule(
                rule_id=f"CR-{i:02d}",
                rule_label=meta.get("rule_label", c.get("sponsor_term", "")),
                rule_type=meta.get("rule_type", ""),
                description=c.get("snippet", ""),
                applies_to=meta.get("applies_to"),
                page=c.get("page"),
                section=c.get("section_title"),
                confidence=c.get("llm_confidence", 0.0) or 0.0,
            ))

    return spec
