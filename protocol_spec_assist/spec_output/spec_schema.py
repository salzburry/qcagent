"""
Program spec schema — the draft spec generated from evidence packs.

NOTE: This generates a DRAFT spec from the top-ranked candidate per concept.
If a human has selected a candidate (selected_candidate_id is set), that
candidate is used instead. The spec should always be reviewed before use.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack


class SpecEntry(BaseModel):
    """A single spec entry with value, provenance, and confidence."""
    value: str = ""
    source_snippet: str = ""
    page: Optional[int] = None
    confidence: Optional[float] = None
    explicit: str = "explicit"
    notes: str = ""


class CriterionEntry(BaseModel):
    """An inclusion or exclusion criterion."""
    label: str = ""
    value: str = ""
    domain: str = "other"
    lookback_window: Optional[str] = None
    operational_detail: Optional[str] = None
    confidence: Optional[float] = None
    explicit: str = "explicit"
    page: Optional[int] = None


class CensoringRuleEntry(BaseModel):
    """A censoring rule entry."""
    label: str = ""
    value: str = ""
    rule_type: str = "event_based"
    applies_to: Optional[str] = None
    confidence: Optional[float] = None
    explicit: str = "explicit"
    page: Optional[int] = None


class StudyDesign(BaseModel):
    design_type: SpecEntry = Field(default_factory=SpecEntry)
    data_source: SpecEntry = Field(default_factory=SpecEntry)
    study_period_start: SpecEntry = Field(default_factory=SpecEntry)
    study_period_end: SpecEntry = Field(default_factory=SpecEntry)


class ProgramSpec(BaseModel):
    """Draft program spec generated from evidence packs."""
    protocol_id: str = ""
    spec_version: str = "0.3.0"
    generation_mode: str = "draft"  # draft | reviewed

    study_design: StudyDesign = Field(default_factory=StudyDesign)
    index_date: SpecEntry = Field(default_factory=SpecEntry)
    follow_up_end: SpecEntry = Field(default_factory=SpecEntry)
    primary_endpoint: SpecEntry = Field(default_factory=SpecEntry)

    inclusion_criteria: list[CriterionEntry] = Field(default_factory=list)
    exclusion_criteria: list[CriterionEntry] = Field(default_factory=list)
    censoring_rules: list[CensoringRuleEntry] = Field(default_factory=list)

    qc_warnings: list[str] = Field(default_factory=list)


def _get_governing_candidate(pack: EvidencePack):
    """Get the governing candidate: selected by reviewer, or top-ranked if draft."""
    if pack.selected_candidate is not None:
        return pack.selected_candidate
    # Draft mode: use top candidate (first in list = highest ranked by finder)
    if pack.candidates:
        return pack.candidates[0]
    return None


def _make_entry(pack: EvidencePack) -> SpecEntry:
    """Build a SpecEntry from a pack's governing candidate."""
    candidate = _get_governing_candidate(pack)
    if candidate is None:
        return SpecEntry(notes="No candidates found")

    return SpecEntry(
        value=candidate.snippet,
        source_snippet=candidate.snippet,
        page=candidate.page,
        confidence=candidate.llm_confidence,
        explicit=candidate.explicit,
        notes=f"sponsor_term: {candidate.sponsor_term or 'n/a'}",
    )


def build_program_spec(
    packs: dict[str, EvidencePack],
    protocol_id: str = "",
    qc_warnings: Optional[list[str]] = None,
) -> ProgramSpec:
    """
    Translate evidence packs → draft ProgramSpec.

    Uses selected_candidate_id if a human has reviewed, otherwise uses
    the top-ranked candidate (draft mode).
    """
    spec = ProgramSpec(
        protocol_id=protocol_id,
        qc_warnings=qc_warnings or [],
    )

    # Determine generation mode
    any_reviewed = any(p.selected_candidate_id is not None for p in packs.values())
    spec.generation_mode = "reviewed" if any_reviewed else "draft"

    # Single-value concepts
    if "index_date" in packs:
        spec.index_date = _make_entry(packs["index_date"])
    if "follow_up_end" in packs:
        spec.follow_up_end = _make_entry(packs["follow_up_end"])
    if "primary_endpoint" in packs:
        spec.primary_endpoint = _make_entry(packs["primary_endpoint"])

    # Study design from study_period pack
    if "study_period" in packs:
        sp = packs["study_period"]
        meta = sp.concept_metadata or {}

        spec.study_design.study_period_start = SpecEntry(
            value=meta.get("study_period_start") or "",
        )
        spec.study_design.study_period_end = SpecEntry(
            value=meta.get("study_period_end") or "",
        )
        spec.study_design.data_source = SpecEntry(
            value=meta.get("data_source") or "",
            notes=f"version: {meta.get('data_source_version') or 'n/a'}",
        )
        spec.study_design.design_type = SpecEntry(
            value=meta.get("design_type") or "",
        )

    # Multi-value concepts: inclusion criteria
    if "eligibility_inclusion" in packs:
        inc_pack = packs["eligibility_inclusion"]
        meta = (inc_pack.concept_metadata or {}).get("per_candidate", {})
        # Honor reviewer selection if present; otherwise include all (draft mode)
        candidates = inc_pack.selected_candidates if inc_pack.selected_candidates is not None else inc_pack.candidates
        for cand in candidates:
            cm = meta.get(cand.candidate_id, {})
            spec.inclusion_criteria.append(CriterionEntry(
                label=cand.sponsor_term or "",
                value=cand.snippet,
                domain=cm.get("domain", "other"),
                lookback_window=cm.get("lookback_window"),
                operational_detail=cm.get("operational_detail"),
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
                page=cand.page,
            ))

    # Multi-value concepts: exclusion criteria
    if "eligibility_exclusion" in packs:
        exc_pack = packs["eligibility_exclusion"]
        meta = (exc_pack.concept_metadata or {}).get("per_candidate", {})
        candidates = exc_pack.selected_candidates if exc_pack.selected_candidates is not None else exc_pack.candidates
        for cand in candidates:
            cm = meta.get(cand.candidate_id, {})
            spec.exclusion_criteria.append(CriterionEntry(
                label=cand.sponsor_term or "",
                value=cand.snippet,
                domain=cm.get("domain", "other"),
                lookback_window=cm.get("lookback_window"),
                operational_detail=cm.get("operational_detail"),
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
                page=cand.page,
            ))

    # Multi-value concepts: censoring rules
    if "censoring_rules" in packs:
        cr_pack = packs["censoring_rules"]
        meta = (cr_pack.concept_metadata or {}).get("per_candidate", {})
        candidates = cr_pack.selected_candidates if cr_pack.selected_candidates is not None else cr_pack.candidates
        for cand in candidates:
            cm = meta.get(cand.candidate_id, {})
            spec.censoring_rules.append(CensoringRuleEntry(
                label=cand.sponsor_term or "",
                value=cand.snippet,
                rule_type=cm.get("rule_type", "event_based"),
                applies_to=cm.get("applies_to"),
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
                page=cand.page,
            ))

    return spec
