"""
Core evidence schemas.
EvidencePack is the central data structure of the entire pipeline.
Every concept finder produces one. Every row writer consumes one.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Canonical concept names ───────────────────────────────────────────────────
# Fixed inventory. Adding a concept = adding to this literal + new finder.

ConceptName = Literal[
    "index_date",
    "follow_up_start",
    "follow_up_end",
    "censoring_rules",
    "study_period",
    "enrollment_window",
    "baseline_window",
    "washout_period",
    "primary_endpoint",
    "secondary_endpoint",
    "eligibility_inclusion",
    "eligibility_exclusion",
    "data_source",
    "analysis_population",
    "key_covariate",
]

SourceType = Literal["narrative", "table", "appendix", "footnote", "amendment"]

ExplicitType = Literal["explicit", "inferred", "assumed", "ambiguous", "not_found"]


class EvidenceCandidate(BaseModel):
    """
    A single protocol passage that may support a concept.
    Produced by retrieval + reranking.
    """
    candidate_id: str                       # deterministic ID for stable references
    chunk_id: Optional[str] = None          # link back to indexed chunk for provenance
    snippet: str
    page: Optional[int] = None
    section_title: Optional[str] = None
    source_type: SourceType = "narrative"
    sponsor_term: Optional[str] = None      # e.g. "cohort entry" → maps to index_date
    canonical_term: Optional[str] = None    # normalized term from TA pack
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    llm_confidence: Optional[float] = None  # candidate-level LLM confidence
    explicit: ExplicitType = "explicit"


class EvidencePack(BaseModel):
    """
    Output of every concept finder.
    One per concept per protocol run.
    This is the handoff between automation and human review.
    """
    protocol_id: str
    concept: ConceptName
    candidates: list[EvidenceCandidate] = Field(default_factory=list)

    # Conflict/quality signals
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    low_retrieval_signal: bool = False      # top candidate score below threshold
    adjudicator_used: bool = False          # was second-pass model invoked?

    # Confidence
    overall_confidence: Optional[float] = None

    # Concept-specific metadata (preserves concept finder semantics)
    concept_metadata: Optional[dict] = None

    # Human review state
    requires_human_selection: bool = True
    selected_candidate_id: Optional[str] = None  # stable candidate_id, not index
    selected_candidate_ids: Optional[list[str]] = None  # for multi-row concepts (inclusion, exclusion, censoring)
    reviewer_notes: Optional[str] = None
    reviewer_override: Optional[str] = None  # free-text if no candidate is right

    # Run metadata
    finder_version: str = "0.3.0"
    model_used: str = ""
    prompt_version: str = ""

    @property
    def is_resolved(self) -> bool:
        return (
            self.selected_candidate is not None
            or self.reviewer_override is not None
        )

    @property
    def selected_candidate(self) -> Optional[EvidenceCandidate]:
        if self.selected_candidate_id is not None:
            for c in self.candidates:
                if c.candidate_id == self.selected_candidate_id:
                    return c
        return None

    def select_candidate(self, candidate_id: str) -> bool:
        """Select a candidate by its stable ID. Returns False if not found."""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                self.selected_candidate_id = candidate_id
                return True
        return False

    @property
    def selected_candidates(self) -> Optional[list[EvidenceCandidate]]:
        """For multi-row concepts: return only reviewer-selected candidates.
        Returns None if no multi-row selection has been made (draft mode)."""
        if self.selected_candidate_ids is None:
            return None
        id_set = set(self.selected_candidate_ids)
        return [c for c in self.candidates if c.candidate_id in id_set]

    @property
    def governing_text(self) -> Optional[str]:
        if self.reviewer_override:
            return self.reviewer_override
        if self.selected_candidate:
            return self.selected_candidate.snippet
        return None
