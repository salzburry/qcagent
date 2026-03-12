"""
Eligibility criteria concept finders (inclusion + exclusion).
Same fixed workflow pattern as index_date.py.
"""

from __future__ import annotations
import hashlib
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning, get_section_priority

FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"
CONFIDENCE_THRESHOLD = 0.65


# ══════════════════════════════════════════════════════════════════════════════
# Inclusion Criteria
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_INC = "eligibility_inclusion"


class InclusionCriteriaExtraction(BaseModel):
    class CriterionExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        criterion_label: str = Field(description="Short label, e.g. 'Age requirement'")
        domain: str = Field(
            description="demographic | clinical | treatment | enrollment | other"
        )
        operational_detail: Optional[str] = Field(
            default=None,
            description="Operational detail: lookback window, code list ref, etc."
        )
        lookback_window: Optional[str] = Field(
            default=None,
            description="Time window if applicable, e.g. '12 months prior to index'"
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    criteria: list[CriterionExtraction]
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_INC = """You are an expert RWE protocol analyst.
Extract ALL inclusion criteria from the protocol text.

Inclusion criteria define who is eligible for the study. They may include:
- Age requirements
- Diagnosis requirements (with ICD codes or clinical descriptions)
- Prior treatment requirements
- Enrollment/database requirements (continuous enrollment, data availability)
- Lab values, clinical measures

Rules:
- Extract EVERY distinct inclusion criterion — do not merge separate criteria.
- Classify each by domain: demographic, clinical, treatment, enrollment, other.
- Capture lookback windows (e.g. "12 months prior to index date").
- Capture operational details like code list references.
- If criteria conflict across sections, set contradictions_found=true.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_inclusion_criteria(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "inclusion criteria eligible patients patient selection qualifying criteria",
        ta_pack, CONCEPT_INC,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_INC)
    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25,
        top_k_rerank=10,
        include_tables=True,
        priority_sections=priority_sections,
    )

    if not chunks:
        return EvidencePack(
            protocol_id=protocol_id, concept=CONCEPT_INC,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_INC)
    context = _build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_INC,
        user_prompt=context,
        schema=InclusionCriteriaExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT_INC,
                user_prompt=context,
                schema=InclusionCriteriaExtraction,
                use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[InclusionFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    pack = _build_eligibility_pack(
        CONCEPT_INC, protocol_id, extraction, chunks, model_used, used_adjudicator,
    )

    print(f"[InclusionFinder] Done. "
          f"{len(pack.candidates)} criteria | "
          f"confidence={extraction.overall_confidence:.2f}")

    return pack


# ══════════════════════════════════════════════════════════════════════════════
# Exclusion Criteria
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_EXC = "eligibility_exclusion"


class ExclusionCriteriaExtraction(BaseModel):
    class CriterionExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        criterion_label: str = Field(description="Short label, e.g. 'Prior cancer'")
        domain: str = Field(
            description="demographic | clinical | treatment | enrollment | other"
        )
        operational_detail: Optional[str] = Field(
            default=None,
            description="Operational detail: lookback window, code list ref, etc."
        )
        lookback_window: Optional[str] = Field(
            default=None,
            description="Time window if applicable, e.g. '6 months prior to index'"
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    criteria: list[CriterionExtraction]
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_EXC = """You are an expert RWE protocol analyst.
Extract ALL exclusion criteria from the protocol text.

Exclusion criteria define who is NOT eligible. They may include:
- Prior treatments or procedures
- Comorbidities or diagnoses
- Lab values out of range
- Data quality requirements (insufficient data history)
- Pregnancy, age limits, etc.

Rules:
- Extract EVERY distinct exclusion criterion — do not merge separate criteria.
- Classify each by domain: demographic, clinical, treatment, enrollment, other.
- Capture lookback windows and washout periods.
- Capture operational details like code list references.
- If criteria conflict across sections, set contradictions_found=true.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_exclusion_criteria(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "exclusion criteria ineligible not eligible prior therapy washout",
        ta_pack, CONCEPT_EXC,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_EXC)
    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25,
        top_k_rerank=10,
        include_tables=True,
        priority_sections=priority_sections,
    )

    if not chunks:
        return EvidencePack(
            protocol_id=protocol_id, concept=CONCEPT_EXC,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_EXC)
    context = _build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_EXC,
        user_prompt=context,
        schema=ExclusionCriteriaExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT_EXC,
                user_prompt=context,
                schema=ExclusionCriteriaExtraction,
                use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[ExclusionFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    pack = _build_eligibility_pack(
        CONCEPT_EXC, protocol_id, extraction, chunks, model_used, used_adjudicator,
    )

    print(f"[ExclusionFinder] Done. "
          f"{len(pack.candidates)} criteria | "
          f"confidence={extraction.overall_confidence:.2f}")

    return pack


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_context(chunks: list[RetrievedChunk], ta_warning: Optional[str], protocol_id: str) -> str:
    parts = [f"Protocol ID: {protocol_id}"]
    if ta_warning:
        parts.append(f"\nTA PACK WARNING: {ta_warning}\n")
    for c in chunks:
        parts.append(
            f"[chunk_id={c.chunk_id} | Section: {c.heading} | Type: {c.source_type} | "
            f"Page: {c.page} | Score: {c.score:.2f}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


LOW_RETRIEVAL_THRESHOLD = 3
RERANK_SCORE_FLOOR = 0.2


def _build_eligibility_pack(
    concept, protocol_id, extraction, chunks, model_used,
    adjudicator_used: bool = False,
) -> EvidencePack:
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    for c in extraction.criteria:
        matching = chunk_by_id.get(c.chunk_id) if c.chunk_id else None

        candidate_id = hashlib.sha256(
            f"{protocol_id}:{concept}:{c.chunk_id or ''}:{c.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=c.chunk_id,
            snippet=c.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else c.section_title,
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=c.criterion_label,
            canonical_term=concept,
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=c.confidence,
            explicit=c.explicit,
        ))

    low_signal = len(chunks) < LOW_RETRIEVAL_THRESHOLD
    if chunks and chunks[0].rerank_score is not None:
        low_signal = low_signal or chunks[0].rerank_score < RERANK_SCORE_FLOOR

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=concept,
        candidates=candidates,
        contradictions_found=extraction.contradictions_found,
        contradiction_detail=extraction.contradiction_detail,
        overall_confidence=extraction.overall_confidence,
        low_retrieval_signal=low_signal,
        adjudicator_used=adjudicator_used,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used=model_used,
        prompt_version=PROMPT_VERSION,
    )

    # Per-candidate metadata with domain/lookback info
    per_candidate = {}
    for ec, cand in zip(extraction.criteria, pack.candidates):
        per_candidate[cand.candidate_id] = {
            "domain": ec.domain,
            "operational_detail": ec.operational_detail,
            "lookback_window": ec.lookback_window,
        }
    pack.concept_metadata = {"per_candidate": per_candidate}

    return pack
