"""
Eligibility criteria concept finders: inclusion and exclusion.
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
        criterion_label: str = Field(description="Short label, e.g. 'Age ≥ 18' or 'Confirmed diagnosis'")
        domain: str = Field(description="demographic | clinical | treatment | enrollment | other")
        operational_detail: Optional[str] = Field(
            default=None,
            description="Operational detail if specified (e.g. '2 ICD-10 codes within 90 days')"
        )
        lookback_window: Optional[str] = Field(
            default=None,
            description="Time window if specified (e.g. '6 months prior to index')"
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    criteria: list[CriterionExtraction] = Field(
        description="All inclusion criteria found, in order of appearance"
    )
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_INC = """You are an expert RWE protocol analyst.
Extract ALL inclusion criteria (eligibility criteria for cohort entry) from the protocol text.

Inclusion criteria define who qualifies to enter the study cohort. They may include:
- Age requirements
- Diagnosis requirements (confirmed diagnosis, specific codes)
- Enrollment/activity requirements (continuous enrollment, database activity)
- Treatment requirements (received specific therapy)
- Follow-up requirements (minimum observation period)
- Clinical requirements (lab values, staging, performance status)

Rules:
- Extract EVERY inclusion criterion. Do not skip any.
- For each criterion, identify the domain (demographic, clinical, treatment, enrollment, other).
- If an operational definition is given (e.g. "2 ICD codes within 90 days"), capture it.
- If a lookback window is given (e.g. "6 months prior to index"), capture it.
- If criteria are implied but not stated, mark as "inferred".
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


# ══════════════════════════════════════════════════════════════════════════════
# Exclusion Criteria
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_EXC = "eligibility_exclusion"


class ExclusionCriteriaExtraction(BaseModel):
    class CriterionExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        criterion_label: str = Field(description="Short label, e.g. 'Prior malignancy' or 'Clinical trial'")
        domain: str = Field(description="demographic | clinical | treatment | enrollment | other")
        operational_detail: Optional[str] = Field(
            default=None,
            description="Operational detail if specified (e.g. 'any malignancy within 5 years')"
        )
        lookback_window: Optional[str] = Field(
            default=None,
            description="Time window if specified (e.g. '5 years prior to index')"
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    criteria: list[CriterionExtraction] = Field(
        description="All exclusion criteria found, in order of appearance"
    )
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_EXC = """You are an expert RWE protocol analyst.
Extract ALL exclusion criteria from the protocol text.

Exclusion criteria define who is removed from the study cohort after initial identification.
They may include:
- Prior conditions (e.g. prior malignancy, pregnancy)
- Missing data (e.g. missing gender, missing date of birth)
- Clinical trial participation
- Prior/concurrent treatment (washout violations)
- Insufficient data quality or coverage
- Age-related (e.g. pediatric exclusion)

Rules:
- Extract EVERY exclusion criterion. Do not skip any.
- For each criterion, identify the domain (demographic, clinical, treatment, enrollment, other).
- If an operational definition is given, capture it.
- If a lookback window is given (e.g. "within 5 years of index"), capture it.
- If criteria are implied but not stated, mark as "inferred".
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_context(chunks: list[RetrievedChunk], ta_warning: Optional[str], protocol_id: str) -> str:
    parts = [f"Protocol ID: {protocol_id}"]
    if ta_warning:
        parts.append(f"\nTA PACK WARNING: {ta_warning}\n")
    for i, c in enumerate(chunks, 1):
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
    candidate_metadata = {}

    for i, c in enumerate(extraction.criteria):
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

        candidate_metadata[candidate_id] = {
            "criterion_label": c.criterion_label,
            "domain": c.domain,
            "operational_detail": c.operational_detail,
            "lookback_window": c.lookback_window,
        }

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
        concept_metadata={"per_candidate": candidate_metadata},
    )

    return pack


# ── Main finder functions ────────────────────────────────────────────────────

def find_inclusion_criteria(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "inclusion criteria eligibility patient selection qualifying criteria",
        ta_pack, CONCEPT_INC,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_INC)
    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25,
        top_k_rerank=12,
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
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[InclusionCriteriaFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    pack = _build_eligibility_pack(CONCEPT_INC, protocol_id, extraction, chunks, model_used, used_adjudicator)

    print(f"[InclusionCriteriaFinder] Done. "
          f"{len(pack.candidates)} criteria | "
          f"confidence={extraction.overall_confidence:.2f}")

    return pack


def find_exclusion_criteria(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "exclusion criteria ineligible not eligible prior condition washout",
        ta_pack, CONCEPT_EXC,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_EXC)
    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25,
        top_k_rerank=12,
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
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[ExclusionCriteriaFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    pack = _build_eligibility_pack(CONCEPT_EXC, protocol_id, extraction, chunks, model_used, used_adjudicator)

    print(f"[ExclusionCriteriaFinder] Done. "
          f"{len(pack.candidates)} criteria | "
          f"confidence={extraction.overall_confidence:.2f}")

    return pack
