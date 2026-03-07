"""
Follow-up end and primary endpoint concept finders.
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

FINDER_VERSION = "0.2.0"
PROMPT_VERSION = "0.2.0"
CONFIDENCE_THRESHOLD = 0.65


# ══════════════════════════════════════════════════════════════════════════════
# Follow-up End
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_FUE = "follow_up_end"

class FollowUpEndExtraction(BaseModel):
    class CandidateExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        summary: Optional[str] = None
        section_title: str
        sponsor_term: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        rule_type: str = Field(
            description="date_based | event_based | data_cutoff | enrollment_end | composite"
        )
        reasoning: str

    candidates: list[CandidateExtraction]
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_FUE = """You are an expert RWE protocol analyst.
Extract the follow-up end definition — when observation of a patient stops.

This may be: a fixed date, end of continuous enrollment, death, data cutoff,
disenrollment, loss to follow-up, or a composite of multiple rules.

Rules:
- Identify ALL follow-up end conditions — there are often multiple.
- Distinguish date-based vs event-based vs data cutoff rules.
- If different sections give different rules, flag as contradiction.
- Mark inferred definitions clearly.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_follow_up_end(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "follow-up end censoring data cutoff end of observation enrollment end",
        ta_pack, CONCEPT_FUE,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_FUE)
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
            protocol_id=protocol_id, concept=CONCEPT_FUE,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_FUE)
    context = _build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_FUE,
        user_prompt=context,
        schema=FollowUpEndExtraction,
        use_adjudicator=False,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        result = client.extract(
            system_prompt=SYSTEM_PROMPT_FUE,
            user_prompt=context,
            schema=FollowUpEndExtraction,
            use_adjudicator=True,
        )
        extraction, model_used = result.parsed, result.model_used
        used_adjudicator = True

    # Build pack first to get stable candidate_ids, then attach metadata
    pack = _build_pack(CONCEPT_FUE, protocol_id, extraction, chunks, model_used, used_adjudicator)

    # Attach concept-specific metadata keyed by candidate_id (not order-dependent)
    candidate_metadata = {}
    for ec, candidate in zip(extraction.candidates, pack.candidates):
        candidate_metadata[candidate.candidate_id] = {
            "rule_type": ec.rule_type,
        }
    pack.concept_metadata = {"per_candidate": candidate_metadata}

    return pack


# ══════════════════════════════════════════════════════════════════════════════
# Primary Endpoint
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_PE = "primary_endpoint"

class PrimaryEndpointExtraction(BaseModel):
    class CandidateExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        summary: Optional[str] = None
        section_title: str
        sponsor_term: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        is_composite: bool = Field(description="True if endpoint is composite of multiple events")
        components: list[str] = Field(
            default_factory=list,
            description="Component events if composite (e.g. ['CV death', 'MI', 'stroke'])"
        )
        time_to_event: bool = Field(description="True if time-to-event outcome")
        reasoning: str

    candidates: list[CandidateExtraction]
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_PE = """You are an expert RWE protocol analyst.
Extract the primary endpoint definition from protocol text.

Look for: primary outcome, primary objective, main endpoint, key endpoint.
For composite endpoints, list all component events.
For time-to-event endpoints, note the event definition and time-zero.

Rules:
- Distinguish primary from secondary endpoints explicitly.
- For MACE or composite endpoints, capture all components.
- Note whether time-to-event or binary/rate outcome.
- If endpoint definition appears in multiple sections with differences, flag contradiction.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_primary_endpoint(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "primary endpoint primary outcome primary objective key endpoint",
        ta_pack, CONCEPT_PE,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_PE)
    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=20,
        top_k_rerank=8,
        include_tables=True,
        priority_sections=priority_sections,
    )

    if not chunks:
        return EvidencePack(
            protocol_id=protocol_id, concept=CONCEPT_PE,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_PE)
    context = _build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_PE,
        user_prompt=context,
        schema=PrimaryEndpointExtraction,
        use_adjudicator=False,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        result = client.extract(
            system_prompt=SYSTEM_PROMPT_PE,
            user_prompt=context,
            schema=PrimaryEndpointExtraction,
            use_adjudicator=True,
        )
        extraction, model_used = result.parsed, result.model_used
        used_adjudicator = True

    # Build pack first to get stable candidate_ids, then attach metadata
    pack = _build_pack(CONCEPT_PE, protocol_id, extraction, chunks, model_used, used_adjudicator)

    # Attach concept-specific metadata keyed by candidate_id (not order-dependent)
    candidate_metadata = {}
    for ec, candidate in zip(extraction.candidates, pack.candidates):
        candidate_metadata[candidate.candidate_id] = {
            "is_composite": ec.is_composite,
            "components": ec.components,
            "time_to_event": ec.time_to_event,
        }
    pack.concept_metadata = {"per_candidate": candidate_metadata}

    return pack


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


LOW_RETRIEVAL_THRESHOLD = 3     # fewer chunks than this → low signal
RERANK_SCORE_FLOOR = 0.2       # top rerank score below this → low signal


def _build_pack(
    concept, protocol_id, extraction, chunks, model_used,
    adjudicator_used: bool = False,
) -> EvidencePack:
    # Build chunk lookup by chunk_id for deterministic provenance
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    for i, c in enumerate(extraction.candidates):
        # Deterministic provenance via chunk_id
        matching = chunk_by_id.get(c.chunk_id) if c.chunk_id else None

        # Deterministic from content, not from position in list
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
            sponsor_term=c.sponsor_term,
            canonical_term=concept,
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=c.confidence,
            explicit=c.explicit,
        ))

    # low_retrieval_signal: few chunks OR top rerank score too low
    low_signal = len(chunks) < LOW_RETRIEVAL_THRESHOLD
    if chunks and chunks[0].rerank_score is not None:
        low_signal = low_signal or chunks[0].rerank_score < RERANK_SCORE_FLOOR

    return EvidencePack(
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
