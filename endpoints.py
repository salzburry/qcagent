"""
Follow-up end and primary endpoint concept finders.
Same fixed workflow pattern as index_date.py.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning

FINDER_VERSION = "0.1.0"
PROMPT_VERSION = "0.1.0"
CONFIDENCE_THRESHOLD = 0.65


# ══════════════════════════════════════════════════════════════════════════════
# Follow-up End
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_FUE = "follow_up_end"

class FollowUpEndExtraction(BaseModel):
    class CandidateExtraction(BaseModel):
        snippet: str
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

    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25,
        top_k_rerank=10,
        include_tables=True,
    )

    if not chunks:
        return EvidencePack(
            protocol_id=protocol_id, concept=CONCEPT_FUE,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_FUE)
    context = _build_context(chunks, ta_warning, protocol_id)

    extraction, model_used = client.extract(
        system_prompt=SYSTEM_PROMPT_FUE,
        user_prompt=context,
        schema=FollowUpEndExtraction,
        use_adjudicator=False,
    )

    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        extraction, model_used = client.extract(
            system_prompt=SYSTEM_PROMPT_FUE,
            user_prompt=context,
            schema=FollowUpEndExtraction,
            use_adjudicator=True,
        )

    return _build_pack(CONCEPT_FUE, protocol_id, extraction, chunks, model_used)


# ══════════════════════════════════════════════════════════════════════════════
# Primary Endpoint
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_PE = "primary_endpoint"

class PrimaryEndpointExtraction(BaseModel):
    class CandidateExtraction(BaseModel):
        snippet: str
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
- Respond ONLY with valid JSON matching the schema exactly."""


def find_primary_endpoint(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_pack(
        "primary endpoint primary outcome primary objective key endpoint",
        ta_pack, CONCEPT_PE,
    )

    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=20,
        top_k_rerank=8,
        include_tables=True,
    )

    if not chunks:
        return EvidencePack(
            protocol_id=protocol_id, concept=CONCEPT_PE,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_PE)
    context = _build_context(chunks, ta_warning, protocol_id)

    extraction, model_used = client.extract(
        system_prompt=SYSTEM_PROMPT_PE,
        user_prompt=context,
        schema=PrimaryEndpointExtraction,
        use_adjudicator=False,
    )

    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        extraction, model_used = client.extract(
            system_prompt=SYSTEM_PROMPT_PE,
            user_prompt=context,
            schema=PrimaryEndpointExtraction,
            use_adjudicator=True,
        )

    return _build_pack(CONCEPT_PE, protocol_id, extraction, chunks, model_used)


# ── Shared helpers ────────────────────────────────────────────────────────────

def build_query_pack(base: str, pack, concept: str) -> list[str]:
    from ..ta_packs.loader import build_query_bank
    return build_query_bank(base, pack, concept)


def _build_context(chunks: list[RetrievedChunk], ta_warning: Optional[str], protocol_id: str) -> str:
    parts = [f"Protocol ID: {protocol_id}"]
    if ta_warning:
        parts.append(f"\n⚠ TA PACK WARNING: {ta_warning}\n")
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Section: {c.heading} | Type: {c.source_type} | "
            f"Page: {c.page} | Score: {c.score:.2f}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_pack(concept, protocol_id, extraction, chunks, model_used) -> EvidencePack:
    candidates = []
    for c in extraction.candidates:
        matching = next(
            (ch for ch in chunks if c.snippet[:50] in ch.text or ch.text[:50] in c.snippet),
            None
        )
        candidates.append(EvidenceCandidate(
            snippet=c.snippet,
            page=matching.page if matching else None,
            section_title=c.section_title or (matching.heading if matching else None),
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=c.sponsor_term,
            canonical_term=concept,
            retrieval_score=matching.dense_score if matching else 0.0,
            rerank_score=matching.rerank_score if matching else None,
            explicit=c.explicit,
        ))

    return EvidencePack(
        protocol_id=protocol_id,
        concept=concept,
        candidates=candidates,
        contradictions_found=extraction.contradictions_found,
        contradiction_detail=extraction.contradiction_detail,
        low_retrieval_signal=len(chunks) < 3,
        adjudicator_used=extraction.overall_confidence < CONFIDENCE_THRESHOLD,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used=model_used,
        prompt_version=PROMPT_VERSION,
    )
