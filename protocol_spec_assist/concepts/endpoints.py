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
from .base import CONFIDENCE_THRESHOLD, build_context, compute_low_signal, try_adjudicator

FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"


# ══════════════════════════════════════════════════════════════════════════════
# Follow-up End
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_FUE = "follow_up_end"

class FollowUpEndExtraction(BaseModel):
    class CandidateExtraction(BaseModel):
        reasoning: str
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

    chain_of_thought: str = Field(default="",
        description="Think step by step about the endpoints/follow-up definitions in the protocol text. "
        "Identify key passages, assess specificity, and note any ambiguities before structuring the answer."
    )
    candidates: list[CandidateExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


SYSTEM_PROMPT_FUE = """You are an expert RWE protocol analyst.
Extract the follow-up end definition — when observation of a patient stops.

This may be: a fixed date, end of continuous enrollment, death, data cutoff,
disenrollment, loss to follow-up, or a composite of multiple rules.

IMPORTANT: Use the chain_of_thought field to reason about the protocol text BEFORE filling in candidates.
Think step by step — identify relevant passages, assess their specificity, and consider alternative interpretations.

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
    context = build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_FUE,
        user_prompt=context,
        schema=FollowUpEndExtraction,
        use_adjudicator=False,
    )
    extraction, model_used = result.parsed, result.model_used

    adj_extraction, adj_model, used_adjudicator = try_adjudicator(
        client, SYSTEM_PROMPT_FUE, context, FollowUpEndExtraction,
        extraction, PROMPT_VERSION, "FollowUpEndFinder",
    )
    if used_adjudicator:
        extraction, model_used = adj_extraction, adj_model

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
        reasoning: str
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

    chain_of_thought: str = Field(default="",
        description="Think step by step about the endpoints/follow-up definitions in the protocol text. "
        "Identify key passages, assess specificity, and note any ambiguities before structuring the answer."
    )
    candidates: list[CandidateExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


SYSTEM_PROMPT_PE = """You are an expert RWE protocol analyst.
Extract the primary endpoint definition from protocol text.

Look for: primary outcome, primary objective, main endpoint, key endpoint.
For composite endpoints, list all component events.
For time-to-event endpoints, note the event definition and time-zero.

IMPORTANT: Use the chain_of_thought field to reason about the protocol text BEFORE filling in candidates.
Think step by step — identify relevant passages, assess their specificity, and consider alternative interpretations.

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
    context = build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_PE,
        user_prompt=context,
        schema=PrimaryEndpointExtraction,
        use_adjudicator=False,
    )
    extraction, model_used = result.parsed, result.model_used

    adj_extraction, adj_model, used_adjudicator = try_adjudicator(
        client, SYSTEM_PROMPT_PE, context, PrimaryEndpointExtraction,
        extraction, PROMPT_VERSION, "PrimaryEndpointFinder",
    )
    if used_adjudicator:
        extraction, model_used = adj_extraction, adj_model

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

    return EvidencePack(
        protocol_id=protocol_id,
        concept=concept,
        candidates=candidates,
        contradictions_found=extraction.contradictions_found,
        contradiction_detail=extraction.contradiction_detail,
        overall_confidence=extraction.overall_confidence,
        low_retrieval_signal=compute_low_signal(chunks),
        adjudicator_used=adjudicator_used,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used=model_used,
        prompt_version=PROMPT_VERSION,
    )
