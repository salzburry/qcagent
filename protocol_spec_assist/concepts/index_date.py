"""
Index date concept finder.
Fixed workflow — not an agent.

Steps (always the same, every run):
  1. Build query bank from base query + TA pack synonyms
  2. Hybrid retrieval (dense + sparse) with section-priority boost
  3. Rerank candidates
  4. LLM extract → EvidencePack (schema-constrained)
  5. Confidence router → adjudicator if needed
  6. Return EvidencePack for human review
"""

from __future__ import annotations
import hashlib
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning, get_section_priority
from .base import build_context, compute_low_signal, try_adjudicator

CONCEPT = "index_date"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"


# ── LLM output schema for this concept ───────────────────────────────────────

class IndexDateExtraction(BaseModel):
    """Schema-constrained LLM output for index date extraction."""

    class CandidateExtraction(BaseModel):
        reasoning: str = Field(description="Why this snippet supports index date — explain your analysis before giving the structured answer")
        chunk_id: Optional[str] = Field(
            default=None,
            description="chunk_id from the input chunk, for provenance tracking"
        )
        quoted_text: str = Field(description="Exact quoted text from the chunk")
        summary: Optional[str] = Field(
            default=None,
            description="Brief normalized summary if the quote needs clarification"
        )
        section_title: str = Field(description="Section where this was found")
        sponsor_term: str = Field(description="Term used in protocol (e.g. cohort entry)")
        explicit: ExplicitType = Field(description="Whether explicitly stated or inferred")
        confidence: float = Field(ge=0.0, le=1.0)

    chain_of_thought: str = Field(
        default="",
        description="Think step by step about what index date definitions are present in the protocol text. "
        "Identify key phrases, assess whether they are explicit or inferred, and note any contradictions."
    )
    candidates: list[CandidateExtraction] = Field(
        default_factory=list,
        description="All candidate index date definitions found, ranked by confidence"
    )
    contradictions_found: bool = Field(
        default=False,
        description="True if different sections give conflicting index date definitions"
    )
    contradiction_detail: Optional[str] = Field(
        default=None,
        description="Describe contradictions if found"
    )
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ── System prompt (versioned) ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert RWE protocol analyst.
Your task is to identify the index date (cohort entry date) definition from protocol text.

The index date is the anchor date used to define cohort entry — it may be called:
"index date", "cohort entry", "treatment initiation", "first qualifying event",
"first dispensing", "line of therapy start", or similar.

IMPORTANT: Use the chain_of_thought field to reason about the protocol text BEFORE filling in candidates.
Think step by step — identify relevant passages, assess their specificity, and consider alternative interpretations.

Rules:
- Extract ONLY what the protocol text actually says. Do not invent definitions.
- If a definition is implied but not stated, mark it as "inferred".
- If sections contradict each other, set contradictions_found=true.
- Rank candidates by confidence — most likely governing definition first.
- If nothing relevant is found, return empty candidates list with low confidence.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


# ── Main finder function ──────────────────────────────────────────────────────

def find_index_date(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:
    """
    Run the index date concept finder workflow.
    Fixed sequence, always the same steps.
    """

    # Step 1: Build query bank
    queries = build_query_bank(
        "index date cohort entry first qualifying event treatment initiation",
        ta_pack,
        CONCEPT,
    )

    # Step 2: Hybrid retrieval — include tables (index dates often in design diagrams)
    priority_sections = get_section_priority(ta_pack, CONCEPT)
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
            protocol_id=protocol_id,
            concept=CONCEPT,
            candidates=[],
            low_retrieval_signal=True,
            requires_human_selection=True,
            finder_version=FINDER_VERSION,
            prompt_version=PROMPT_VERSION,
        )

    # Step 3: Get TA ambiguity warning
    ta_warning = get_hotspot_warning(ta_pack, CONCEPT)

    # Step 4: LLM extraction (schema-constrained)
    user_prompt = build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=IndexDateExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    # Step 5: Confidence router — adjudicator pass if needed
    adj_extraction, adj_model, used_adjudicator = try_adjudicator(
        client, SYSTEM_PROMPT, user_prompt, IndexDateExtraction,
        extraction, PROMPT_VERSION, "IndexDateFinder",
    )
    if used_adjudicator:
        extraction, model_used = adj_extraction, adj_model

    # Step 6: Build EvidencePack from extraction
    # Build chunk lookup by chunk_id for deterministic provenance
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    for i, c in enumerate(extraction.candidates):
        # Deterministic provenance via chunk_id (not fuzzy snippet matching)
        matching_chunk = chunk_by_id.get(c.chunk_id) if c.chunk_id else None

        # Deterministic from content, not from position in list
        candidate_id = hashlib.sha256(
            f"{protocol_id}:{CONCEPT}:{c.chunk_id or ''}:{c.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=c.chunk_id,
            snippet=c.quoted_text,
            page=matching_chunk.page if matching_chunk else None,
            section_title=matching_chunk.heading if matching_chunk else c.section_title,
            source_type=matching_chunk.source_type if matching_chunk else "narrative",
            sponsor_term=c.sponsor_term,
            canonical_term="index_date",
            retrieval_score=matching_chunk.retrieval_score if matching_chunk else None,
            rerank_score=matching_chunk.rerank_score if matching_chunk else None,
            llm_confidence=c.confidence,
            explicit=c.explicit,
        ))

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=CONCEPT,
        candidates=candidates,
        contradictions_found=extraction.contradictions_found,
        contradiction_detail=extraction.contradiction_detail,
        overall_confidence=extraction.overall_confidence,
        low_retrieval_signal=compute_low_signal(chunks),
        adjudicator_used=used_adjudicator,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used=model_used,
        prompt_version=PROMPT_VERSION,
    )

    print(f"[IndexDateFinder] Done. "
          f"{len(candidates)} candidates | "
          f"confidence={extraction.overall_confidence:.2f} | "
          f"contradictions={extraction.contradictions_found}")

    return pack
