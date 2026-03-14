"""
Eligibility criteria concept finders (inclusion + exclusion).

Two-pass extraction to avoid token overflow:
  Pass 1 (inventory): Light schema — criterion_label, chunk_id, domain, confidence.
  Pass 2 (detail): Per-criterion detail — quoted_text, operational_detail, lookback.

This prevents the 1536-token (now 4096) truncation that occurred when extracting
20+ criteria with full schemas in a single call.
"""

from __future__ import annotations
import hashlib
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning, get_section_priority

FINDER_VERSION = "0.4.0"
PROMPT_VERSION = "0.4.0"
CONFIDENCE_THRESHOLD = 0.65


# ══════════════════════════════════════════════════════════════════════════════
# Pass 1: Inventory — lightweight list of criteria
# ══════════════════════════════════════════════════════════════════════════════

class CriterionInventory(BaseModel):
    """Pass 1: lightweight inventory of all criteria found."""
    class CriterionStub(BaseModel):
        chunk_id: Optional[str] = None
        criterion_label: str = Field(description="Short label, e.g. 'Age >= 18'")
        domain: str = Field(
            description="demographic | clinical | treatment | enrollment | other"
        )
        confidence: float = Field(ge=0.0, le=1.0)

    criteria: list[CriterionStub]
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_INVENTORY_INC = """You are an expert RWE protocol analyst.
Identify ALL inclusion criteria from the protocol text. Return a lightweight
inventory — just the label, domain, and confidence for each criterion.

Inclusion criteria define who is eligible for the study (age, diagnosis,
enrollment requirements, etc.).

Rules:
- List EVERY distinct inclusion criterion — do not merge separate criteria.
- Classify each by domain: demographic, clinical, treatment, enrollment, other.
- Return the chunk_id from each chunk header for provenance.
- Keep criterion_label short (under 60 characters).
- If different sections of the protocol contradict each other on inclusion
  criteria, set contradictions_found=true and explain in contradiction_detail.
- Respond ONLY with valid JSON matching the schema exactly."""


SYSTEM_PROMPT_INVENTORY_EXC = """You are an expert RWE protocol analyst.
Identify ALL exclusion criteria from the protocol text. Return a lightweight
inventory — just the label, domain, and confidence for each criterion.

Exclusion criteria define who is NOT eligible (prior treatments, comorbidities,
data quality requirements, etc.).

Rules:
- List EVERY distinct exclusion criterion — do not merge separate criteria.
- Classify each by domain: demographic, clinical, treatment, enrollment, other.
- Return the chunk_id from each chunk header for provenance.
- Keep criterion_label short (under 60 characters).
- If different sections of the protocol contradict each other on exclusion
  criteria, set contradictions_found=true and explain in contradiction_detail.
- Respond ONLY with valid JSON matching the schema exactly."""


# ══════════════════════════════════════════════════════════════════════════════
# Pass 2: Detail — full extraction per criterion
# ══════════════════════════════════════════════════════════════════════════════

class CriterionDetail(BaseModel):
    """Pass 2: full detail for a single criterion."""
    quoted_text: str = Field(description="Exact quoted text from the protocol")
    operational_detail: Optional[str] = Field(
        default=None,
        description="Operational detail: code list reference, measurement threshold, etc."
    )
    lookback_window: Optional[str] = Field(
        default=None,
        description="Time window if applicable, e.g. '12 months prior to index'"
    )
    explicit: ExplicitType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


SYSTEM_PROMPT_DETAIL = """You are an expert RWE protocol analyst.
Extract the FULL detail for exactly ONE criterion from the protocol text.

The criterion to extract: "{criterion_label}" (domain: {domain})

Rules:
- Find the exact passage that defines this criterion.
- Return the exact quoted_text — do not paraphrase.
- Capture operational details (code lists, thresholds, lab values).
- Capture lookback windows if applicable.
- If you cannot find this criterion in the text, set confidence to 0.
- Respond ONLY with valid JSON matching the schema exactly."""


# ══════════════════════════════════════════════════════════════════════════════
# Full extraction schemas (kept for backward compat / single-pass fallback)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Inclusion Criteria Finder (two-pass)
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_INC = "eligibility_inclusion"


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
            overall_confidence=0.0,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_INC)
    context = _build_context(chunks, ta_warning, protocol_id)

    pack = _two_pass_extract(
        concept=CONCEPT_INC,
        protocol_id=protocol_id,
        context=context,
        chunks=chunks,
        client=client,
        inventory_prompt=SYSTEM_PROMPT_INVENTORY_INC,
        ta_warning=ta_warning,
    )

    _conf = f"{pack.overall_confidence:.2f}" if pack.overall_confidence is not None else "N/A"
    print(f"[InclusionFinder] Done. "
          f"{len(pack.candidates)} criteria | "
          f"confidence={_conf}")

    return pack


# ══════════════════════════════════════════════════════════════════════════════
# Exclusion Criteria Finder (two-pass)
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_EXC = "eligibility_exclusion"


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
            overall_confidence=0.0,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_EXC)
    context = _build_context(chunks, ta_warning, protocol_id)

    pack = _two_pass_extract(
        concept=CONCEPT_EXC,
        protocol_id=protocol_id,
        context=context,
        chunks=chunks,
        client=client,
        inventory_prompt=SYSTEM_PROMPT_INVENTORY_EXC,
        ta_warning=ta_warning,
    )

    _conf = f"{pack.overall_confidence:.2f}" if pack.overall_confidence is not None else "N/A"
    print(f"[ExclusionFinder] Done. "
          f"{len(pack.candidates)} criteria | "
          f"confidence={_conf}")

    return pack


# ── Two-pass extraction engine ───────────────────────────────────────────────

def _two_pass_extract(
    concept: str,
    protocol_id: str,
    context: str,
    chunks: list[RetrievedChunk],
    client: LocalModelClient,
    inventory_prompt: str,
    ta_warning: Optional[str],
) -> EvidencePack:
    """Two-pass extraction: inventory then per-criterion detail.

    Pass 1: Get lightweight list of all criteria (small output, never truncated).
    Pass 2: For each criterion, extract full detail using only relevant chunks.
    """
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    # ── Pass 1: Inventory ──
    result = client.extract(
        system_prompt=inventory_prompt,
        user_prompt=context,
        schema=CriterionInventory,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
        max_tokens=2048,  # inventory is small
    )
    inventory = result.parsed
    model_used = result.model_used

    if not inventory.criteria:
        return EvidencePack(
            protocol_id=protocol_id, concept=concept,
            candidates=[], low_retrieval_signal=True,
            overall_confidence=0.0,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
            model_used=model_used,
        )

    # ── Pass 2: Detail per criterion ──
    candidates = []
    per_candidate = {}

    for stub in inventory.criteria:
        # Build focused context: if we have a chunk_id, use just that chunk + neighbors
        if stub.chunk_id and stub.chunk_id in chunk_by_id:
            # Use the specific chunk plus 2 neighbors for context
            focused_chunks = _get_chunk_neighborhood(chunks, stub.chunk_id, radius=2)
        else:
            # Fall back to top-5 chunks
            focused_chunks = chunks[:5]

        focused_context = _build_context(focused_chunks, ta_warning, protocol_id)

        detail_prompt = SYSTEM_PROMPT_DETAIL.format(
            criterion_label=stub.criterion_label,
            domain=stub.domain,
        )

        try:
            detail_result = client.extract(
                system_prompt=detail_prompt,
                user_prompt=focused_context,
                schema=CriterionDetail,
                use_adjudicator=False,
                prompt_version=PROMPT_VERSION,
                max_tokens=1024,  # single criterion detail is small
            )
            detail = detail_result.parsed
        except Exception as e:
            print(f"[EligibilityFinder] Detail extraction failed for '{stub.criterion_label}': {e}")
            # Create a minimal detail from the inventory stub
            detail = CriterionDetail(
                quoted_text=stub.criterion_label,
                explicit="inferred",
                confidence=stub.confidence * 0.5,
                reasoning=f"Detail extraction failed: {e}",
            )

        matching = chunk_by_id.get(stub.chunk_id) if stub.chunk_id else None

        candidate_id = hashlib.sha256(
            f"{protocol_id}:{concept}:{stub.chunk_id or ''}:{detail.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=stub.chunk_id,
            snippet=detail.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else "",
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=stub.criterion_label,
            canonical_term=concept,
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=detail.confidence,
            explicit=detail.explicit,
        ))

        per_candidate[candidate_id] = {
            "domain": stub.domain,
            "operational_detail": detail.operational_detail,
            "lookback_window": detail.lookback_window,
        }

    low_signal = len(chunks) < LOW_RETRIEVAL_THRESHOLD
    if chunks and chunks[0].rerank_score is not None:
        low_signal = low_signal or chunks[0].rerank_score < RERANK_SCORE_FLOOR

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=concept,
        candidates=candidates,
        contradictions_found=inventory.contradictions_found,
        contradiction_detail=inventory.contradiction_detail,
        overall_confidence=inventory.overall_confidence,
        low_retrieval_signal=low_signal,
        adjudicator_used=False,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used=model_used,
        prompt_version=PROMPT_VERSION,
    )
    pack.concept_metadata = {"per_candidate": per_candidate}

    return pack


def _get_chunk_neighborhood(
    chunks: list[RetrievedChunk],
    target_chunk_id: str,
    radius: int = 2,
) -> list[RetrievedChunk]:
    """Get a chunk and its document-order neighbors.

    Sorts by page number first so neighbors are page-neighbors,
    not ranking-neighbors. This matters because eligibility criteria
    are usually contiguous in the document.
    """
    # Sort by page for document-order neighborhood
    page_sorted = sorted(chunks, key=lambda c: (c.page or 0, c.chunk_id or ""))
    for i, ch in enumerate(page_sorted):
        if ch.chunk_id == target_chunk_id:
            start = max(0, i - radius)
            end = min(len(page_sorted), i + radius + 1)
            return page_sorted[start:end]
    # Not found — return the target chunk alone if possible
    return [ch for ch in chunks if ch.chunk_id == target_chunk_id] or chunks[:3]


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
