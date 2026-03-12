"""
Study design concept finders: study_period and censoring_rules.
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
# Study Period / Data Source
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_SP = "study_period"


class StudyPeriodExtraction(BaseModel):
    class CandidateExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        section_title: str
        sponsor_term: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    candidates: list[CandidateExtraction]

    # Study-level fields (not per-candidate)
    study_period_start: Optional[str] = Field(
        default=None, description="Start date or description of study period"
    )
    study_period_end: Optional[str] = Field(
        default=None, description="End date or description of study period"
    )
    data_source: Optional[str] = Field(
        default=None, description="Database/data source name (e.g. Optum, CPRD, MarketScan)"
    )
    data_source_version: Optional[str] = Field(
        default=None, description="Data source version or cut date"
    )
    design_type: Optional[str] = Field(
        default=None,
        description="retrospective_cohort | prospective_cohort | case_control | cross_sectional | other"
    )

    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_SP = """You are an expert RWE protocol analyst.
Extract the study period, data source, and study design type from the protocol text.

Look for:
- Study period dates (start/end)
- Data source name (e.g. Optum CDM, CPRD GOLD, MarketScan, Flatiron)
- Data source version or data cut date
- Study design type (retrospective cohort, prospective cohort, case-control, etc.)

Rules:
- Extract ALL passages that define the study period or data source.
- Capture exact date ranges if stated.
- Identify the database/data source by name.
- If design type is explicitly stated, capture it. Otherwise mark as inferred.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_study_period(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "study period data source database study dates retrospective cohort design",
        ta_pack, CONCEPT_SP,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_SP)
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
            protocol_id=protocol_id, concept=CONCEPT_SP,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_SP)
    context = _build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_SP,
        user_prompt=context,
        schema=StudyPeriodExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT_SP,
                user_prompt=context,
                schema=StudyPeriodExtraction,
                use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[StudyPeriodFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    pack = _build_pack(CONCEPT_SP, protocol_id, extraction, chunks, model_used, used_adjudicator)

    # Study-level metadata (not per-candidate)
    pack.concept_metadata = {
        "study_period_start": extraction.study_period_start,
        "study_period_end": extraction.study_period_end,
        "data_source": extraction.data_source,
        "data_source_version": extraction.data_source_version,
        "design_type": extraction.design_type,
    }

    print(f"[StudyPeriodFinder] Done. "
          f"{len(pack.candidates)} candidates | "
          f"confidence={extraction.overall_confidence:.2f} | "
          f"data_source={extraction.data_source or 'not found'}")

    return pack


# ══════════════════════════════════════════════════════════════════════════════
# Censoring Rules
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_CR = "censoring_rules"


class CensoringRulesExtraction(BaseModel):
    class RuleExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        rule_label: str = Field(description="Short label, e.g. 'Death censoring'")
        rule_type: str = Field(
            description="event_based | date_based | administrative | competing_risk | composite"
        )
        applies_to: Optional[str] = Field(
            default=None,
            description="Which endpoint(s) this rule applies to, e.g. 'primary endpoint' or 'all'"
        )
        section_title: str
        sponsor_term: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    rules: list[RuleExtraction]
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_CR = """You are an expert RWE protocol analyst.
Extract ALL censoring rules from the protocol text.

Censoring rules define when a patient's observation ends for reasons other than
the primary event. They may include:
- Death (if not the primary endpoint)
- End of enrollment / disenrollment
- Data cutoff / end of data availability
- Loss to follow-up
- Competing events (for cause-specific analyses)
- Administrative censoring at a fixed date
- Treatment switching / discontinuation

Rules:
- Extract EVERY distinct censoring rule.
- Classify each: event_based, date_based, administrative, competing_risk, composite.
- Note which endpoint(s) each rule applies to if specified.
- If censoring rules differ across endpoints, capture all variations.
- If sections contradict, set contradictions_found=true.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_censoring_rules(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "censoring rules censored competing event loss to follow-up administrative censoring",
        ta_pack, CONCEPT_CR,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_CR)
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
            protocol_id=protocol_id, concept=CONCEPT_CR,
            candidates=[], low_retrieval_signal=True,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT_CR)
    context = _build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT_CR,
        user_prompt=context,
        schema=CensoringRulesExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT_CR,
                user_prompt=context,
                schema=CensoringRulesExtraction,
                use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[CensoringRulesFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    pack = _build_censoring_pack(
        protocol_id, extraction, chunks, model_used, used_adjudicator,
    )

    print(f"[CensoringRulesFinder] Done. "
          f"{len(pack.candidates)} rules | "
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


def _build_pack(
    concept, protocol_id, extraction, chunks, model_used,
    adjudicator_used: bool = False,
) -> EvidencePack:
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    for c in extraction.candidates:
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
            sponsor_term=c.sponsor_term,
            canonical_term=concept,
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=c.confidence,
            explicit=c.explicit,
        ))

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


def _build_censoring_pack(
    protocol_id, extraction, chunks, model_used,
    adjudicator_used: bool = False,
) -> EvidencePack:
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    for c in extraction.rules:
        matching = chunk_by_id.get(c.chunk_id) if c.chunk_id else None

        candidate_id = hashlib.sha256(
            f"{protocol_id}:censoring_rules:{c.chunk_id or ''}:{c.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=c.chunk_id,
            snippet=c.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else c.section_title,
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=c.sponsor_term,
            canonical_term="censoring_rules",
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
        concept=CONCEPT_CR,
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

    # Per-candidate metadata with rule_type/applies_to
    per_candidate = {}
    for ec, cand in zip(extraction.rules, pack.candidates):
        per_candidate[cand.candidate_id] = {
            "rule_type": ec.rule_type,
            "applies_to": ec.applies_to,
        }
    pack.concept_metadata = {"per_candidate": per_candidate}

    return pack
