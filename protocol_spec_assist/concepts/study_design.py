"""
Study period, data source, and censoring rules concept finders.
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
CONCEPT_DS = "data_source"


class StudyPeriodExtraction(BaseModel):
    class PeriodDetail(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    study_period_start: Optional[str] = Field(
        default=None,
        description="Start of the study period (date or description, e.g. 'January 2016')"
    )
    study_period_end: Optional[str] = Field(
        default=None,
        description="End of the study period (date or description, e.g. 'June 2024' or 'most recent data cut')"
    )
    data_source: Optional[str] = Field(
        default=None,
        description="Primary data source (e.g. 'Flatiron LBCL', 'Optum CDM', 'CPRD GOLD')"
    )
    data_source_version: Optional[str] = Field(
        default=None,
        description="Data version or vintage if specified"
    )
    design_type: Optional[str] = Field(
        default=None,
        description="Study design (e.g. 'retrospective cohort', 'prospective observational')"
    )
    details: list[PeriodDetail] = Field(
        description="Supporting evidence passages"
    )
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_SP = """You are an expert RWE protocol analyst.
Extract the study period, data source, and study design from the protocol text.

Look for:
- Study period: start and end dates of the observation window
- Data source: the database(s) used (e.g. Flatiron, MarketScan, CPRD, Optum)
- Data version or vintage
- Study design type (retrospective cohort, prospective, registry, etc.)

Rules:
- Extract dates as written (e.g. "January 1, 2016" or "Q1 2016")
- If the data source has a version or cut date, capture it
- If multiple databases are used, capture the primary one
- IMPORTANT: Return the chunk_id from each chunk header for provenance
- Return the exact quoted_text from the protocol, not a paraphrase
- Respond ONLY with valid JSON matching the schema exactly."""


# ══════════════════════════════════════════════════════════════════════════════
# Censoring Rules
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_CR = "censoring_rules"


class CensoringRulesExtraction(BaseModel):
    class CensoringRule(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        rule_label: str = Field(description="Short label, e.g. 'Death', 'Disenrollment', 'Data cutoff'")
        rule_type: str = Field(
            description="event_based | date_based | administrative | competing_risk | composite"
        )
        applies_to: Optional[str] = Field(
            default=None,
            description="Which endpoint this censoring applies to (e.g. 'all', 'primary only', 'OS')"
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    rules: list[CensoringRule] = Field(
        description="All censoring rules found"
    )
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_CR = """You are an expert RWE protocol analyst.
Extract ALL censoring rules from the protocol text.

Censoring rules define when a patient's observation period ends without experiencing
the outcome of interest. They may include:
- Death (may be outcome OR censoring depending on endpoint)
- Disenrollment / loss to follow-up
- Data cutoff / end of data availability
- End of continuous enrollment
- Administrative censoring at study end date
- Competing events (events that preclude the outcome)
- Treatment switching or discontinuation

Rules:
- Extract EVERY censoring rule mentioned. Do not skip any.
- Identify the rule type (event_based, date_based, administrative, competing_risk, composite).
- Note which endpoint(s) the rule applies to if specified.
- Different endpoints may have different censoring rules — capture all.
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


# ── Main finder functions ────────────────────────────────────────────────────

def find_study_period(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "study period data source database study design observation window data cut",
        ta_pack, CONCEPT_SP,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT_SP)
    chunks = index.search(
        query=queries[0],
        protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=20,
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
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[StudyPeriodFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    # Build candidates from detail passages
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}
    candidates = []
    for i, d in enumerate(extraction.details):
        matching = chunk_by_id.get(d.chunk_id) if d.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:{CONCEPT_SP}:{d.chunk_id or ''}:{d.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=d.chunk_id,
            snippet=d.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else d.section_title,
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=None,
            canonical_term=CONCEPT_SP,
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=d.confidence,
            explicit=d.explicit,
        ))

    low_signal = len(chunks) < LOW_RETRIEVAL_THRESHOLD
    if chunks and chunks[0].rerank_score is not None:
        low_signal = low_signal or chunks[0].rerank_score < RERANK_SCORE_FLOOR

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=CONCEPT_SP,
        candidates=candidates,
        contradictions_found=extraction.contradictions_found,
        contradiction_detail=extraction.contradiction_detail,
        overall_confidence=extraction.overall_confidence,
        low_retrieval_signal=low_signal,
        adjudicator_used=used_adjudicator,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used=model_used,
        prompt_version=PROMPT_VERSION,
        concept_metadata={
            "study_period_start": extraction.study_period_start,
            "study_period_end": extraction.study_period_end,
            "data_source": extraction.data_source,
            "data_source_version": extraction.data_source_version,
            "design_type": extraction.design_type,
        },
    )

    print(f"[StudyPeriodFinder] Done. "
          f"period={extraction.study_period_start} to {extraction.study_period_end} | "
          f"source={extraction.data_source} | "
          f"confidence={extraction.overall_confidence:.2f}")

    return pack


def find_censoring_rules(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:

    queries = build_query_bank(
        "censoring rules censored competing event loss to follow-up administrative censoring end of data",
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
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[CensoringRulesFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    # Build candidates from rules
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}
    candidates = []
    candidate_metadata = {}

    for i, r in enumerate(extraction.rules):
        matching = chunk_by_id.get(r.chunk_id) if r.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:{CONCEPT_CR}:{r.chunk_id or ''}:{r.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=r.chunk_id,
            snippet=r.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else r.section_title,
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=r.rule_label,
            canonical_term=CONCEPT_CR,
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=r.confidence,
            explicit=r.explicit,
        ))

        candidate_metadata[candidate_id] = {
            "rule_label": r.rule_label,
            "rule_type": r.rule_type,
            "applies_to": r.applies_to,
        }

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
        adjudicator_used=used_adjudicator,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used=model_used,
        prompt_version=PROMPT_VERSION,
        concept_metadata={"per_candidate": candidate_metadata},
    )

    print(f"[CensoringRulesFinder] Done. "
          f"{len(candidates)} rules | "
          f"confidence={extraction.overall_confidence:.2f}")

    return pack
