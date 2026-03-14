"""
Study design concept finders: data_prep_dates and censoring_rules.

data_prep_dates replaces the old monolithic study_period concept.
It targets the exact rows needed by the Data Prep tab:
  - Important Dates: INIT, INDEX, FUED, CENSDT, ENROLLDT
  - Time Periods: STUDY_PD, PRE_INT, FU, BASELINE, WASHOUT, ASSESS_*

This structured extraction is the fix for the vendor's GPT-4 failure on
study period — the model is now asked for specific, named rows rather than
a vague "extract the study period" instruction.
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
# Data Prep Dates (replaces old study_period)
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_DP = "data_prep_dates"

# Keep the old concept name as an alias for backward compatibility
CONCEPT_SP = "study_period"


class DataPrepExtraction(BaseModel):
    """Structured extraction targeting real Data Prep tab rows."""

    class ImportantDateExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        variable: str = Field(
            description="Variable name: INIT | INDEX | FUED | CENSDT | ENROLLDT | other"
        )
        label: str = Field(
            description="Human-readable label, e.g. 'Initial DLBCL diagnosis date'"
        )
        definition: str = Field(
            description="Operational definition, e.g. 'First ICD-10 C83.3 diagnosis in EHR'"
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    class TimePeriodExtraction(BaseModel):
        chunk_id: Optional[str] = None
        quoted_text: str
        time_period: str = Field(
            description="Period name: STUDY_PD | PRE_INT | FU | BASELINE | WASHOUT | ASSESS_* | other"
        )
        label: str = Field(
            description="Human-readable label, e.g. 'Pre-index period'"
        )
        definition: str = Field(
            description="Operational definition with date arithmetic, "
            "e.g. 'INDEX - 365 days to INDEX - 1 day'"
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str

    important_dates: list[ImportantDateExtraction]
    time_periods: list[TimePeriodExtraction]

    # Study-level fields
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


# Legacy schema kept for backward compatibility with existing evidence packs
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
    study_period_start: Optional[str] = None
    study_period_end: Optional[str] = None
    data_source: Optional[str] = None
    data_source_version: Optional[str] = None
    design_type: Optional[str] = None
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT_DP = """You are an expert RWE protocol analyst building a program specification.
Extract the study's important dates and time periods as they would appear
in a real-world data analytics program specification (Data Prep tab).

## Important Dates to look for:
- INIT: initial diagnosis date (first qualifying diagnosis)
- INDEX: index date (treatment initiation, line of therapy start, cohort entry)
- FUED: follow-up end date (last known activity, data cutoff, end of observation)
- CENSDT: censoring date (when observation is censored for a reason other than the event)
- ENROLLDT: enrollment/continuous enrollment start date

## Time Periods to look for:
- STUDY_PD: full study period (calendar start through calendar end)
- PRE_INT: pre-index period (e.g. 12 months before INDEX for baseline assessment)
- FU: follow-up period (INDEX to FUED or event)
- BASELINE: baseline assessment window (often same as PRE_INT or a subset)
- WASHOUT: treatment washout window (gap before INDEX to ensure treatment-naive)
- ASSESS_*: specific assessment windows (e.g. ASSESS_ECOG, ASSESS_LAB)

## Also extract:
- Data source name (e.g. Optum CDM, CPRD GOLD, Flatiron, COTA)
- Data source version or data cut date
- Study design type (retrospective_cohort, prospective_cohort, case_control, etc.)

Rules:
- For each date/period, provide the OPERATIONAL DEFINITION with date arithmetic
  (e.g. "INDEX - 365 to INDEX - 1" not just "12 months before index").
- Use the variable/period names above (INIT, INDEX, FUED, STUDY_PD, etc.).
- If the protocol defines custom assessment windows, use ASSESS_ prefix.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- It is OK to return an empty list if no dates/periods are found.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_data_prep_dates(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:
    """Extract Data Prep important dates and time periods from the protocol."""

    queries = build_query_bank(
        "study period index date follow-up enrollment baseline washout "
        "data source database study dates cohort entry diagnosis date",
        ta_pack, CONCEPT_SP,  # use study_period synonyms from TA pack
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
        system_prompt=SYSTEM_PROMPT_DP,
        user_prompt=context,
        schema=DataPrepExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT_DP,
                user_prompt=context,
                schema=DataPrepExtraction,
                use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception as e:
            print(f"[DataPrepFinder] Adjudicator unavailable ({e}), keeping first-pass result.")

    pack = _build_data_prep_pack(protocol_id, extraction, chunks, model_used, used_adjudicator)

    n_dates = len(extraction.important_dates)
    n_periods = len(extraction.time_periods)
    print(f"[DataPrepFinder] Done. "
          f"{n_dates} dates + {n_periods} periods | "
          f"confidence={extraction.overall_confidence:.2f} | "
          f"data_source={extraction.data_source or 'not found'}")

    return pack


def _build_data_prep_pack(
    protocol_id: str,
    extraction: DataPrepExtraction,
    chunks: list[RetrievedChunk],
    model_used: str,
    adjudicator_used: bool = False,
) -> EvidencePack:
    """Build EvidencePack from DataPrepExtraction.

    Creates one EvidenceCandidate per important date and per time period.
    Per-candidate metadata carries the structured date/period info for
    downstream row writers.
    """
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    per_candidate = {}

    # Important dates → candidates
    for dt in extraction.important_dates:
        matching = chunk_by_id.get(dt.chunk_id) if dt.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:data_prep_dates:date:{dt.variable}:{dt.chunk_id or ''}:{dt.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=dt.chunk_id,
            snippet=dt.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else dt.section_title,
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=dt.variable,
            canonical_term="data_prep_dates",
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=dt.confidence,
            explicit=dt.explicit,
        ))

        per_candidate[candidate_id] = {
            "row_type": "important_date",
            "variable": dt.variable,
            "label": dt.label,
            "definition": dt.definition,
        }

    # Time periods → candidates
    for tp in extraction.time_periods:
        matching = chunk_by_id.get(tp.chunk_id) if tp.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:data_prep_dates:period:{tp.time_period}:{tp.chunk_id or ''}:{tp.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=tp.chunk_id,
            snippet=tp.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else tp.section_title,
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=tp.time_period,
            canonical_term="data_prep_dates",
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=tp.confidence,
            explicit=tp.explicit,
        ))

        per_candidate[candidate_id] = {
            "row_type": "time_period",
            "time_period": tp.time_period,
            "label": tp.label,
            "definition": tp.definition,
        }

    low_signal = len(chunks) < LOW_RETRIEVAL_THRESHOLD
    if chunks and chunks[0].rerank_score is not None:
        low_signal = low_signal or chunks[0].rerank_score < RERANK_SCORE_FLOOR

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=CONCEPT_SP,  # keep "study_period" as concept name for backward compat
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

    pack.concept_metadata = {
        "per_candidate": per_candidate,
        "data_source": extraction.data_source,
        "data_source_version": extraction.data_source_version,
        "design_type": extraction.design_type,
    }

    return pack


# Legacy wrapper — calls the new finder but returns the same concept name
def find_study_period(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:
    """Legacy wrapper. Calls find_data_prep_dates internally."""
    return find_data_prep_dates(protocol_id, index, client, ta_pack)


# ══════════════════════════════════════════════════════════════════════════════
# Censoring Rules (unchanged workflow, updated version)
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
