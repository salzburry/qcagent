"""
Cohort definition concept finder.

Extracts cohort definitions from the protocol for StudyPop Section B.
Cohorts define how the study population is divided into analysis groups
(e.g. treatment vs comparator, exposed vs unexposed, case vs control).

These are distinct from eligibility criteria (Section A) which define
who enters the study. Cohort definitions define how those eligible patients
are classified into groups for analysis.
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

CONCEPT = "cohort_definitions"
FINDER_VERSION = "0.5.0"
PROMPT_VERSION = "0.5.0"


class CohortDefinitionExtraction(BaseModel):
    """Schema for cohort definition extraction."""

    chain_of_thought: str = Field(
        description="Think step by step about cohort definitions in the protocol. "
        "Identify treatment arms, comparator groups, and analysis populations before structuring."
    )

    class CohortExtraction(BaseModel):
        reasoning: str = Field(description="Why this cohort definition is relevant and how it was identified")
        chunk_id: Optional[str] = None
        quoted_text: str = Field(description="Exact quoted text from the protocol")
        cohort_label: str = Field(
            description="Short label for the cohort, e.g. 'Treatment cohort', 'Comparator cohort'"
        )
        cohort_variable: str = Field(
            description="Variable name, e.g. 'COHORT', 'TRTGRP', 'EXPOSURE'"
        )
        values: str = Field(
            description="Possible values, e.g. '1=Treated, 2=Comparator' or 'Exposed/Unexposed'"
        )
        definition: str = Field(
            description="Operational definition: how to identify patients in this cohort. "
            "Include code lists, procedures, drugs, or algorithm details."
        )
        section_title: str
        explicit: ExplicitType
        confidence: float = Field(ge=0.0, le=1.0)

    cohorts: list[CohortExtraction]
    contradictions_found: bool
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst building a program specification.
Extract ALL cohort definitions from the protocol text.

Cohort definitions describe how eligible patients are divided into analysis groups.
These are NOT inclusion/exclusion criteria — they define sub-populations AFTER
eligibility is applied.

## What to look for:
- Treatment vs comparator cohorts (drug A vs drug B, exposed vs unexposed)
- Case vs control groups
- Index treatment cohorts (e.g. patients initiating drug X)
- Comparator cohorts (e.g. patients on standard of care)
- Sub-cohort definitions (e.g. by line of therapy, by biomarker status)
- Multiple analysis populations (ITT, per-protocol, safety)

## For each cohort, extract:
- cohort_label: Human-readable name (e.g. "Treatment cohort - Drug A")
- cohort_variable: Variable name (e.g. "COHORT", "TRTGRP", "EXPOSURE")
- values: Coded values (e.g. "1=Drug A, 2=Drug B")
- definition: Operational definition with codes, procedures, or algorithm details

IMPORTANT: Use the chain_of_thought field to reason about the protocol text BEFORE listing cohorts.
Think step by step — identify treatment arms, comparators, and analysis populations.

Rules:
- Extract EVERY distinct cohort or analysis population.
- Include the operational definition (how to programmatically identify each cohort).
- If the protocol mentions treatment arms, comparators, or analysis populations,
  those are cohorts.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- Return the exact quoted_text from the protocol, not a paraphrase.
- It is OK to return an empty list if no cohort definitions are found.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_cohort_definitions(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:
    """Extract cohort definitions from the protocol for StudyPop Section B."""

    queries = build_query_bank(
        "cohort definition treatment arm comparator exposed unexposed "
        "analysis population treatment group case control study arm "
        "line of therapy ITT per-protocol safety population",
        ta_pack, CONCEPT,
    )

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
            protocol_id=protocol_id, concept=CONCEPT,
            candidates=[], low_retrieval_signal=True,
            overall_confidence=0.0,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT)
    context = build_context(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=context,
        schema=CohortDefinitionExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    adj_extraction, adj_model, used_adjudicator = try_adjudicator(
        client, SYSTEM_PROMPT, context, CohortDefinitionExtraction,
        extraction, PROMPT_VERSION, "CohortDefinitionFinder",
    )
    if used_adjudicator:
        extraction, model_used = adj_extraction, adj_model

    pack = _build_cohort_pack(protocol_id, extraction, chunks, model_used, used_adjudicator)

    print(f"[CohortDefinitionFinder] Done. "
          f"{len(pack.candidates)} cohorts | "
          f"confidence={extraction.overall_confidence:.2f}")

    return pack


def _build_cohort_pack(
    protocol_id: str,
    extraction: CohortDefinitionExtraction,
    chunks: list[RetrievedChunk],
    model_used: str,
    adjudicator_used: bool = False,
) -> EvidencePack:
    """Build EvidencePack from CohortDefinitionExtraction."""
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    per_candidate = {}

    for coh in extraction.cohorts:
        matching = chunk_by_id.get(coh.chunk_id) if coh.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:cohort_definitions:{coh.chunk_id or ''}:{coh.quoted_text}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=coh.chunk_id,
            snippet=coh.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else coh.section_title,
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=coh.cohort_label,
            canonical_term="cohort_definitions",
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=coh.confidence,
            explicit=coh.explicit,
        ))

        per_candidate[candidate_id] = {
            "cohort_label": coh.cohort_label,
            "cohort_variable": coh.cohort_variable,
            "values": coh.values,
            "definition": coh.definition,
        }

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=CONCEPT,
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
    pack.concept_metadata = {"per_candidate": per_candidate}

    return pack
