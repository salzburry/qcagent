"""
Source data preparation issue finder.

Generates Data Prep Section D rows — source-specific preparation decisions
that must be made before building analytic files.

This uses two inputs:
  1. Retrieved protocol text (for study-specific requirements)
  2. Known data source limitations (from registry.py)

Issues include: missing variables, table linkage requirements, code mapping
decisions, date imputation rules, and derivation caveats.
"""

from __future__ import annotations
import hashlib
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning, get_section_priority
from ..data_sources.registry import SOURCE_AVAILABILITY, DEFINITIONS
from .base import build_context, compute_low_signal, try_adjudicator

CONCEPT = "source_data_prep"
FINDER_VERSION = "0.5.0"
PROMPT_VERSION = "0.5.0"


class SourceDataPrepExtraction(BaseModel):
    """Schema for source data preparation issue extraction."""

    chain_of_thought: str = Field(
        description="Think step by step about data preparation issues. "
        "Consider what the protocol requires vs what the data source provides, "
        "and identify gaps, mapping needs, and derivation challenges."
    )

    class PrepIssue(BaseModel):
        reasoning: str = Field(
            description="Why this action is appropriate"
        )
        chunk_id: Optional[str] = None
        quoted_text: str = Field(
            description="Protocol text that triggers this preparation issue, "
            "or 'N/A - source limitation' if purely source-driven"
        )
        source_table_variable: str = Field(
            description="Source table/variable affected, e.g. 'MEDICAL_CLAIMS.DX_CD'"
        )
        situation: str = Field(
            description="What situation requires resolution, e.g. "
            "'Race/ethnicity not available in MarketScan'"
        )
        action: str = Field(
            description="Recommended action, e.g. 'Set to Missing' or 'Derive from ICD codes'"
        )
        confidence: float = Field(ge=0.0, le=1.0)

    issues: list[PrepIssue]
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE data engineer building a program specification.
Identify ALL source data preparation issues for this study.

Source data preparation issues are decisions that must be resolved BEFORE
building analytic files. They typically involve:

## Categories of issues:
1. **Missing variables**: Variables required by the protocol but unavailable
   in the data source (e.g. ECOG not in claims data)
2. **Code mapping**: ICD/NDC/HCPCS codes that need mapping to study variables
3. **Date imputation**: Partial dates, month-only dates, missing day components
4. **Table linkage**: How to join source tables to derive the needed variables
5. **Unit conversion**: Lab values needing unit standardization
6. **Duplicate handling**: How to handle multiple records per patient per date
7. **Data quality**: Known data quality issues in the source

## Data source context:
{source_context}

IMPORTANT: Use the chain_of_thought field to reason about source data issues BEFORE listing them.
Think step by step — compare protocol requirements against known data source capabilities.

Rules:
- Focus on issues that a PROGRAMMER would need to resolve.
- Be specific about source tables and variables when possible.
- For each issue, recommend a concrete action (not just "investigate").
- Flag issues where the protocol requires something the data source cannot provide.
- IMPORTANT: Return the chunk_id from each chunk header for provenance.
- It is OK to return an empty list if no issues are found.
- Respond ONLY with valid JSON matching the schema exactly."""


def find_source_data_prep(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Extract source data preparation issues from protocol + source knowledge."""

    queries = build_query_bank(
        "data source database table variable derivation algorithm "
        "code list ICD NDC HCPCS mapping imputation missing data "
        "data quality linkage enrollment claims EHR lab results",
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
        # Even without retrieved chunks, we can generate source-limitation issues
        pack = _build_source_limitation_pack(protocol_id, data_source)
        if pack.candidates:
            return pack
        return EvidencePack(
            protocol_id=protocol_id, concept=CONCEPT,
            candidates=[], low_retrieval_signal=True,
            overall_confidence=0.0,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        )

    # Build source context from registry knowledge
    source_context = _build_source_context(data_source)

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT)
    context = build_context(chunks, ta_warning, protocol_id)

    system = SYSTEM_PROMPT.format(source_context=source_context)

    result = client.extract(
        system_prompt=system,
        user_prompt=context,
        schema=SourceDataPrepExtraction,
        use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    adj_extraction, adj_model, used_adjudicator = try_adjudicator(
        client, system, context, SourceDataPrepExtraction,
        extraction, PROMPT_VERSION, "SourceDataPrepFinder",
    )
    if used_adjudicator:
        extraction, model_used = adj_extraction, adj_model

    pack = _build_prep_pack(protocol_id, extraction, chunks, model_used,
                            used_adjudicator, data_source)

    print(f"[SourceDataPrepFinder] Done. "
          f"{len(pack.candidates)} issues | "
          f"confidence={extraction.overall_confidence:.2f} | "
          f"source={data_source}")

    return pack


def _build_source_context(data_source: str) -> str:
    """Build a context string describing the data source capabilities."""
    if data_source == "generic":
        return "Data source: Generic/unknown. Flag any issues that might vary by source."

    avail = SOURCE_AVAILABILITY.get(data_source, SOURCE_AVAILABILITY["generic"])
    defs = DEFINITIONS.get(data_source, {})

    parts = [f"Data source: {data_source}"]

    # Availability summary
    unavailable = [cat for cat, available in avail.items() if not available]
    if unavailable:
        parts.append(f"UNAVAILABLE categories: {', '.join(unavailable)}")

    # Known "Not available" / "Missing" derivations
    missing_vars = [
        f"  - {var}: {defn}"
        for var, defn in defs.items()
        if "not available" in defn.lower() or "set to missing" in defn.lower()
    ]
    if missing_vars:
        parts.append("Known missing/unavailable variables:\n" + "\n".join(missing_vars))

    # Known derivation complexity
    complex_vars = [
        f"  - {var}: {defn}"
        for var, defn in defs.items()
        if any(kw in defn.lower() for kw in ["derive", "calculate", "algorithm", "infer", "nlp"])
    ]
    if complex_vars:
        parts.append("Variables requiring complex derivation:\n" + "\n".join(complex_vars[:10]))

    return "\n".join(parts)


def _build_source_limitation_pack(
    protocol_id: str,
    data_source: str,
) -> EvidencePack:
    """Generate source-limitation issues without LLM, from registry knowledge alone."""
    if data_source == "generic":
        return EvidencePack(
            protocol_id=protocol_id, concept=CONCEPT,
            candidates=[], overall_confidence=0.5,
            finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
            model_used="registry_only",
        )

    avail = SOURCE_AVAILABILITY.get(data_source, {})
    defs = DEFINITIONS.get(data_source, {})

    candidates = []
    per_candidate = {}

    # Generate issues for unavailable categories
    unavailable_cats = [cat for cat, av in avail.items() if not av]
    for cat in unavailable_cats:
        cid = hashlib.sha256(
            f"{protocol_id}:source_data_prep:unavailable:{cat}:{data_source}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=cid,
            snippet=f"{cat} variables not available in {data_source}",
            page=None,
            source_type="narrative",
            sponsor_term=cat,
            canonical_term="source_data_prep",
            llm_confidence=0.9,
            explicit="explicit",
        ))
        per_candidate[cid] = {
            "source_table_variable": f"{data_source} ({cat})",
            "situation": f"{cat} category is not available in {data_source} data",
            "action": f"Set {cat} variables to Missing or source from linked data if available",
            "reasoning": f"{data_source} does not capture {cat} data natively",
        }

    # Generate issues for specific "Not available" / "Missing" variables
    for var, defn in defs.items():
        if "not available" in defn.lower() or "set to missing" in defn.lower():
            cid = hashlib.sha256(
                f"{protocol_id}:source_data_prep:missing_var:{var}:{data_source}".encode()
            ).hexdigest()[:12]

            candidates.append(EvidenceCandidate(
                candidate_id=cid,
                snippet=defn,
                page=None,
                source_type="narrative",
                sponsor_term=var,
                canonical_term="source_data_prep",
                llm_confidence=0.9,
                explicit="explicit",
            ))
            per_candidate[cid] = {
                "source_table_variable": var,
                "situation": f"{var}: {defn}",
                "action": "Set to Missing unless linked data available",
                "reasoning": f"Source limitation of {data_source}",
            }

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=CONCEPT,
        candidates=candidates,
        overall_confidence=0.8 if candidates else 0.0,
        low_retrieval_signal=True,
        requires_human_selection=True,
        finder_version=FINDER_VERSION,
        model_used="registry_only",
        prompt_version=PROMPT_VERSION,
    )
    pack.concept_metadata = {"per_candidate": per_candidate}

    return pack


def _build_prep_pack(
    protocol_id: str,
    extraction: SourceDataPrepExtraction,
    chunks: list[RetrievedChunk],
    model_used: str,
    adjudicator_used: bool,
    data_source: str,
) -> EvidencePack:
    """Build EvidencePack from SourceDataPrepExtraction."""
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}

    candidates = []
    per_candidate = {}

    for issue in extraction.issues:
        matching = chunk_by_id.get(issue.chunk_id) if issue.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:source_data_prep:{issue.chunk_id or ''}:{issue.source_table_variable}:{issue.situation}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id,
            chunk_id=issue.chunk_id,
            snippet=issue.quoted_text,
            page=matching.page if matching else None,
            section_title=matching.heading if matching else "",
            source_type=matching.source_type if matching else "narrative",
            sponsor_term=issue.source_table_variable,
            canonical_term="source_data_prep",
            retrieval_score=matching.retrieval_score if matching else None,
            rerank_score=matching.rerank_score if matching else None,
            llm_confidence=issue.confidence,
            explicit="explicit",
        ))

        per_candidate[candidate_id] = {
            "source_table_variable": issue.source_table_variable,
            "situation": issue.situation,
            "action": issue.action,
            "reasoning": issue.reasoning,
        }

    # Also add registry-only limitations not covered by LLM
    reg_pack = _build_source_limitation_pack(protocol_id, data_source)
    if reg_pack.candidates:
        reg_meta = (reg_pack.concept_metadata or {}).get("per_candidate", {})
        existing_vars = {pc.get("source_table_variable", "") for pc in per_candidate.values()}
        for rc in reg_pack.candidates:
            rm = reg_meta.get(rc.candidate_id, {})
            if rm.get("source_table_variable", "") not in existing_vars:
                candidates.append(rc)
                per_candidate[rc.candidate_id] = rm

    pack = EvidencePack(
        protocol_id=protocol_id,
        concept=CONCEPT,
        candidates=candidates,
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
