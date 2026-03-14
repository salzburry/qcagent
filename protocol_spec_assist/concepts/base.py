"""
Shared utilities for concept finders.

Extracts the common boilerplate that was duplicated across all template-based
concept finders (demographics, clinical_characteristics, biomarkers,
lab_variables, treatment_variables) and the candidate-based finders
(index_date, endpoints, eligibility, study_design).
"""

from __future__ import annotations
import hashlib
import logging
from typing import Optional

from ..schemas.evidence import EvidencePack, EvidenceCandidate
from ..retrieval.search import RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..data_sources.registry import resolve_static_template
from .evidence_auditor import audit_candidates, format_candidates_for_audit, AuditResult
from .evidence_merger import merge_candidates, MergeResult

logger = logging.getLogger(__name__)


# ── Common constants ─────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.65
LOW_RETRIEVAL_THRESHOLD = 3
RERANK_SCORE_FLOOR = 0.2

# Concepts that get author→auditor→merger treatment
# (high-value, error-prone, ambiguous temporal/definitional semantics)
AUDITED_CONCEPTS = {
    "index_date",
    "follow_up_end",
    "study_period",
    "censoring_rules",
    "cohort_definitions",
    "primary_endpoint",
    "eligibility_inclusion",
    "eligibility_exclusion",
}


# ── Context building ────────────────────────────────────────────────────────

def build_context(
    chunks: list[RetrievedChunk],
    ta_warning: Optional[str],
    protocol_id: str,
) -> str:
    """Build the user prompt context from retrieved chunks.

    Used identically by all concept finders.
    """
    parts = [f"Protocol ID: {protocol_id}"]
    if ta_warning:
        parts.append(f"\nTA PACK WARNING: {ta_warning}\n")
    for chunk in chunks:
        parts.append(
            f"[chunk_id={chunk.chunk_id} | Section: {chunk.heading} | "
            f"Type: {chunk.source_type} | Page: {chunk.page} | "
            f"Score: {chunk.score:.2f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def compute_low_signal(chunks: list[RetrievedChunk]) -> bool:
    """Determine if retrieval signal is too weak for confident extraction."""
    if len(chunks) < LOW_RETRIEVAL_THRESHOLD:
        return True
    if chunks and chunks[0].rerank_score is not None:
        return chunks[0].rerank_score < RERANK_SCORE_FLOOR
    return False


# ── Adjudicator pass ─────────────────────────────────────────────────────────

def try_adjudicator(
    client: LocalModelClient,
    system_prompt: str,
    user_prompt: str,
    schema,
    extraction,
    prompt_version: str,
    finder_name: str,
):
    """Confidence router: try adjudicator model if confidence is low or contradictions found.

    Returns (extraction, model_used, used_adjudicator).
    """
    if extraction.overall_confidence >= CONFIDENCE_THRESHOLD and not extraction.contradictions_found:
        return extraction, None, False

    try:
        result = client.extract(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            use_adjudicator=True,
            prompt_version=prompt_version,
        )
        return result.parsed, result.model_used, True
    except Exception as e:
        print(f"[{finder_name}] Adjudicator unavailable ({e}), keeping first-pass result.")
        return extraction, None, False


# ── Author → Auditor → Merger ─────────────────────────────────────────────────

def audit_and_merge(
    client: LocalModelClient,
    candidates: list[EvidenceCandidate],
    extraction_candidates: list,
    chunks_text: str,
    concept: str,
) -> tuple[list[EvidenceCandidate], bool, list[str]]:
    """Run evidence audit and deterministic merge on candidates.

    Only called for AUDITED_CONCEPTS. Template-based finders skip this.

    Args:
        client: Model client
        candidates: EvidenceCandidate list (already built from extraction)
        extraction_candidates: Raw extraction candidate objects (for formatting)
        chunks_text: Original chunk text for the auditor to verify against
        concept: Concept name

    Returns:
        (final_candidates, has_contradictions, auditor_notes)
    """
    if concept not in AUDITED_CONCEPTS:
        return candidates, False, []

    if not candidates:
        return candidates, False, []

    # Format candidates for the auditor
    candidates_text = format_candidates_for_audit(extraction_candidates, concept)

    # Run auditor
    audit = audit_candidates(client, candidates_text, chunks_text, concept)

    # Deterministic merge
    result = merge_candidates(candidates, audit, concept)

    # Final list = accepted + repaired (rejected are dropped)
    final = result.accepted + result.repaired

    return final, result.has_contradictions, result.auditor_notes


# ── Template merge ───────────────────────────────────────────────────────────

def merge_with_static_template(extraction_variables, static_template, var_class):
    """Merge LLM-extracted variables with a static template.

    Strategy: start from the static template (guaranteed correct tab placement).
    If the LLM found a matching variable (by variable_name), use the LLM's
    definition/values/notes (protocol-specific). Otherwise keep the static
    default and mark as 'inferred'.

    Any extra variables the LLM found that aren't in the template are returned
    separately as unmapped.
    """
    if not static_template:
        # No template — return only LLM-found variables, nothing unmapped
        return list(extraction_variables), []

    llm_by_name = {v.variable_name.upper(): v for v in extraction_variables}

    merged = []
    used_names = set()

    for tmpl in static_template:
        name = tmpl["variable_name"].upper()
        used_names.add(name)
        if name in llm_by_name:
            merged.append(llm_by_name[name])
        else:
            merged.append(var_class(
                chunk_id=None,
                time_period=tmpl["time_period"],
                variable_name=tmpl["variable_name"],
                label=tmpl["label"],
                values=tmpl["values"],
                definition=tmpl["definition"],
                code_lists_group=tmpl.get("code_lists_group", ""),
                additional_notes=tmpl.get("additional_notes", ""),
                sponsor_term=None,
                explicit="inferred",
                confidence=0.5,
                reasoning="Static template default — not confirmed in protocol text",
            ))

    unmapped = [v for v in extraction_variables if v.variable_name.upper() not in used_names]
    return merged, unmapped


def build_static_only_pack(
    protocol_id: str,
    concept: str,
    static_template: list[dict],
    data_source: str,
    finder_version: str,
    prompt_version: str,
) -> EvidencePack:
    """Build an EvidencePack from static template alone (no LLM, no retrieval)."""
    resolved = resolve_static_template(static_template, data_source, concept)
    candidates = []
    per_candidate_meta = {}
    for tmpl in resolved:
        candidate_id = hashlib.sha256(
            f"{protocol_id}:{concept}:static:{tmpl['variable_name']}:{tmpl['definition']}".encode()
        ).hexdigest()[:12]
        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id, chunk_id=None,
            snippet=tmpl["definition"], page=None, section_title="",
            source_type="narrative",
            sponsor_term=tmpl["variable_name"],
            canonical_term=tmpl["variable_name"],
            llm_confidence=0.5, explicit="inferred",
        ))
        per_candidate_meta[candidate_id] = {
            "time_period": tmpl["time_period"],
            "variable_name": tmpl["variable_name"],
            "label": tmpl["label"],
            "values": tmpl["values"],
            "code_lists_group": tmpl.get("code_lists_group", ""),
            "additional_notes": tmpl.get("additional_notes", ""),
        }
    return EvidencePack(
        protocol_id=protocol_id, concept=concept, candidates=candidates,
        overall_confidence=0.5,
        concept_metadata={"per_candidate": per_candidate_meta},
        low_retrieval_signal=True, requires_human_selection=True,
        finder_version=finder_version, prompt_version=prompt_version,
        model_used="static_template",
    )


# ── Template-based finder workflow ───────────────────────────────────────────

def run_template_finder(
    protocol_id: str,
    index,
    client: LocalModelClient,
    ta_pack,
    data_source: str,
    *,
    concept: str,
    base_query: str,
    system_prompt: str,
    extraction_schema,
    static_template: list[dict],
    finder_version: str,
    prompt_version: str,
    finder_name: str,
) -> EvidencePack:
    """Full template-based concept finder workflow.

    Shared by demographics, clinical_characteristics, biomarkers,
    lab_variables, and treatment_variables.
    """
    from ..ta_packs.loader import build_query_bank, get_hotspot_warning, get_section_priority

    # Step 1: Build query bank
    queries = build_query_bank(base_query, ta_pack, concept)

    # Step 2: Hybrid retrieval
    priority_sections = get_section_priority(ta_pack, concept)
    chunks = index.search(
        query=queries[0], protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25, top_k_rerank=10,
        include_tables=True, priority_sections=priority_sections,
    )

    if not chunks:
        if not static_template:
            return EvidencePack(
                protocol_id=protocol_id, concept=concept, candidates=[],
                overall_confidence=0.0, low_retrieval_signal=True,
                requires_human_selection=True,
                finder_version=finder_version, prompt_version=prompt_version,
                model_used="none",
            )
        return build_static_only_pack(
            protocol_id, concept, static_template, data_source,
            finder_version, prompt_version,
        )

    # Step 3: TA warning + context
    ta_warning = get_hotspot_warning(ta_pack, concept)
    user_prompt = build_context(chunks, ta_warning, protocol_id)

    # Step 4: LLM extraction
    result = client.extract(
        system_prompt=system_prompt, user_prompt=user_prompt,
        schema=extraction_schema, use_adjudicator=False,
        prompt_version=prompt_version,
    )
    extraction, model_used = result.parsed, result.model_used

    # Step 5: Confidence router
    adj_extraction, adj_model, used_adjudicator = try_adjudicator(
        client, system_prompt, user_prompt, extraction_schema,
        extraction, prompt_version, finder_name,
    )
    if used_adjudicator:
        extraction, model_used = adj_extraction, adj_model

    # Step 6: Merge with source-resolved static template
    resolved_template = resolve_static_template(static_template, data_source, concept)
    var_class = extraction_schema.VariableExtraction
    merged_variables, unmapped_variables = merge_with_static_template(
        extraction.variables, resolved_template, var_class,
    )

    # Step 7: Build EvidencePack
    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}
    candidates = []
    per_candidate_meta = {}

    for v in merged_variables:
        matching_chunk = chunk_by_id.get(v.chunk_id) if v.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:{concept}:{v.chunk_id or ''}:{v.variable_name}:{v.definition}".encode()
        ).hexdigest()[:12]

        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id, chunk_id=v.chunk_id,
            snippet=v.definition,
            page=matching_chunk.page if matching_chunk else None,
            section_title=matching_chunk.heading if matching_chunk else "",
            source_type=matching_chunk.source_type if matching_chunk else "narrative",
            sponsor_term=v.sponsor_term or v.variable_name,
            canonical_term=v.variable_name,
            retrieval_score=matching_chunk.retrieval_score if matching_chunk else None,
            rerank_score=matching_chunk.rerank_score if matching_chunk else None,
            llm_confidence=v.confidence, explicit=v.explicit,
        ))
        per_candidate_meta[candidate_id] = {
            "time_period": v.time_period,
            "variable_name": v.variable_name,
            "label": v.label,
            "values": v.values,
            "code_lists_group": v.code_lists_group,
            "additional_notes": v.additional_notes,
        }

    pack = EvidencePack(
        protocol_id=protocol_id, concept=concept, candidates=candidates,
        contradictions_found=extraction.contradictions_found,
        contradiction_detail=extraction.contradiction_detail,
        overall_confidence=extraction.overall_confidence,
        concept_metadata={
            "per_candidate": per_candidate_meta,
            "unmapped_variables": [
                {"variable_name": v.variable_name, "label": v.label,
                 "definition": v.definition, "confidence": v.confidence}
                for v in unmapped_variables
            ],
        },
        low_retrieval_signal=compute_low_signal(chunks),
        adjudicator_used=used_adjudicator,
        requires_human_selection=True,
        finder_version=finder_version, model_used=model_used,
        prompt_version=prompt_version,
    )

    print(f"[{finder_name}] Done. {len(candidates)} variables | "
          f"confidence={extraction.overall_confidence:.2f}")
    return pack
