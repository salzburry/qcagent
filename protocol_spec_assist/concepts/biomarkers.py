"""
Biomarker variables concept finder.
Extracts biomarker variable definitions from protocol text.

Variables typically found: BCL2, BCL6, MYC, CD10, CD19, CD20, CD3, CD5,
CD30, CD45, IRF4, TP53, D17P expression/rearrangement status, etc.
"""

from __future__ import annotations
import hashlib
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning, get_section_priority
from ..data_sources.registry import resolve_static_template

CONCEPT = "biomarkers"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"
CONFIDENCE_THRESHOLD = 0.65

# ── Static variable template ─────────────────────────────────────────────────
# Each biomarker has up to 5 sub-variables: status, numeric, closest-date, closest-status, closest-numeric.
# This is the exhaustive oncology (LBCL) default. Protocol-specific markers are added by LLM.

def _biomarker_group(prefix, marker_name, source_field):
    """Generate the standard 5-variable group for a biomarker."""
    return [
        {"time_period": "PRE_BIO", "variable_name": prefix, "label": f"{marker_name} Expression or Rearrangement Status", "values": "Positive; Negative; Unknown; Missing", "definition": f"molecular_marker where molecular_marker_name='{source_field}'; map interpretation to Positive/Negative/Unknown; Baseline Ever Status (any positive record in PRE_BIO → Positive)", "code_lists_group": "", "additional_notes": "Baseline Ever Status: if any record is Positive, overall is Positive"},
        {"time_period": "PRE_BIO", "variable_name": f"{prefix}N", "label": f"{marker_name} Status (N)", "values": "1; 2; 98; 99", "definition": f"Numeric code for {prefix}: 1=Positive, 2=Negative, 98=Unknown, 99=Missing", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "PRE_BIO", "variable_name": f"{prefix}COT", "label": f"{marker_name} Closest Test Date", "values": "Date", "definition": f"Date of {source_field} record closest to INDEX date within PRE_BIO period", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "PRE_BIO", "variable_name": f"{prefix}C", "label": f"{marker_name} Closest Status", "values": "Positive; Negative; Unknown; Missing", "definition": f"Status from {source_field} record closest to INDEX date (not ever-status)", "code_lists_group": "", "additional_notes": "Single closest record, not aggregated"},
        {"time_period": "PRE_BIO", "variable_name": f"{prefix}CN", "label": f"{marker_name} Closest Status (N)", "values": "1; 2; 98; 99", "definition": f"Numeric code for {prefix}C: 1=Positive, 2=Negative, 98=Unknown, 99=Missing", "code_lists_group": "", "additional_notes": ""},
    ]

STATIC_TEMPLATE = [
    *_biomarker_group("BCL2", "BCL2", "bcl2"),
    *_biomarker_group("BCL6", "BCL6", "bcl6"),
    *_biomarker_group("MYC", "MYC", "myc"),
    *_biomarker_group("CD10", "CD10", "cd10"),
    *_biomarker_group("CD19", "CD19", "cd19"),
    *_biomarker_group("CD20", "CD20", "cd20"),
    *_biomarker_group("CD3", "CD3", "cd3"),
    *_biomarker_group("CD5", "CD5", "cd5"),
    *_biomarker_group("CD30", "CD30", "cd30"),
    *_biomarker_group("CD45", "CD45", "cd45"),
    *_biomarker_group("IRF4", "IRF4/MUM1", "irf4_mum1"),
    *_biomarker_group("TP53", "TP53", "tp53"),
    *_biomarker_group("D17P", "Del(17p)", "del_17p"),
    *_biomarker_group("PDL1", "PD-L1", "pd_l1"),
    # Double/triple hit composite
    {"time_period": "PRE_BIO", "variable_name": "DBLHIT", "label": "Double Hit Lymphoma", "values": "Yes; No; Unknown; Missing", "definition": "MYC rearrangement + (BCL2 OR BCL6 rearrangement); requires both FISH results", "code_lists_group": "", "additional_notes": "Derived from MYC + BCL2/BCL6 rearrangement status"},
    {"time_period": "PRE_BIO", "variable_name": "TRPLHIT", "label": "Triple Hit Lymphoma", "values": "Yes; No; Unknown; Missing", "definition": "MYC + BCL2 + BCL6 all rearranged; requires all three FISH results", "code_lists_group": "", "additional_notes": "Derived from MYC + BCL2 + BCL6 rearrangement status"},
]


class BiomarkersExtraction(BaseModel):
    class VariableExtraction(BaseModel):
        chunk_id: Optional[str] = Field(default=None)
        time_period: str = Field(description="e.g. PRE_BIO, PRE_INT, ASSESS_BIO")
        variable_name: str = Field(description="e.g. BCL2, BCL2N, BCL2COT, BCL2C, CD10, MYC")
        label: str = Field(description="e.g. 'BCL2 Expression or Rearrangement Status'")
        values: str = Field(description="e.g. 'Positive; Negative; Unknown; Missing', '1,2,3,4', 'Date'")
        definition: str = Field(description="Derivation logic: how to determine status from data source records")
        code_lists_group: str = Field(default="")
        additional_notes: str = Field(default="", description="Method notes (e.g. 'Baseline Ever Status'), assessment window, closest date logic")
        sponsor_term: Optional[str] = None
        explicit: ExplicitType = "explicit"
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str = ""

    variables: list[VariableExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst specializing in programming specifications.
Your task is to identify BIOMARKER VARIABLES defined in the protocol text.

Biomarker variables typically include expression and rearrangement status for molecular markers:
- BCL2, BCL6, MYC (double/triple hit lymphoma markers)
- CD markers: CD10, CD19, CD20, CD3, CD5, CD30, CD45
- IRF4/MUM1, TP53 mutation/deletion status
- D17P deletion status
- PD-L1, EGFR, ALK, BRAF, HER2 and other oncology markers

For each biomarker variable, there are often 4-5 related variables:
1. Status variable (e.g. BCL2) — Positive/Negative/Unknown/Missing
2. Numeric coded version (e.g. BCL2N) — 1,2,3,4
3. Closest date variable (e.g. BCL2COT/BCL2DT) — Date of assessment
4. Closest status variable (e.g. BCL2C) — Status from closest test to index
5. Closest numeric version (e.g. BCL2CN) — Numeric of closest status

Provide the derivation logic for each variable: how to look up records where
molecular_marker matches, how to determine positive/negative from interpretation fields,
how to select the closest record to index date, assessment time period, etc.

Rules:
- Extract ONLY biomarker variables, NOT lab values or clinical characteristics.
- Include all related sub-variables (status, numeric, date, closest variants).
- Describe the assessment time period (e.g. PRE_BIO for baseline biomarker period).
- Mark inferred variables as "inferred".
- Return chunk_id for provenance.
- Respond ONLY with valid JSON."""


def _build_user_prompt(chunks, ta_warning, protocol_id):
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[chunk_id={chunk.chunk_id} | Section: {chunk.heading} | "
            f"Type: {chunk.source_type} | Page: {chunk.page} | "
            f"Score: {chunk.score:.2f}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)
    warning_block = f"\nTA PACK WARNING: {ta_warning}\n" if ta_warning else ""
    return f"""Protocol ID: {protocol_id}
{warning_block}
Extract all biomarker variable definitions from the following protocol text.

{context}"""


def _merge_with_static_template(extraction_variables, static_template):
    """Merge LLM-extracted variables with the static template."""
    llm_by_name = {v.variable_name.upper(): v for v in extraction_variables}
    VarClass = BiomarkersExtraction.VariableExtraction
    merged = []
    used_names = set()
    for tmpl in static_template:
        name = tmpl["variable_name"].upper()
        used_names.add(name)
        if name in llm_by_name:
            merged.append(llm_by_name[name])
        else:
            merged.append(VarClass(
                chunk_id=None, time_period=tmpl["time_period"],
                variable_name=tmpl["variable_name"], label=tmpl["label"],
                values=tmpl["values"], definition=tmpl["definition"],
                code_lists_group=tmpl.get("code_lists_group", ""),
                additional_notes=tmpl.get("additional_notes", ""),
                sponsor_term=None, explicit="inferred", confidence=0.5,
                reasoning="Static template default — not confirmed in protocol text",
            ))
    for v in extraction_variables:
        if v.variable_name.upper() not in used_names:
            merged.append(v)
    return merged


def _build_static_only_pack(protocol_id: str, data_source: str = "generic") -> EvidencePack:
    """Build an EvidencePack from static template alone."""
    resolved = resolve_static_template(STATIC_TEMPLATE, data_source, CONCEPT)
    candidates = []
    per_candidate_meta = {}
    for tmpl in resolved:
        candidate_id = hashlib.sha256(
            f"{protocol_id}:{CONCEPT}:static:{tmpl['variable_name']}:{tmpl['definition']}".encode()
        ).hexdigest()[:12]
        candidates.append(EvidenceCandidate(
            candidate_id=candidate_id, chunk_id=None,
            snippet=tmpl["definition"], page=None, section_title="",
            source_type="narrative", sponsor_term=tmpl["variable_name"],
            canonical_term=tmpl["variable_name"],
            llm_confidence=0.5, explicit="inferred",
        ))
        per_candidate_meta[candidate_id] = {
            "time_period": tmpl["time_period"], "variable_name": tmpl["variable_name"],
            "label": tmpl["label"], "values": tmpl["values"],
            "code_lists_group": tmpl.get("code_lists_group", ""),
            "additional_notes": tmpl.get("additional_notes", ""),
        }
    return EvidencePack(
        protocol_id=protocol_id, concept=CONCEPT, candidates=candidates,
        overall_confidence=0.5,
        concept_metadata={"per_candidate": per_candidate_meta},
        low_retrieval_signal=True, requires_human_selection=True,
        finder_version=FINDER_VERSION, prompt_version=PROMPT_VERSION,
        model_used="static_template",
    )


def find_biomarkers(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the biomarkers concept finder workflow."""

    queries = build_query_bank(
        "biomarker molecular marker expression rearrangement BCL2 BCL6 MYC CD10 CD19 CD20 "
        "PD-L1 EGFR ALK mutation deletion IHC FISH immunohistochemistry",
        ta_pack, CONCEPT,
    )

    priority_sections = get_section_priority(ta_pack, CONCEPT)
    chunks = index.search(
        query=queries[0], protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25, top_k_rerank=10,
        include_tables=True, priority_sections=priority_sections,
    )

    if not chunks:
        return _build_static_only_pack(protocol_id, data_source)

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT)
    user_prompt = _build_user_prompt(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
        schema=BiomarkersExtraction, use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                schema=BiomarkersExtraction, use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception:
            pass

    # Merge with static template
    resolved_template = resolve_static_template(STATIC_TEMPLATE, data_source, CONCEPT)
    merged_variables = _merge_with_static_template(extraction.variables, resolved_template)

    chunk_by_id = {ch.chunk_id: ch for ch in chunks if ch.chunk_id}
    candidates = []
    per_candidate_meta = {}

    for v in merged_variables:
        matching_chunk = chunk_by_id.get(v.chunk_id) if v.chunk_id else None
        candidate_id = hashlib.sha256(
            f"{protocol_id}:{CONCEPT}:{v.chunk_id or ''}:{v.variable_name}:{v.definition}".encode()
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
            "time_period": v.time_period, "variable_name": v.variable_name,
            "label": v.label, "values": v.values,
            "code_lists_group": v.code_lists_group,
            "additional_notes": v.additional_notes,
        }

    pack = EvidencePack(
        protocol_id=protocol_id, concept=CONCEPT, candidates=candidates,
        contradictions_found=extraction.contradictions_found,
        contradiction_detail=extraction.contradiction_detail,
        overall_confidence=extraction.overall_confidence,
        concept_metadata={"per_candidate": per_candidate_meta},
        low_retrieval_signal=len(chunks) < 3,
        adjudicator_used=used_adjudicator,
        requires_human_selection=True,
        finder_version=FINDER_VERSION, model_used=model_used,
        prompt_version=PROMPT_VERSION,
    )

    print(f"[BiomarkersFinder] Done. {len(candidates)} variables | "
          f"confidence={extraction.overall_confidence:.2f}")
    return pack
