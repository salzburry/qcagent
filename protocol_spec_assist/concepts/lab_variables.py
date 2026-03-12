"""
Laboratory variables concept finder.
Extracts lab test variable definitions from protocol text.

Variables typically found: neutrophils (ANC), platelets, hemoglobin,
creatinine, bilirubin, AST, ALT, alkaline phosphatase, albumin,
LDH, lymphocytes, etc. — each with Present/N/Date/Value sub-variables.
"""

from __future__ import annotations
import hashlib
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning, get_section_priority

CONCEPT = "lab_variables"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"
CONFIDENCE_THRESHOLD = 0.65

# ── Static variable template ─────────────────────────────────────────────────
# Each lab test follows the 4-variable pattern: presence, presence-N, date, value.

def _lab_group(prefix, lab_name, source_field, unit="", notes=""):
    """Generate the standard 4-variable group for a lab test."""
    return [
        {"time_period": "PRE_INT", "variable_name": f"LAB{prefix}", "label": f"{lab_name} Value Present", "values": "Present; Unknown; Missing", "definition": f"lab_test where lab_name='{source_field}' AND assessed='true'; Present if at least one non-missing quantitative_result in PRE_INT period", "code_lists_group": "", "additional_notes": notes},
        {"time_period": "PRE_INT", "variable_name": f"LAB{prefix}N", "label": f"{lab_name} Value Present (N)", "values": "1; 98; 99", "definition": f"Numeric code for LAB{prefix}: 1=Present, 98=Unknown, 99=Missing", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "PRE_INT", "variable_name": f"LAB{prefix}DT", "label": f"{lab_name} Date", "values": "Date", "definition": f"Date of {source_field} record closest to INDEX date within PRE_INT; use timedelta fields to compute date", "code_lists_group": "", "additional_notes": "Closest to index date"},
        {"time_period": "PRE_INT", "variable_name": f"LAB{prefix}V", "label": f"{lab_name} Value{', ' + unit if unit else ''}", "values": "numeric", "definition": f"quantitative_result from {source_field} record closest to INDEX date within PRE_INT; standardize units{' to ' + unit if unit else ''}", "code_lists_group": "", "additional_notes": f"Unit: {unit}" if unit else ""},
    ]

STATIC_TEMPLATE = [
    *_lab_group("NEU", "Neutrophil/ANC", "absolute_neutrophil_count_anc", "10^9/L", "Use standardize_labs() or COTA R package for unit conversion"),
    *_lab_group("PLA", "Platelet", "platelet_count", "10^9/L"),
    *_lab_group("HEM", "Hemoglobin", "hemoglobin", "g/dL"),
    *_lab_group("CRE", "Serum Creatinine", "creatinine", "mg/dL"),
    *_lab_group("BIL", "Total Bilirubin", "total_bilirubin", "mg/dL"),
    *_lab_group("AST", "AST", "aspartate_aminotransferase_ast", "U/L"),
    *_lab_group("ALT", "ALT", "alanine_aminotransferase_alt", "U/L"),
    *_lab_group("ALP", "Alkaline Phosphatase", "alkaline_phosphatase", "U/L"),
    *_lab_group("ALB", "Albumin", "albumin", "g/dL"),
    *_lab_group("LDH", "LDH", "lactate_dehydrogenase", "U/L", "Elevated LDH is a prognostic factor; note ULN reference range"),
    *_lab_group("LYM", "Lymphocyte/ALC", "absolute_lymphocyte_count_alc", "10^9/L"),
    *_lab_group("WBC", "White Blood Cell", "white_blood_cell_count", "10^9/L"),
]


class LabVariablesExtraction(BaseModel):
    class VariableExtraction(BaseModel):
        chunk_id: Optional[str] = Field(default=None)
        time_period: str = Field(description="e.g. PRE_INT (baseline labs before index)")
        variable_name: str = Field(description="e.g. LABNEU, LABNEUV, LABNEUDT, LABPLA, LABHEMV")
        label: str = Field(description="e.g. 'Neutrophil Value Present', 'Hemoglobin value, g/dL'")
        values: str = Field(description="e.g. 'Present; Unknown; Missing', 'Numeric', 'Date', '1,2,3'")
        definition: str = Field(description="Derivation: which lab_test/lab_name to query, unit standardization, closest-to-index logic")
        code_lists_group: str = Field(default="")
        additional_notes: str = Field(default="", description="e.g. COTA R package function references, unit conversion notes")
        sponsor_term: Optional[str] = None
        explicit: ExplicitType = "explicit"
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str = ""

    variables: list[VariableExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst specializing in programming specifications.
Your task is to identify LABORATORY VARIABLE definitions from protocol text.

Lab variables typically follow a 4-variable pattern per lab test:
1. Presence flag (e.g. LABNEU) — 'Present'/'Unknown'/'Missing'
2. Presence numeric (e.g. LABNEUN) — 1,98,99
3. Date variable (e.g. LABNEUDT) — date of record closest to index
4. Value variable (e.g. LABNEUV) — standardized test value closest to index

Common lab tests in RWE studies:
- Neutrophils / ANC (LABNEU) — absolute_neutrophil_count_anc
- Platelets (LABPLA) — platelet_count
- Hemoglobin (LABHEM) — hemoglobin
- Serum creatinine (LABCRE) — creatinine
- Bilirubin (LABBIL) — total_bilirubin
- AST (LABAST) — aspartate_aminotransferase_ast
- ALT (LABALT) — alanine_aminotransferase_alt
- Alkaline phosphatase (LABALP) — alkaline_phosphatase
- Albumin (LABALB) — albumin
- LDH (LABLDH) — lactate_dehydrogenase
- Lymphocytes (LABLYM) — absolute_lymphocyte_count_alc
- WBC (LABWBC) — white_blood_cell_count

For each variable, define:
- How to determine presence (assessed='true' AND at least one of timedelta fields not missing AND quantitative_result not missing)
- How to select the closest record to index date
- Unit standardization (which COTA R function / target units)
- Whether to take lowest or highest value when multiple records on same date

Rules:
- Extract ONLY lab variables, NOT biomarkers or clinical characteristics.
- Include all 4 sub-variables per lab test when protocol defines them.
- Note unit standardization requirements.
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
Extract all laboratory variable definitions from the following protocol text.

{context}"""


def _merge_with_static_template(extraction_variables, static_template):
    """Merge LLM-extracted variables with the static template."""
    llm_by_name = {v.variable_name.upper(): v for v in extraction_variables}
    VarClass = LabVariablesExtraction.VariableExtraction
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


def _build_static_only_pack(protocol_id: str) -> EvidencePack:
    """Build an EvidencePack from static template alone."""
    candidates = []
    per_candidate_meta = {}
    for tmpl in STATIC_TEMPLATE:
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


def find_lab_variables(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:
    """Run the lab variables concept finder workflow."""

    queries = build_query_bank(
        "laboratory lab values baseline labs neutrophil platelet hemoglobin creatinine "
        "bilirubin AST ALT alkaline phosphatase albumin LDH lymphocyte lab test",
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
        return _build_static_only_pack(protocol_id)

    ta_warning = get_hotspot_warning(ta_pack, CONCEPT)
    user_prompt = _build_user_prompt(chunks, ta_warning, protocol_id)

    result = client.extract(
        system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
        schema=LabVariablesExtraction, use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                schema=LabVariablesExtraction, use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception:
            pass

    # Merge with static template
    merged_variables = _merge_with_static_template(extraction.variables, STATIC_TEMPLATE)

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

    print(f"[LabVarsFinder] Done. {len(candidates)} variables | "
          f"confidence={extraction.overall_confidence:.2f}")
    return pack
