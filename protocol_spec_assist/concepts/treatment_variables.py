"""
Treatment variables concept finder.
Extracts treatment-related variable definitions from protocol text.

Variables typically found: LOT start/end dates, drug names, regimen names,
regimen categories, number of LOTs, maintenance therapy, clinical study drug flag,
number of treatment regimens per LOT, etc.
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

CONCEPT = "treatment_variables"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"
CONFIDENCE_THRESHOLD = 0.65

# ── Static variable template ─────────────────────────────────────────────────

def _lot_group(n):
    """Generate the standard variable group for a single line of therapy (LOTn)."""
    return [
        {"time_period": "FU", "variable_name": f"LOT{n}SD", "label": f"LOT{n} Start Date", "values": "Date", "definition": f"line_of_therapy/start_timedelta for LOT number={n}; compute date from INDEX + timedelta", "code_lists_group": "", "additional_notes": f"Index LOT varies by cohort; LOT{n} means the {n}th line after index"},
        {"time_period": "FU", "variable_name": f"LOT{n}ED", "label": f"LOT{n} End Date", "values": "Date", "definition": f"line_of_therapy/end_timedelta for LOT number={n}; compute date from INDEX + timedelta", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "FU", "variable_name": f"LOT{n}", "label": f"LOT{n} Drug Names", "values": "character", "definition": f"line_of_therapy/line_of_therapy_name for LOT number={n}; concatenated drug names", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "FU", "variable_name": f"LOT{n}NAME", "label": f"LOT{n} Regimen Name", "values": "character", "definition": f"regimen/regimen_name mapped to LOT number={n}; standardized regimen classification", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "FU", "variable_name": f"LOT{n}CAT", "label": f"LOT{n} Treatment Category", "values": "character", "definition": f"Categorized regimen for LOT{n}: e.g. chemoimmunotherapy, immunotherapy, chemotherapy, targeted, other", "code_lists_group": "", "additional_notes": "Based on drug classification mapping"},
        {"time_period": "FU", "variable_name": f"LOT{n}CATN", "label": f"LOT{n} Treatment Category (N)", "values": "numeric", "definition": f"Numeric code for LOT{n}CAT", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "FU", "variable_name": f"LOT{n}REGN", "label": f"LOT{n} Number of Regimens", "values": "numeric", "definition": f"Count of distinct regimen records within LOT number={n}", "code_lists_group": "", "additional_notes": ""},
    ]

def _lot_maintenance_group(n):
    """Generate maintenance therapy variables for a LOT."""
    return [
        {"time_period": "FU", "variable_name": f"LOT{n}MSD", "label": f"LOT{n} Maintenance Start Date", "values": "Date", "definition": f"line_of_therapy where LOT number={n} AND ismaintenancetherapy='true'; start_timedelta -> date", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "FU", "variable_name": f"LOT{n}MED", "label": f"LOT{n} Maintenance End Date", "values": "Date", "definition": f"line_of_therapy where LOT number={n} AND ismaintenancetherapy='true'; end_timedelta -> date", "code_lists_group": "", "additional_notes": ""},
        {"time_period": "FU", "variable_name": f"LOT{n}M", "label": f"LOT{n} Maintenance Drug", "values": "character", "definition": f"line_of_therapy_name where LOT number={n} AND ismaintenancetherapy='true'", "code_lists_group": "", "additional_notes": ""},
    ]

STATIC_TEMPLATE = [
    {"time_period": "FU", "variable_name": "LOTN", "label": "Number of Lines of Therapy", "values": "numeric", "definition": "Count of distinct line_of_therapy records per patient; max(line_number) from line_of_therapy table", "code_lists_group": "", "additional_notes": "Single value per patient"},
    *_lot_group(1), *_lot_maintenance_group(1),
    *_lot_group(2), *_lot_maintenance_group(2),
    *_lot_group(3), *_lot_maintenance_group(3),
    *_lot_group(4), *_lot_maintenance_group(4),
    {"time_period": "FU", "variable_name": "CSD", "label": "Clinical Study Drug Flag", "values": "1=Yes; 0=No", "definition": "DrugEpisode where isclinicalstudydrug='true'; 1 if patient has any clinical study drug episode, 0 otherwise", "code_lists_group": "", "additional_notes": "Flags patients receiving investigational therapy"},
    {"time_period": "FU", "variable_name": "LOT1DUR", "label": "LOT1 Treatment Duration (days)", "values": "numeric", "definition": "LOT1ED - LOT1SD + 1; duration in days of first line of therapy", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "FU", "variable_name": "TTNT", "label": "Time to Next Treatment (days)", "values": "numeric", "definition": "LOT2SD - LOT1SD; time from start of LOT1 to start of LOT2 in days", "code_lists_group": "", "additional_notes": "Missing if patient has only 1 LOT"},
]


class TreatmentVarsExtraction(BaseModel):
    class VariableExtraction(BaseModel):
        chunk_id: Optional[str] = Field(default=None)
        time_period: str = Field(description="e.g. FU, INDEX, PRE_INT/FU")
        variable_name: str = Field(description="e.g. LOTN, LOT1SD, LOT1ED, LOT1, LOT1NAME, LOT1CAT, CSD")
        label: str = Field(description="e.g. 'Number of LOTs', 'LOT1 Start Date', 'LOT1 drug names'")
        values: str = Field(description="e.g. 'numeric', 'date', 'character', '1=Yes, 0=No'")
        definition: str = Field(description="Derivation logic from line_of_therapy, regimen tables, DrugEpisode")
        code_lists_group: str = Field(default="")
        additional_notes: str = Field(default="", description="e.g. 'Single value per patient, copy to all cohorts'")
        sponsor_term: Optional[str] = None
        explicit: ExplicitType = "explicit"
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str = ""

    variables: list[VariableExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst specializing in programming specifications.
Your task is to identify TREATMENT-RELATED VARIABLES from the protocol text.

Treatment variables typically include:
- Number of lines of therapy (LOTN) — count of LOTs
- Per-LOT variables (for LOT1 through LOT4+):
  - Start date (LOT1SD, LOT2SD, etc.) — from line_of_therapy/start_timedelta
  - End date (LOT1ED, LOT2ED, etc.) — from line_of_therapy/end_timedelta
  - Drug names (LOT1, LOT2, etc.) — from line_of_therapy/line_of_therapy_name
  - Regimen names (LOT1NAME, etc.) — mapped from regimen/regimen_name
  - Treatment category (LOT1CAT, LOT1CATN) — categorized regimen
  - Number of regimens per LOT (LOT1REGN, etc.)
- Maintenance therapy (LOT1MSD, LOT1MED, LOT1M) — ismaintenancetherapy flag
- Clinical study drug flag (CSD) — from DrugEpisode table
- Treatment duration, time on treatment

For each variable provide:
- time_period: typically FU (follow-up) for treatment variables
- variable_name: short code
- label: human-readable description
- values: data type (numeric, date, character)
- definition: derivation from data tables (line_of_therapy, regimen, DrugEpisode)
- additional_notes: per-patient vs per-cohort, missing value handling

Rules:
- Extract ONLY treatment variables, NOT outcomes/endpoints.
- Include variables for multiple LOTs if protocol defines them.
- Note cohort-dependent definitions (index LOT varies by cohort).
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
Extract all treatment-related variable definitions from the following protocol text.

{context}"""


def _merge_with_static_template(extraction_variables, static_template):
    """Merge LLM-extracted variables with the static template."""
    llm_by_name = {v.variable_name.upper(): v for v in extraction_variables}
    VarClass = TreatmentVarsExtraction.VariableExtraction
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


def find_treatment_variables(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the treatment variables concept finder workflow."""

    queries = build_query_bank(
        "treatment line of therapy LOT regimen drug therapy chemotherapy immunotherapy "
        "maintenance treatment duration clinical study drug treatment pattern",
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
        schema=TreatmentVarsExtraction, use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                schema=TreatmentVarsExtraction, use_adjudicator=True,
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

    print(f"[TreatVarsFinder] Done. {len(candidates)} variables | "
          f"confidence={extraction.overall_confidence:.2f}")
    return pack
