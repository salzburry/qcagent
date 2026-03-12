"""
Demographics concept finder.
Extracts demographic variable definitions from protocol text.

Variables typically found: age, sex, race, ethnicity, practice type,
BMI, weight, height, BSA, smoking status, etc.
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

CONCEPT = "demographics"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"
CONFIDENCE_THRESHOLD = 0.65

# ── Static variable template ─────────────────────────────────────────────────
# Exhaustive list of expected demographic variables.
# These are ALWAYS included in the spec output (as "inferred" if not found in protocol).
# The LLM's job is to confirm and customize definitions from actual protocol text.
STATIC_TEMPLATE = [
    {"time_period": "STUDY_PD", "variable_name": "AGE", "label": "Age at Index", "values": "numeric", "definition": "Computed from patient/date_of_birth and INDEX date; if date_of_birth missing, use patient/age_at_diagnosis", "code_lists_group": "", "additional_notes": "Single value per patient, copy to all cohorts"},
    {"time_period": "STUDY_PD", "variable_name": "AGEGR", "label": "Age Group", "values": "<65; 65-74; >=75", "definition": "Derived from AGE: <65, 65-74, >=75", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "AGEGRN", "label": "Age Group (N)", "values": "1; 2; 3", "definition": "Numeric code for AGEGR: 1=<65, 2=65-74, 3=>=75", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "SEX", "label": "Sex", "values": "Male; Female; Unknown", "definition": "patient/sex", "code_lists_group": "", "additional_notes": "Single value per patient, copy to all cohorts"},
    {"time_period": "STUDY_PD", "variable_name": "SEXN", "label": "Sex (N)", "values": "1; 2; 3", "definition": "Numeric code for SEX: 1=Male, 2=Female, 3=Unknown", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "RACE", "label": "Race", "values": "White; Black; Asian; Other; Unknown", "definition": "patient/race; if missing use 'Unknown'", "code_lists_group": "", "additional_notes": "Single value per patient"},
    {"time_period": "STUDY_PD", "variable_name": "RACEN", "label": "Race (N)", "values": "1; 2; 3; 4; 5", "definition": "Numeric code for RACE: 1=White, 2=Black, 3=Asian, 4=Other, 5=Unknown", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "ETH", "label": "Ethnicity", "values": "Hispanic or Latino; Not Hispanic or Latino; Unknown", "definition": "patient/ethnicity; if missing use 'Unknown'", "code_lists_group": "", "additional_notes": "Single value per patient"},
    {"time_period": "STUDY_PD", "variable_name": "ETHN", "label": "Ethnicity (N)", "values": "1; 2; 3", "definition": "Numeric code for ETH: 1=Hispanic or Latino, 2=Not Hispanic or Latino, 3=Unknown", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "PRACTIC", "label": "Practice Type", "values": "Academic; Community; Unknown", "definition": "practice/practice_type mapped to Academic/Community; if missing use 'Unknown'", "code_lists_group": "", "additional_notes": "Single value per patient, copy to all cohorts"},
    {"time_period": "STUDY_PD", "variable_name": "PRACTICN", "label": "Practice Type (N)", "values": "1; 2; 3", "definition": "Numeric code for PRACTIC: 1=Academic, 2=Community, 3=Unknown", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "PRE_INT", "variable_name": "WEIGHT", "label": "Weight (kg)", "values": "numeric", "definition": "vitals/weight closest to INDEX date within PRE_INT period; convert lbs to kg if needed", "code_lists_group": "", "additional_notes": "Closest to index date"},
    {"time_period": "PRE_INT", "variable_name": "HEIGHT", "label": "Height (cm)", "values": "numeric", "definition": "vitals/height closest to INDEX date within PRE_INT period; convert inches to cm if needed", "code_lists_group": "", "additional_notes": "Closest to index date"},
    {"time_period": "PRE_INT", "variable_name": "BMI", "label": "BMI (kg/m²)", "values": "numeric", "definition": "Computed as WEIGHT / (HEIGHT/100)^2; or from vitals/bmi if available", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "PRE_INT", "variable_name": "BMIGR", "label": "BMI Group", "values": "Underweight (<18.5); Normal (18.5-24.9); Overweight (25-29.9); Obese (>=30); Unknown", "definition": "Derived from BMI: <18.5, 18.5-24.9, 25-29.9, >=30", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "PRE_INT", "variable_name": "BSA", "label": "Body Surface Area (m²)", "values": "numeric", "definition": "Computed from HEIGHT and WEIGHT using Mosteller formula: sqrt(HEIGHT_cm * WEIGHT_kg / 3600)", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "INDEXYR", "label": "Index Year", "values": "numeric (YYYY)", "definition": "Year component of INDEX date", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "REGION", "label": "Geographic Region", "values": "character", "definition": "practice/region or patient geographic region if available", "code_lists_group": "", "additional_notes": "Single value per patient"},
    {"time_period": "STUDY_PD", "variable_name": "SMOKING", "label": "Smoking Status", "values": "Current; Former; Never; Unknown", "definition": "social_history/smoking_status closest to INDEX date; if missing use 'Unknown'", "code_lists_group": "", "additional_notes": ""},
]


class DemographicsExtraction(BaseModel):
    """Schema-constrained LLM output for demographics extraction."""

    class VariableExtraction(BaseModel):
        chunk_id: Optional[str] = Field(default=None, description="chunk_id from input chunk")
        time_period: str = Field(description="Time period this variable applies to, e.g. STUDY_PD, PRE_INT, FU")
        variable_name: str = Field(description="Short variable name, e.g. AGE, SEX, RACE, BMI")
        label: str = Field(description="Human-readable label, e.g. 'Age at Index'")
        values: str = Field(description="Possible values/format, e.g. 'numeric', 'Male; Female', 'date'")
        definition: str = Field(description="How to derive/compute this variable from data source")
        code_lists_group: str = Field(default="", description="Code list reference if applicable, e.g. N/A")
        additional_notes: str = Field(default="", description="Extra notes, caveats, single-value-for-all-cohorts, etc.")
        sponsor_term: Optional[str] = Field(default=None, description="Term used in the protocol")
        explicit: ExplicitType = Field(description="Whether explicitly stated or inferred")
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str = Field(description="Why this variable was identified")

    variables: list[VariableExtraction] = Field(
        description="All demographic variables found in protocol, each as a row definition"
    )
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst specializing in real-world data programming specifications.
Your task is to identify DEMOGRAPHIC VARIABLES defined or implied in the protocol text.

Demographic variables typically include:
- Age at index (AGE), age groups
- Sex (SEX, SEXN)
- Race (RACE, RACEN), ethnicity (ETH, ETHN)
- Practice type (PRACTIC, PRACTICN) — academic, community, etc.
- Weight (WEIGHT), Height (HEIGHT), BMI, BMI groups, BSA
- Index year, diagnosis year
- Geographic region
- Insurance type
- Smoking status

For each variable, provide:
- time_period: when to assess (STUDY_PD for all patients, PRE_INT for pre-index, FU for follow-up)
- variable_name: short code like AGE, SEX, RACE, BMI
- label: human-readable description
- values: expected data types/categories (e.g. 'numeric', 'Male; Female', '1,2,3')
- definition: how to derive from data source (e.g. 'patient/age_at_diagnosis', 'patient/sex')
- code_lists_group: N/A if not applicable
- additional_notes: any caveats (censoring rules, single value across cohorts, etc.)

Rules:
- Extract ONLY variables that the protocol defines or clearly requires.
- Include both the categorical variable AND its numeric coded version (e.g. SEX and SEXN).
- Mark inferred variables as "inferred".
- Return chunk_id for provenance.
- Respond ONLY with valid JSON matching the schema."""


def _merge_with_static_template(extraction_variables, static_template):
    """Merge LLM-extracted variables with the static template.

    Strategy: start from the static template (guaranteed correct tab placement).
    If the LLM found a matching variable (by variable_name), use the LLM's
    definition/values/notes (which are protocol-specific). Otherwise keep the
    static default and mark as 'inferred'.

    Any extra variables the LLM found that aren't in the template are appended
    at the end (the LLM may have found protocol-specific variables we didn't
    pre-define).
    """
    llm_by_name = {}
    for v in extraction_variables:
        llm_by_name[v.variable_name.upper()] = v

    merged = []
    used_names = set()

    # Use the VariableExtraction class from the extraction schema
    VarClass = DemographicsExtraction.VariableExtraction

    for tmpl in static_template:
        name = tmpl["variable_name"].upper()
        used_names.add(name)
        if name in llm_by_name:
            # LLM found this variable — use its protocol-specific data
            merged.append(llm_by_name[name])
        else:
            # Not found by LLM — keep static default as inferred
            merged.append(VarClass(
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

    # Extra LLM-found variables not in template go to unmapped bucket
    # (not auto-appended to this tab — prevents model contamination)
    unmapped = []
    for v in extraction_variables:
        if v.variable_name.upper() not in used_names:
            unmapped.append(v)

    return [m for m in merged if m is not None], unmapped


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
Extract all demographic variable definitions from the following protocol text chunks.

{context}"""


def _build_static_only_pack(protocol_id: str, data_source: str = "generic") -> EvidencePack:
    """Build an EvidencePack from static template alone (no LLM, no retrieval)."""
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
            source_type="narrative",
            sponsor_term=tmpl["variable_name"],
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


def find_demographics(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the demographics concept finder workflow."""

    # Step 1: Build query bank
    queries = build_query_bank(
        "demographics age sex race ethnicity BMI weight height practice type baseline characteristics patient population",
        ta_pack, CONCEPT,
    )

    # Step 2: Hybrid retrieval
    priority_sections = get_section_priority(ta_pack, CONCEPT)
    chunks = index.search(
        query=queries[0], protocol_id=protocol_id,
        concept_queries=queries[1:],
        top_k_retrieve=25, top_k_rerank=10,
        include_tables=True, priority_sections=priority_sections,
    )

    if not chunks:
        # No protocol text found — return static template as inferred defaults
        return _build_static_only_pack(protocol_id, data_source)

    # Step 3: TA warning
    ta_warning = get_hotspot_warning(ta_pack, CONCEPT)

    # Step 4: LLM extraction
    user_prompt = _build_user_prompt(chunks, ta_warning, protocol_id)
    result = client.extract(
        system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
        schema=DemographicsExtraction, use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    # Step 5: Confidence router
    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                schema=DemographicsExtraction, use_adjudicator=True,
                prompt_version=PROMPT_VERSION,
            )
            extraction, model_used = result.parsed, result.model_used
            used_adjudicator = True
        except Exception:
            pass

    # Step 6: Merge with source-resolved static template (ensures correct tab placement)
    resolved_template = resolve_static_template(STATIC_TEMPLATE, data_source, CONCEPT)
    merged_variables, unmapped_variables = _merge_with_static_template(extraction.variables, resolved_template)

    # Step 7: Build EvidencePack
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
        concept_metadata={
            "per_candidate": per_candidate_meta,
            "unmapped_variables": [
                {"variable_name": v.variable_name, "label": v.label,
                 "definition": v.definition, "confidence": v.confidence}
                for v in unmapped_variables
            ],
        },
        low_retrieval_signal=len(chunks) < 3,
        adjudicator_used=used_adjudicator,
        requires_human_selection=True,
        finder_version=FINDER_VERSION, model_used=model_used,
        prompt_version=PROMPT_VERSION,
    )

    print(f"[DemographicsFinder] Done. {len(candidates)} variables | "
          f"confidence={extraction.overall_confidence:.2f}")
    return pack
