"""
Clinical characteristics concept finder.
Extracts clinical characteristic variable definitions (excluding biomarkers and labs).

Variables typically found: ECOG, stage, disease subtype, comorbidity index (CCI),
B symptoms, bulky disease, extranodal involvement, histology flags, etc.
"""

from __future__ import annotations
import hashlib
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, EvidenceCandidate, ExplicitType
from ..retrieval.search import ProtocolIndex, RetrievedChunk
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack, build_query_bank, get_hotspot_warning, get_section_priority

CONCEPT = "clinical_characteristics"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"
CONFIDENCE_THRESHOLD = 0.65

# ── Static variable template ─────────────────────────────────────────────────
STATIC_TEMPLATE = [
    {"time_period": "ASSESS_ECOG", "variable_name": "ECOG", "label": "ECOG Performance Status", "values": "0; 1; 2; 3; 4; Unknown; Missing", "definition": "ecog_performance_status/ecog_score closest to INDEX date within assessment window; if multiple on same date use worst (highest) score", "code_lists_group": "", "additional_notes": "Assessment window defined per protocol; typically ±30 days of index"},
    {"time_period": "ASSESS_ECOG", "variable_name": "ECOGN", "label": "ECOG Performance Status (N)", "values": "0; 1; 2; 3; 4; 98; 99", "definition": "Numeric code for ECOG: 0-4 as-is, 98=Unknown, 99=Missing", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "STAGE", "label": "Stage at Diagnosis", "values": "I; II; III; IV; Unknown; Missing", "definition": "staging/stage_at_diagnosis; use Ann Arbor staging system for lymphoma", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "STAGEN", "label": "Stage at Diagnosis (N)", "values": "1; 2; 3; 4; 98; 99", "definition": "Numeric code for STAGE: I=1, II=2, III=3, IV=4, Unknown=98, Missing=99", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "STAGEGR", "label": "Stage Group", "values": "Early (I-II); Advanced (III-IV); Unknown", "definition": "Derived from STAGE: I-II=Early, III-IV=Advanced", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "HISTOLOGY", "label": "Histology/Subtype", "values": "character", "definition": "diagnosis/histology_description or disease_subtype from diagnosis table", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "DLBCLFL", "label": "DLBCL Flag", "values": "Yes; No", "definition": "1 if histology indicates DLBCL (diffuse large B-cell lymphoma), 0 otherwise", "code_lists_group": "", "additional_notes": "Based on histology coding"},
    {"time_period": "PRE_INT", "variable_name": "CCI", "label": "Charlson Comorbidity Index", "values": "numeric", "definition": "Calculated from diagnosis codes in PRE_INT period using Quan adaptation of CCI; sum of weighted comorbidity categories", "code_lists_group": "", "additional_notes": "Excludes cancer diagnosis itself from calculation"},
    {"time_period": "PRE_INT", "variable_name": "CCIGR", "label": "CCI Group", "values": "0; 1-2; 3+; Missing", "definition": "Derived from CCI: 0, 1-2, 3+", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "PRE_INT", "variable_name": "CCIGRN", "label": "CCI Group (N)", "values": "1; 2; 3; 99", "definition": "Numeric code for CCIGR: 1=0, 2=1-2, 3=3+, 99=Missing", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "ASSESS_BSYMPT", "variable_name": "BSYMPT", "label": "B Symptoms", "values": "Yes; No; Unknown; Missing", "definition": "b_symptoms/b_symptoms_status; 'Yes' if any B symptom recorded (fever, night sweats, weight loss >10%)", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "ASSESS_BSYMPT", "variable_name": "BSYMPTN", "label": "B Symptoms (N)", "values": "1; 0; 98; 99", "definition": "Numeric code for BSYMPT: 1=Yes, 0=No, 98=Unknown, 99=Missing", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "BULKY", "label": "Bulky Disease", "values": "Yes; No; Unknown; Missing", "definition": "bulky_disease/bulky_disease_status; 'Yes' if any mass ≥7.5cm or ≥10cm per protocol definition", "code_lists_group": "", "additional_notes": "Threshold varies by protocol"},
    {"time_period": "STUDY_PD", "variable_name": "BULKYN", "label": "Bulky Disease (N)", "values": "1; 0; 98; 99", "definition": "Numeric code for BULKY: 1=Yes, 0=No, 98=Unknown, 99=Missing", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "EXTRANOD", "label": "Extranodal Involvement", "values": "Yes; No; Unknown; Missing", "definition": "extranodal_sites/extranodal_involvement; 'Yes' if ≥1 extranodal site involved", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "EXTRANODN", "label": "Extranodal Involvement (N)", "values": "1; 0; 98; 99", "definition": "Numeric code for EXTRANOD: 1=Yes, 0=No, 98=Unknown, 99=Missing", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "CNSINV", "label": "CNS Involvement", "values": "Yes; No; Unknown; Missing", "definition": "cns_involvement/cns_status; 'Yes' if CNS disease present at diagnosis", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "BMINV", "label": "Bone Marrow Involvement", "values": "Yes; No; Unknown; Missing", "definition": "bone_marrow/bone_marrow_involvement; 'Yes' if bone marrow biopsy positive", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "STUDY_PD", "variable_name": "IPI", "label": "IPI Score", "values": "0; 1; 2; 3; 4; 5; Unknown; Missing", "definition": "International Prognostic Index: age>60 + stage III-IV + ECOG≥2 + elevated LDH + >1 extranodal site; sum of risk factors", "code_lists_group": "", "additional_notes": "Derived from AGE, STAGE, ECOG, LDH, EXTRANOD"},
    {"time_period": "STUDY_PD", "variable_name": "IPIGR", "label": "IPI Risk Group", "values": "Low (0-1); Low-Int (2); High-Int (3); High (4-5); Unknown", "definition": "Derived from IPI score: 0-1=Low, 2=Low-Int, 3=High-Int, 4-5=High", "code_lists_group": "", "additional_notes": ""},
]


class ClinicalCharsExtraction(BaseModel):
    class VariableExtraction(BaseModel):
        chunk_id: Optional[str] = Field(default=None)
        time_period: str = Field(description="e.g. STUDY_PD, PRE_INT, ASSESS_ECOG, ASSESS_BSYMPT")
        variable_name: str = Field(description="e.g. ECOG, STAGE, CCI, DLBCLFL, BSYMPT, BULKY")
        label: str = Field(description="e.g. 'ECOG Performance Status', 'Stage at diagnosis'")
        values: str = Field(description="e.g. '0,1,2,3,4,Unknown', 'I,II,III,IV', 'Yes;No;Missing'")
        definition: str = Field(description="Derivation logic from data source")
        code_lists_group: str = Field(default="")
        additional_notes: str = Field(default="")
        sponsor_term: Optional[str] = None
        explicit: ExplicitType = "explicit"
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str = ""

    variables: list[VariableExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst specializing in programming specifications.
Your task is to identify CLINICAL CHARACTERISTIC VARIABLES (excluding biomarkers and lab values).

Clinical characteristics typically include:
- ECOG performance status (ECOG, ECOGN) — assessment window, scoring rules
- Disease stage at diagnosis (STAGE, STAGEN) — staging system used
- Disease subtype/histology flags (e.g. DLBCLFL, SUBTYPE) — histology descriptions
- Charlson Comorbidity Index (CCI, CCIGR, CCIGRN) — calculation method
- B symptoms (BSYMPT, BSYMPTN) — from b_symptoms tables
- Bulky disease (BULKY, BULKYN) — from bulky_disease tables
- Extranodal involvement (EXTRANOD, EXTRANODN) — from extranodal_sites tables
- CNS involvement (CNSINV), bone marrow involvement (BMINV)
- IPI score (IPI, IPIGR)
- Prior therapy history, lines of therapy count
- Disease duration, time since diagnosis

For each variable provide time_period, variable_name, label, values, definition,
code_lists_group, additional_notes. Include both categorical and numeric coded versions.

Rules:
- Extract ONLY clinical characteristics, NOT biomarkers, labs, or treatment variables.
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
Extract all clinical characteristic variable definitions from the following protocol text.

{context}"""


def _merge_with_static_template(extraction_variables, static_template):
    """Merge LLM-extracted variables with the static template."""
    llm_by_name = {v.variable_name.upper(): v for v in extraction_variables}
    VarClass = ClinicalCharsExtraction.VariableExtraction
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


def find_clinical_characteristics(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
) -> EvidencePack:
    """Run the clinical characteristics concept finder workflow."""

    queries = build_query_bank(
        "clinical characteristics ECOG performance status disease stage comorbidity "
        "CCI B symptoms bulky disease extranodal histology subtype baseline clinical",
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
        schema=ClinicalCharsExtraction, use_adjudicator=False,
        prompt_version=PROMPT_VERSION,
    )
    extraction, model_used = result.parsed, result.model_used

    used_adjudicator = False
    if extraction.overall_confidence < CONFIDENCE_THRESHOLD or extraction.contradictions_found:
        try:
            result = client.extract(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                schema=ClinicalCharsExtraction, use_adjudicator=True,
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

    print(f"[ClinCharsFinder] Done. {len(candidates)} variables | "
          f"confidence={extraction.overall_confidence:.2f}")
    return pack
