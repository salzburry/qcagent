"""
Laboratory variables concept finder.
Extracts lab test variable definitions from protocol text.

Variables typically found: neutrophils (ANC), platelets, hemoglobin,
creatinine, bilirubin, AST, ALT, alkaline phosphatase, albumin,
LDH, lymphocytes, etc. — each with Present/N/Date/Value sub-variables.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, ExplicitType
from ..retrieval.search import ProtocolIndex
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack
from .base import run_template_finder, build_static_only_pack as _base_static

CONCEPT = "lab_variables"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"

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

For each variable, define:
- How to determine presence (assessed='true' AND at least one of timedelta fields not missing AND quantitative_result not missing)
- How to select the closest record to index date
- Unit standardization (which function / target units)
- Whether to take lowest or highest value when multiple records on same date

Rules:
- Extract ONLY lab variables, NOT biomarkers or clinical characteristics.
- Include all 4 sub-variables per lab test when protocol defines them.
- Note unit standardization requirements.
- Mark inferred variables as "inferred".
- Return chunk_id for provenance.
- Respond ONLY with valid JSON."""

BASE_QUERY = (
    "laboratory lab values baseline labs neutrophil platelet hemoglobin creatinine "
    "bilirubin AST ALT alkaline phosphatase albumin LDH lymphocyte lab test"
)


def _build_static_only_pack(protocol_id, data_source="generic"):
    return _base_static(protocol_id, CONCEPT, STATIC_TEMPLATE, data_source, FINDER_VERSION, PROMPT_VERSION)


def find_lab_variables(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the lab variables concept finder workflow."""
    return run_template_finder(
        protocol_id, index, client, ta_pack, data_source,
        concept=CONCEPT,
        base_query=BASE_QUERY,
        system_prompt=SYSTEM_PROMPT,
        extraction_schema=LabVariablesExtraction,
        static_template=STATIC_TEMPLATE,
        finder_version=FINDER_VERSION,
        prompt_version=PROMPT_VERSION,
        finder_name="LabVarsFinder",
    )
