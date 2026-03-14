"""
Clinical characteristics concept finder.
Extracts clinical characteristic variable definitions (excluding biomarkers and labs).

Domain-neutral by default. Disease-specific variables (ECOG, Ann Arbor stage,
DLBCLFL, B symptoms, etc.) are extracted only if found in the protocol text.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, ExplicitType
from ..retrieval.search import ProtocolIndex
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack
from .base import run_template_finder, merge_with_static_template as _base_merge, build_static_only_pack as _base_static

CONCEPT = "clinical_characteristics"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"

# ── Static variable template ─────────────────────────────────────────────────
# Domain-neutral only. Disease-specific variables (ECOG, Ann Arbor stage,
# DLBCLFL, B symptoms, bulky disease, extranodal, CNS, IPI, etc.) belong in
# TA packs, NOT here.
STATIC_TEMPLATE = [
    {"time_period": "PRE_INT", "variable_name": "CCI", "label": "Charlson Comorbidity Index", "values": "numeric", "definition": "Calculated from diagnosis codes in PRE_INT period using Quan adaptation of CCI; sum of weighted comorbidity categories", "code_lists_group": "", "additional_notes": "Excludes primary diagnosis from calculation"},
    {"time_period": "PRE_INT", "variable_name": "CCIGR", "label": "CCI Group", "values": "0; 1-2; 3+; Missing", "definition": "Derived from CCI: 0, 1-2, 3+", "code_lists_group": "", "additional_notes": ""},
    {"time_period": "PRE_INT", "variable_name": "CCIGRN", "label": "CCI Group (N)", "values": "1; 2; 3; 99", "definition": "Numeric code for CCIGR: 1=0, 2=1-2, 3=3+, 99=Missing", "code_lists_group": "", "additional_notes": ""},
]


class ClinicalCharsExtraction(BaseModel):
    chain_of_thought: str = Field(description="Think step by step about the clinical characteristics in the protocol text. Identify which clinical characteristic variables are explicitly mentioned, note any specific definitions, and assess your confidence before structuring the answer.")

    class VariableExtraction(BaseModel):
        reasoning: str = Field(description="Why this variable is relevant and how it was identified")
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

    variables: list[VariableExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst specializing in programming specifications.
Your task is to identify CLINICAL CHARACTERISTIC VARIABLES (excluding biomarkers and lab values)
that are ACTUALLY DEFINED OR REFERENCED in the protocol text.

Clinical characteristics may include (depending on disease area):
- Comorbidity indices (CCI, Elixhauser, etc.)
- Disease stage or severity scores
- Performance status (e.g. ECOG, Karnofsky)
- Disease subtype/histology
- Disease duration, time since diagnosis
- Prior therapy history
- Any other baseline clinical variable the protocol defines

For each variable provide time_period, variable_name, label, values, definition,
code_lists_group, additional_notes. Include both categorical and numeric coded versions.

IMPORTANT: Use the chain_of_thought field to reason about the protocol text BEFORE listing variables.
Think step by step — identify relevant passages and assess what the protocol explicitly requires.

Rules:
- Extract ONLY variables that the protocol ACTUALLY DEFINES or REQUIRES.
- Do NOT invent variables from a different disease area.
- If the protocol is about liver disease, do not add lymphoma variables.
- If the protocol is about cardiovascular disease, do not add oncology variables.
- Mark inferred variables as "inferred".
- Return chunk_id for provenance.
- Respond ONLY with valid JSON."""

BASE_QUERY = (
    "clinical characteristics disease stage comorbidity "
    "CCI baseline clinical performance status disease severity"
)


def _build_static_only_pack(protocol_id, data_source="generic"):
    return _base_static(protocol_id, CONCEPT, STATIC_TEMPLATE, data_source, FINDER_VERSION, PROMPT_VERSION)


def find_clinical_characteristics(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the clinical characteristics concept finder workflow."""
    return run_template_finder(
        protocol_id, index, client, ta_pack, data_source,
        concept=CONCEPT,
        base_query=BASE_QUERY,
        system_prompt=SYSTEM_PROMPT,
        extraction_schema=ClinicalCharsExtraction,
        static_template=STATIC_TEMPLATE,
        finder_version=FINDER_VERSION,
        prompt_version=PROMPT_VERSION,
        finder_name="ClinCharsFinder",
    )
