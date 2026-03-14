"""
Treatment variables concept finder.
Extracts treatment-related variable definitions from protocol text.

Empty template by default. LOT structure, maintenance therapy, and
chemo/immunotherapy categories are oncology-specific and extracted only
if the protocol defines them.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, ExplicitType
from ..retrieval.search import ProtocolIndex
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack
from .base import run_template_finder, build_static_only_pack as _base_static

CONCEPT = "treatment_variables"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"

# ── Static variable template ─────────────────────────────────────────────────
# Empty by default. LOT structure, maintenance therapy, and
# chemo/immunotherapy categories are oncology-specific and must NOT leak
# into non-oncology protocols. Disease-specific templates belong in TA packs.
STATIC_TEMPLATE = []


class TreatmentVarsExtraction(BaseModel):
    class VariableExtraction(BaseModel):
        chunk_id: Optional[str] = Field(default=None)
        time_period: str = Field(description="e.g. FU, INDEX, PRE_INT/FU")
        variable_name: str = Field(description="e.g. LOTN, LOT1SD, LOT1ED, LOT1, LOT1NAME, LOT1CAT, CSD")
        label: str = Field(description="e.g. 'Number of LOTs', 'LOT1 Start Date', 'LOT1 drug names'")
        values: str = Field(description="e.g. 'numeric', 'date', 'character', '1=Yes, 0=No'")
        definition: str = Field(description="Derivation logic from data tables")
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
Your task is to identify TREATMENT-RELATED VARIABLES that are ACTUALLY DEFINED in the protocol text.

Treatment variables may include (depending on disease area):
- Medication/drug exposure variables (start/end dates, drug names, doses)
- Treatment duration
- Treatment patterns or switching
- Lines of therapy (only if protocol explicitly defines LOT structure)
- Concomitant medications

For each variable provide:
- time_period: typically FU (follow-up) for treatment variables
- variable_name: short code
- label: human-readable description
- values: data type (numeric, date, character)
- definition: derivation from data tables
- additional_notes: per-patient vs per-cohort, missing value handling

Rules:
- Extract ONLY treatment variables that the protocol ACTUALLY DEFINES.
- Do NOT invent LOT1/LOT2/maintenance therapy variables for a non-oncology study.
- If the protocol does not define treatment variables, return an empty variables list.
- Mark inferred variables as "inferred".
- Return chunk_id for provenance.
- Respond ONLY with valid JSON."""

BASE_QUERY = (
    "treatment medication drug therapy regimen "
    "treatment duration concomitant medication treatment pattern"
)


def _build_static_only_pack(protocol_id, data_source="generic"):
    return _base_static(protocol_id, CONCEPT, STATIC_TEMPLATE, data_source, FINDER_VERSION, PROMPT_VERSION)


def find_treatment_variables(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the treatment variables concept finder workflow."""
    return run_template_finder(
        protocol_id, index, client, ta_pack, data_source,
        concept=CONCEPT,
        base_query=BASE_QUERY,
        system_prompt=SYSTEM_PROMPT,
        extraction_schema=TreatmentVarsExtraction,
        static_template=STATIC_TEMPLATE,
        finder_version=FINDER_VERSION,
        prompt_version=PROMPT_VERSION,
        finder_name="TreatVarsFinder",
    )
