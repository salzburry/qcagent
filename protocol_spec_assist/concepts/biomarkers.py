"""
Biomarker variables concept finder.
Extracts biomarker variable definitions from protocol text.

Empty template by default — biomarkers are entirely disease-specific.
The LLM extracts only what the protocol actually defines.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, ExplicitType
from ..retrieval.search import ProtocolIndex
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack
from .base import run_template_finder, build_static_only_pack as _base_static

CONCEPT = "biomarkers"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"

# ── Static variable template ─────────────────────────────────────────────────
# Empty by default. Biomarkers are entirely disease-specific (BCL2/MYC for
# lymphoma, HER2 for breast, EGFR for lung, etc.). The generic path must NOT
# pre-populate any disease-family markers. Disease-specific templates belong
# in TA packs.
STATIC_TEMPLATE = []


class BiomarkersExtraction(BaseModel):
    chain_of_thought: str = Field(description="Think step by step about the biomarkers in the protocol text. Identify which biomarkers are explicitly mentioned, note any specific definitions, and assess your confidence before structuring the answer.")

    class VariableExtraction(BaseModel):
        reasoning: str = ""
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

    variables: list[VariableExtraction] = Field(default_factory=list)
    contradictions_found: bool = False
    contradiction_detail: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """You are an expert RWE protocol analyst specializing in programming specifications.
Your task is to identify BIOMARKER VARIABLES that are ACTUALLY DEFINED in the protocol text.

Extract ONLY biomarkers the protocol explicitly mentions or requires. Do NOT assume
a disease area or invent biomarkers from a different therapeutic area.

For each biomarker variable, there are often related sub-variables:
1. Status variable — Positive/Negative/Unknown/Missing
2. Numeric coded version — 1,2,3,4
3. Closest date variable — Date of assessment
4. Value variable — Quantitative result if applicable

Provide the derivation logic for each variable.

IMPORTANT: Use the chain_of_thought field to reason about the protocol text BEFORE listing variables.
Think step by step — identify relevant passages and assess what the protocol explicitly requires.

Rules:
- Extract ONLY biomarker variables that the protocol ACTUALLY DEFINES.
- Do NOT add biomarkers from other disease areas (e.g. no BCL2/MYC for a liver disease study).
- If the protocol does not mention any biomarkers, return an empty variables list.
- Extract ONLY biomarker variables, NOT lab values or clinical characteristics.
- Mark inferred variables as "inferred".
- Return chunk_id for provenance.
- Respond ONLY with valid JSON."""

BASE_QUERY = (
    "biomarker molecular marker expression rearrangement "
    "mutation deletion IHC FISH immunohistochemistry"
)


def _build_static_only_pack(protocol_id, data_source="generic"):
    return _base_static(protocol_id, CONCEPT, STATIC_TEMPLATE, data_source, FINDER_VERSION, PROMPT_VERSION)


def find_biomarkers(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the biomarkers concept finder workflow."""
    return run_template_finder(
        protocol_id, index, client, ta_pack, data_source,
        concept=CONCEPT,
        base_query=BASE_QUERY,
        system_prompt=SYSTEM_PROMPT,
        extraction_schema=BiomarkersExtraction,
        static_template=STATIC_TEMPLATE,
        finder_version=FINDER_VERSION,
        prompt_version=PROMPT_VERSION,
        finder_name="BiomarkersFinder",
    )
