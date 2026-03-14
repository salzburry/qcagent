"""
Demographics concept finder.
Extracts demographic variable definitions from protocol text.

Variables typically found: age, sex, race, ethnicity, practice type,
BMI, weight, height, BSA, smoking status, etc.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack, ExplicitType
from ..retrieval.search import ProtocolIndex
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import TAPack
from .base import run_template_finder, merge_with_static_template as _base_merge, build_static_only_pack as _base_static

CONCEPT = "demographics"
FINDER_VERSION = "0.3.0"
PROMPT_VERSION = "0.3.0"

# ── Static variable template ─────────────────────────────────────────────────
# Demographics are domain-neutral (age, sex, race apply to all TAs).
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

    chain_of_thought: str = Field(
        description="Think step by step about the demographic variables in the protocol. "
        "Identify which demographics are explicitly mentioned vs implied, "
        "and note any specific definitions or categories given."
    )

    class VariableExtraction(BaseModel):
        reasoning: str = Field(description="Why this variable was identified")
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

IMPORTANT: Use the chain_of_thought field to reason about the protocol text BEFORE listing variables.
Think step by step — identify which demographic variables the protocol explicitly requires or implies.

Rules:
- Extract ONLY variables that the protocol defines or clearly requires.
- Include both the categorical variable AND its numeric coded version (e.g. SEX and SEXN).
- Mark inferred variables as "inferred".
- Return chunk_id for provenance.
- Respond ONLY with valid JSON matching the schema."""

BASE_QUERY = (
    "demographics age sex race ethnicity BMI weight height "
    "practice type baseline characteristics patient population"
)


def _merge_with_static_template(extraction_variables, static_template):
    return _base_merge(extraction_variables, static_template, DemographicsExtraction.VariableExtraction)

def _build_static_only_pack(protocol_id, data_source="generic"):
    return _base_static(protocol_id, CONCEPT, STATIC_TEMPLATE, data_source, FINDER_VERSION, PROMPT_VERSION)


def find_demographics(
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack: Optional[TAPack] = None,
    data_source: str = "generic",
) -> EvidencePack:
    """Run the demographics concept finder workflow."""
    return run_template_finder(
        protocol_id, index, client, ta_pack, data_source,
        concept=CONCEPT,
        base_query=BASE_QUERY,
        system_prompt=SYSTEM_PROMPT,
        extraction_schema=DemographicsExtraction,
        static_template=STATIC_TEMPLATE,
        finder_version=FINDER_VERSION,
        prompt_version=PROMPT_VERSION,
        finder_name="DemographicsFinder",
    )
