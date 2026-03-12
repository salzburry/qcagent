"""
Data source definition registry.

Maps variable derivation definitions to specific RWD sources. Each data source
has different table structures, field names, and derivation logic.

The static templates in concept finders use GENERIC definitions by default.
When a data source is identified (from the study_period concept or user input),
source-specific definitions override the generic ones.

Supported sources:
  - cota         : COTA EHR (oncology-focused, structured tables)
  - flatiron     : Flatiron Health EHR (oncology-focused)
  - optum_cdm    : Optum Clinformatics Data Mart (claims)
  - optum_ehr    : Optum EHR (electronic health records)
  - marketscan   : IBM MarketScan (claims)
  - inalon       : Inalon (claims + lab)
  - quest        : Quest Diagnostics (lab-focused)
  - generic      : Fallback for unknown sources
"""

from __future__ import annotations
from typing import Optional


# ── Source detection ──────────────────────────────────────────────────────────

_SOURCE_KEYWORDS = {
    "cota": ["cota", "cota ehr", "cota real-world"],
    "flatiron": ["flatiron", "flatiron health", "flatiron ehr"],
    "optum_cdm": ["optum cdm", "optum clinformatics", "clinformatics"],
    "optum_ehr": ["optum ehr", "optum electronic"],
    "marketscan": ["marketscan", "ibm marketscan", "truven", "merative"],
    "inalon": ["inalon", "inovalon"],
    "quest": ["quest", "quest diagnostics"],
}


def detect_source(data_source_text: str) -> str:
    """Detect data source key from free-text data source name."""
    if not data_source_text:
        return "generic"
    lower = data_source_text.lower()
    for source_key, keywords in _SOURCE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return source_key
    return "generic"


# ── Source-specific definition overrides ──────────────────────────────────────
# Structure: DEFINITIONS[source_key][variable_name] = definition string
# Only overrides need to be specified. Missing = use generic template default.

DEFINITIONS: dict[str, dict[str, str]] = {
    # ── COTA (oncology EHR) ─────────────────────────────────────
    "cota": {
        # Demographics
        "AGE": "Computed from patient/date_of_birth and INDEX date; if date_of_birth missing, use patient/age_at_diagnosis from COTA patient table",
        "SEX": "patient/sex from COTA patient table",
        "RACE": "patient/race from COTA patient table; if missing use 'Unknown'",
        "ETH": "patient/ethnicity from COTA patient table; if missing use 'Unknown'",
        "PRACTIC": "practice/practice_type from COTA practice table mapped to Academic/Community",
        "WEIGHT": "vitals/weight closest to INDEX date within PRE_INT; convert lbs to kg if needed",
        "HEIGHT": "vitals/height closest to INDEX date within PRE_INT; convert inches to cm if needed",
        "REGION": "practice/region from COTA practice table",
        "SMOKING": "social_history/smoking_status closest to INDEX date from COTA social_history table",
        # Clinical characteristics
        "ECOG": "ecog_performance_status/ecog_score closest to INDEX date within assessment window from COTA ecog table",
        "STAGE": "staging/stage_at_diagnosis from COTA staging table; use Ann Arbor for lymphoma",
        "CCI": "Calculated from diagnosis codes in PRE_INT period using Quan adaptation of CCI from COTA diagnosis table",
        "BSYMPT": "b_symptoms/b_symptoms_status from COTA b_symptoms table; 'Yes' if any B symptom recorded",
        "BULKY": "bulky_disease/bulky_disease_status from COTA bulky_disease table",
        "EXTRANOD": "extranodal_sites/extranodal_involvement from COTA extranodal_sites table",
        "CNSINV": "cns_involvement/cns_status from COTA CNS involvement table",
        "BMINV": "bone_marrow/bone_marrow_involvement from COTA bone marrow table",
        "IPI": "International Prognostic Index derived from COTA staging, ecog, lab, extranodal tables",
        # Biomarkers — COTA has structured molecular_marker table
        "BCL2": "molecular_marker where molecular_marker_name='bcl2'; map interpretation to Positive/Negative/Unknown; Baseline Ever Status",
        "BCL6": "molecular_marker where molecular_marker_name='bcl6'; map interpretation; Baseline Ever Status",
        "MYC": "molecular_marker where molecular_marker_name='myc'; map interpretation; Baseline Ever Status",
        # Labs — COTA has structured lab_test table
        "LABNEU": "lab_test where lab_name='absolute_neutrophil_count_anc' AND assessed='true'; Present if quantitative_result not missing in PRE_INT",
        "LABLDH": "lab_test where lab_name='lactate_dehydrogenase' AND assessed='true'; Present if quantitative_result not missing in PRE_INT",
        # Treatment — COTA has line_of_therapy table
        "LOTN": "Count of distinct line_of_therapy records per patient from COTA line_of_therapy table",
        "LOT1SD": "line_of_therapy/start_timedelta for LOT number=1 from COTA; compute date from INDEX + timedelta",
        "CSD": "DrugEpisode where isclinicalstudydrug='true' from COTA DrugEpisode table",
    },

    # ── Flatiron Health (oncology EHR) ──────────────────────────
    "flatiron": {
        "AGE": "Derived from demographics.BIRTH_DT and INDEX date; age = floor((INDEX - BIRTH_DT) / 365.25)",
        "SEX": "demographics.GENDER from Flatiron demographics table",
        "RACE": "demographics.RACE from Flatiron demographics table; standardize to White/Black/Asian/Other/Unknown",
        "ETH": "demographics.ETHNICITY from Flatiron demographics table",
        "PRACTIC": "enhanced_data.PRACTICE_TYPE from Flatiron practice table",
        "WEIGHT": "vitals table WHERE vital_type='WEIGHT' closest to INDEX; convert to kg",
        "HEIGHT": "vitals table WHERE vital_type='HEIGHT' closest to INDEX; convert to cm",
        "ECOG": "ecog table, ecog_score closest to INDEX within assessment window",
        "STAGE": "staging table, stage_group; use AJCC/Ann Arbor system as appropriate",
        "CCI": "Calculated from structured_diagnosis table using Quan CCI algorithm",
        "LOTN": "Count of distinct line_of_therapy records from Flatiron line_of_therapy table",
        "LOT1SD": "line_of_therapy.LINE_START_DT for line_number=1",
        "CSD": "treatment table WHERE is_clinical_trial='Y'",
        "BCL2": "biomarker_test WHERE biomarker_name='BCL2'; result mapped to Positive/Negative",
        "LABNEU": "lab_result WHERE lab_test_name='ANC'; Present if result_value not null in PRE_INT",
        "LABLDH": "lab_result WHERE lab_test_name='LDH'; Present if result_value not null in PRE_INT",
    },

    # ── Optum CDM (claims) ──────────────────────────────────────
    "optum_cdm": {
        "AGE": "Derived from MEMBER_DETAIL.BIRTH_YR and INDEX date; age = INDEX_YEAR - BIRTH_YR",
        "SEX": "MEMBER_DETAIL.GENDER: M=Male, F=Female, U=Unknown",
        "RACE": "MEMBER_DETAIL.RACE_CD; standardize using Optum race code mapping",
        "ETH": "MEMBER_DETAIL.ETHNICITY_CD from Optum member detail; map to standard categories",
        "PRACTIC": "PROVIDER.SPECIALTY_CD for treating provider; map to Academic/Community based on facility type",
        "WEIGHT": "Not routinely available in claims; set to Missing unless linked EHR data available",
        "HEIGHT": "Not routinely available in claims; set to Missing unless linked EHR data available",
        "REGION": "MEMBER_DETAIL.STATE or MEMBER_DETAIL.DIVISION from Optum member geography",
        "SMOKING": "Identify from ICD-10 codes (F17.*, Z87.891, Z72.0) in MEDICAL_CLAIMS within PRE_INT",
        "ECOG": "Not directly available in claims; infer from proxy measures or set Missing",
        "STAGE": "Not directly available in claims; infer from ICD-10 staging codes or set Missing",
        "CCI": "Calculated from ICD-10 diagnosis codes in MEDICAL_CLAIMS within PRE_INT using Quan adaptation",
        "BSYMPT": "Not available in claims data; set to Missing",
        "BULKY": "Not available in claims data; set to Missing",
        "EXTRANOD": "Not available in claims data; set to Missing",
        "LOTN": "Derived from treatment episode algorithm applied to NDC/HCPCS codes in RX_CLAIMS and MEDICAL_CLAIMS",
        "LOT1SD": "First dispensing/administration date in LOT1 episode from claims-based LOT algorithm",
        "CSD": "Identify from J-codes for investigational agents or clinical trial procedure codes in MEDICAL_CLAIMS",
        "BCL2": "Not available in claims data; set to Missing unless linked pathology data available",
        "LABNEU": "LAB_RESULTS where TEST_NAME matches ANC/neutrophil; Present if RESULT_VALUE not null in PRE_INT",
        "LABLDH": "LAB_RESULTS where TEST_NAME matches LDH; Present if RESULT_VALUE not null in PRE_INT",
    },

    # ── Optum EHR ───────────────────────────────────────────────
    "optum_ehr": {
        "AGE": "Derived from PATIENT.BIRTH_DT and INDEX date",
        "SEX": "PATIENT.GENDER from Optum EHR patient table",
        "RACE": "PATIENT.RACE from Optum EHR; standardize categories",
        "WEIGHT": "VITALS.WEIGHT closest to INDEX in PRE_INT; convert to kg",
        "HEIGHT": "VITALS.HEIGHT closest to INDEX in PRE_INT; convert to cm",
        "ECOG": "NLP-extracted or structured ECOG from clinical notes; closest to INDEX",
        "CCI": "Calculated from DIAGNOSIS table ICD-10 codes in PRE_INT using Quan CCI",
        "LOTN": "Derived from treatment episodes in MEDICATION_ADMINISTRATIONS and PRESCRIPTIONS",
        "LABNEU": "LAB_RESULT where COMPONENT_NAME matches ANC; Present if RESULT not null in PRE_INT",
    },

    # ── MarketScan (claims) ─────────────────────────────────────
    "marketscan": {
        "AGE": "Derived from ENROLLMENT.DOBYR and INDEX date; age = INDEX_YEAR - DOBYR",
        "SEX": "ENROLLMENT.SEX: 1=Male, 2=Female",
        "RACE": "Not available in MarketScan; set to Missing",
        "ETH": "Not available in MarketScan; set to Missing",
        "PRACTIC": "Not directly available; derive from FACILITY.STDPROV provider specialty codes",
        "WEIGHT": "Not available in claims; set to Missing",
        "HEIGHT": "Not available in claims; set to Missing",
        "REGION": "ENROLLMENT.EGEOLOC (geographic location code) from MarketScan enrollment table",
        "SMOKING": "Identify from ICD-10 codes (F17.*, Z87.891, Z72.0) in OUTPATIENT_SERVICES/INPATIENT_SERVICES within PRE_INT",
        "ECOG": "Not available in claims; set to Missing",
        "STAGE": "Not available in claims; infer from ICD-10 staging codes or set Missing",
        "CCI": "Calculated from ICD-10 codes in OUTPATIENT_SERVICES + INPATIENT_SERVICES within PRE_INT",
        "BSYMPT": "Not available in claims; set to Missing",
        "BULKY": "Not available in claims; set to Missing",
        "LOTN": "Derived from treatment episode algorithm applied to NDC codes (DRUG_CLAIMS) and HCPCS (OUTPATIENT_SERVICES)",
        "LOT1SD": "First dispensing/administration date in LOT1 episode from claims-based algorithm",
        "CSD": "Identify from revenue codes for clinical trials or J-codes for investigational agents",
        "BCL2": "Not available in claims; set to Missing",
        "LABNEU": "LAB table where PROC_CD matches ANC test codes; Present if RESULT not null in PRE_INT",
    },

    # ── Inalon (claims + lab) ───────────────────────────────────
    "inalon": {
        "AGE": "Derived from MEMBER.DATE_OF_BIRTH and INDEX date",
        "SEX": "MEMBER.GENDER from Inalon member table",
        "RACE": "MEMBER.RACE from Inalon member table; may have limited availability",
        "WEIGHT": "Not routinely available; set to Missing unless linked clinical data",
        "SMOKING": "Identify from diagnosis codes in CLAIMS within PRE_INT",
        "CCI": "Calculated from ICD-10 codes in CLAIMS within PRE_INT",
        "LOTN": "Derived from treatment episode algorithm on pharmacy and medical claims",
        "LABNEU": "LAB_RESULTS where TEST_CODE matches ANC; Present if VALUE not null in PRE_INT",
        "LABLDH": "LAB_RESULTS where TEST_CODE matches LDH; Present if VALUE not null in PRE_INT",
    },

    # ── Quest Diagnostics (lab-focused) ─────────────────────────
    "quest": {
        "AGE": "Derived from PATIENT.DOB and INDEX date; limited to patients with lab orders",
        "SEX": "PATIENT.GENDER from Quest patient demographics",
        "RACE": "Limited availability in Quest data; set to Missing if not present",
        "WEIGHT": "Not available in Quest lab data; set to Missing",
        "ECOG": "Not available in Quest lab data; set to Missing",
        "STAGE": "Not available in Quest lab data; set to Missing",
        "CCI": "Not available in Quest lab data alone; requires linked claims/EHR data",
        "LOTN": "Not available in Quest lab data alone; requires linked claims/EHR data",
        "BCL2": "MOLECULAR_TEST where GENE='BCL2'; map RESULT_INTERPRETATION to Positive/Negative",
        "LABNEU": "LAB_ORDER_RESULT where TEST_NAME matches ANC/neutrophil; Present if NUMERIC_RESULT not null in PRE_INT",
        "LABLDH": "LAB_ORDER_RESULT where TEST_NAME matches LDH; Present if NUMERIC_RESULT not null in PRE_INT",
        "LABHEM": "LAB_ORDER_RESULT where TEST_NAME matches hemoglobin; Present if NUMERIC_RESULT not null in PRE_INT",
    },
}


# ── Variable availability per source ─────────────────────────────────────────
# Which variable categories are available in each source.
# Used to filter out variables that can't be derived from a given source.

SOURCE_AVAILABILITY: dict[str, dict[str, bool]] = {
    "cota":      {"demographics": True, "clinical_characteristics": True, "biomarkers": True, "lab_variables": True, "treatment_variables": True},
    "flatiron":  {"demographics": True, "clinical_characteristics": True, "biomarkers": True, "lab_variables": True, "treatment_variables": True},
    "optum_cdm": {"demographics": True, "clinical_characteristics": True, "biomarkers": False, "lab_variables": True, "treatment_variables": True},
    "optum_ehr": {"demographics": True, "clinical_characteristics": True, "biomarkers": False, "lab_variables": True, "treatment_variables": True},
    "marketscan": {"demographics": True, "clinical_characteristics": True, "biomarkers": False, "lab_variables": True, "treatment_variables": True},
    "inalon":    {"demographics": True, "clinical_characteristics": True, "biomarkers": False, "lab_variables": True, "treatment_variables": True},
    "quest":     {"demographics": True, "clinical_characteristics": False, "biomarkers": True, "lab_variables": True, "treatment_variables": False},
    "generic":   {"demographics": True, "clinical_characteristics": True, "biomarkers": True, "lab_variables": True, "treatment_variables": True},
}


def get_definition(source_key: str, variable_name: str, default: str) -> str:
    """Get the source-specific definition for a variable, falling back to default."""
    source_defs = DEFINITIONS.get(source_key, {})
    return source_defs.get(variable_name, default)


def is_variable_available(source_key: str, concept: str, variable_name: str) -> bool:
    """Check if a variable is derivable from a given data source.

    Returns True for generic source or if the concept category is available.
    Individual variables in unavailable categories are marked 'Not available'
    rather than excluded, so the reviewer can see what's missing.
    """
    avail = SOURCE_AVAILABILITY.get(source_key, SOURCE_AVAILABILITY["generic"])
    return avail.get(concept, True)


def resolve_static_template(
    template: list[dict],
    source_key: str,
    concept: str,
) -> list[dict]:
    """Apply source-specific definition overrides to a static template.

    Returns a new template list with definitions replaced where source-specific
    overrides exist. Variables unavailable in the source get a 'Not available
    in {source} data' note appended.
    """
    is_available = is_variable_available(source_key, concept, "")
    resolved = []

    for tmpl in template:
        entry = dict(tmpl)  # shallow copy
        var_name = entry["variable_name"]

        # Override definition if source-specific one exists
        entry["definition"] = get_definition(source_key, var_name, entry["definition"])

        # If concept category not available in source, add note
        if not is_available:
            if not entry["additional_notes"]:
                entry["additional_notes"] = f"May not be available in {source_key} data"
            else:
                entry["additional_notes"] += f"; May not be available in {source_key} data"

        resolved.append(entry)

    return resolved
