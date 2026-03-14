"""Tests for spec output layer: schema, HTML renderer, Excel writer (9-tab layout)."""

import json
import pytest
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.spec_output.spec_schema import (
    ProgramSpec, build_program_spec, SpecEntry, VariableRow, CriterionRow,
)
from protocol_spec_assist.spec_output.html_renderer import render_html


def _make_pack(concept: str, n_candidates: int = 2, **kwargs) -> EvidencePack:
    candidates = [
        EvidenceCandidate(
            candidate_id=f"c{i}",
            snippet=f"Evidence text {i} for {concept}",
            page=i + 1,
            sponsor_term=f"term_{i}",
            llm_confidence=0.9 - i * 0.1,
            explicit="explicit" if i == 0 else "inferred",
        )
        for i in range(n_candidates)
    ]
    return EvidencePack(
        protocol_id="P001",
        concept=concept,
        candidates=candidates,
        overall_confidence=0.85,
        **kwargs,
    )


# ── ProgramSpec schema tests ────────────────────────────────────────────────

def test_program_spec_defaults():
    spec = ProgramSpec()
    assert spec.protocol_id == ""
    assert spec.spec_version == "1.0.0"
    assert spec.generation_mode == "draft"
    assert spec.inclusion_criteria == []
    assert spec.exclusion_criteria == []
    assert spec.demographics == []
    assert spec.clinical_characteristics == []
    assert spec.biomarkers == []
    assert spec.lab_variables == []
    assert spec.treatment_variables == []
    assert spec.outcome_variables == []


def test_program_spec_serialization():
    spec = ProgramSpec(protocol_id="P001")
    data = spec.model_dump()
    assert data["protocol_id"] == "P001"
    # Roundtrip
    spec2 = ProgramSpec.model_validate(data)
    assert spec2.protocol_id == "P001"


def test_variable_row_model():
    row = VariableRow(
        time_period="STUDY_PD",
        variable="INDEX",
        label="Index date",
        values="date",
        definition="Date of initial diagnosis",
        additional_notes="Linked from DataPrep",
    )
    assert row.time_period == "STUDY_PD"
    assert row.variable == "INDEX"


# ── build_program_spec tests ────────────────────────────────────────────────

def test_build_spec_from_single_concept():
    packs = {"index_date": _make_pack("index_date")}
    spec = build_program_spec(packs, protocol_id="P001")
    assert spec.protocol_id == "P001"
    assert spec.generation_mode == "draft"
    # Index date should be in important_dates (Data Prep tab)
    assert len(spec.important_dates) >= 1
    assert any("Evidence text 0" in d.definition for d in spec.important_dates)


def test_build_spec_uses_selected_candidate():
    pack = _make_pack("index_date")
    pack.select_candidate("c1")
    packs = {"index_date": pack}
    spec = build_program_spec(packs, protocol_id="P001")
    assert spec.generation_mode == "reviewed"
    assert any("Evidence text 1" in d.definition for d in spec.important_dates)


def test_build_spec_with_all_concepts():
    packs = {
        "index_date": _make_pack("index_date"),
        "follow_up_end": _make_pack("follow_up_end"),
        "primary_endpoint": _make_pack("primary_endpoint"),
        "eligibility_inclusion": _make_pack("eligibility_inclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "demographic", "lookback_window": None, "operational_detail": None},
                "c1": {"domain": "clinical", "lookback_window": "12 months", "operational_detail": None},
            }}),
        "eligibility_exclusion": _make_pack("eligibility_exclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "treatment", "lookback_window": "6 months", "operational_detail": None},
            }}),
        "study_period": _make_pack("study_period",
            concept_metadata={
                "study_period_start": "2020-01-01",
                "study_period_end": "2023-12-31",
                "data_source": "Optum CDM",
                "data_source_version": "Q4 2023",
                "design_type": "retrospective_cohort",
            }),
        "censoring_rules": _make_pack("censoring_rules",
            concept_metadata={"per_candidate": {
                "c0": {"rule_type": "event_based", "applies_to": "primary endpoint"},
            }}),
    }
    spec = build_program_spec(packs, protocol_id="P001")

    # Data Prep tab: data source
    assert spec.data_source.data_source == "Optum CDM"
    assert spec.data_source.version == "Q4 2023"

    # Data Prep tab: important dates (index_date + follow_up_end + INIT)
    assert len(spec.important_dates) >= 2

    # Data Prep tab: time periods
    assert len(spec.time_periods) >= 1

    # StudyPop: inclusion criteria
    assert len(spec.inclusion_criteria) == 2
    assert spec.inclusion_criteria[0].domain == "demographic"
    assert spec.inclusion_criteria[1].lookback_window == "12 months"

    # StudyPop: exclusion criteria
    assert len(spec.exclusion_criteria) >= 1

    # Outcomes: primary endpoint + censoring rules
    assert len(spec.outcome_variables) >= 2

    # Cover: tab statuses
    assert len(spec.tab_statuses) == 10


def test_build_spec_empty_packs():
    spec = build_program_spec({}, protocol_id="P001")
    assert spec.protocol_id == "P001"
    assert spec.important_dates == []
    assert spec.outcome_variables == []


def test_build_spec_cover_tab_populated():
    spec = build_program_spec({}, protocol_id="P001")
    assert spec.study_info.study_id == "P001"
    assert len(spec.tab_statuses) == 10
    tab_names = [ts.tab for ts in spec.tab_statuses]
    assert "1.Cover" in tab_names
    assert "7.Outcomes" in tab_names


# ── HTML renderer tests ─────────────────────────────────────────────────────

def test_html_contains_protocol_id():
    spec = ProgramSpec(protocol_id="P001")
    html = render_html(spec)
    assert "P001" in html


def test_html_contains_all_tab_sections():
    packs = {
        "index_date": _make_pack("index_date"),
        "eligibility_inclusion": _make_pack("eligibility_inclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "demographic", "lookback_window": None, "operational_detail": None},
            }}),
    }
    spec = build_program_spec(packs, protocol_id="P001")
    html = render_html(spec)

    # All 9+ sections present
    assert "1. Cover" in html
    assert "2. QC Review" in html
    assert "3. Data Prep" in html
    assert "4. Study Population" in html
    assert "5A. Demographics" in html
    assert "5B. Clinical Characteristics" in html
    assert "5C. Biomarker Variables" in html
    assert "5D. Laboratory Variables" in html
    assert "6. Treatment Variables" in html
    assert "7. Outcomes" in html
    assert "Evidence text 0" in html


def test_html_qc_warnings():
    spec = ProgramSpec(
        protocol_id="P001",
        qc_warnings=["Low signal for follow_up_end"],
    )
    html = render_html(spec)
    assert "QC Warnings" in html
    assert "Low signal" in html


def test_html_draft_mode_label():
    spec = ProgramSpec(protocol_id="P001", generation_mode="draft")
    html = render_html(spec)
    assert "DRAFT" in html


def test_html_tab_navigation():
    spec = ProgramSpec(protocol_id="P001")
    html = render_html(spec)
    assert 'href="#cover"' in html
    assert 'href="#outcomes"' in html


# ── Excel writer tests ──────────────────────────────────────────────────────

def test_build_spec_with_variable_tab_packs():
    """Test that variable tab packs (demographics, etc.) map to correct spec fields."""
    packs = {
        "demographics": EvidencePack(
            protocol_id="P001", concept="demographics",
            candidates=[
                EvidenceCandidate(candidate_id="d1", snippet="patient/age_at_diagnosis",
                    canonical_term="AGE", sponsor_term="AGE", llm_confidence=0.9),
                EvidenceCandidate(candidate_id="d2", snippet="patient/sex",
                    canonical_term="SEX", sponsor_term="SEX", llm_confidence=0.85),
            ],
            overall_confidence=0.88,
            concept_metadata={"per_candidate": {
                "d1": {"time_period": "STUDY_PD", "variable_name": "AGE", "label": "Age at Index",
                        "values": "numeric", "code_lists_group": "", "additional_notes": ""},
                "d2": {"time_period": "STUDY_PD", "variable_name": "SEX", "label": "Sex",
                        "values": "Male; Female", "code_lists_group": "", "additional_notes": ""},
            }},
        ),
        "treatment_variables": EvidencePack(
            protocol_id="P001", concept="treatment_variables",
            candidates=[
                EvidenceCandidate(candidate_id="t1", snippet="Count of LOTs",
                    canonical_term="LOTN", sponsor_term="LOTN", llm_confidence=0.9),
            ],
            overall_confidence=0.85,
            concept_metadata={"per_candidate": {
                "t1": {"time_period": "FU", "variable_name": "LOTN", "label": "Number of LOTs",
                        "values": "numeric", "code_lists_group": "", "additional_notes": ""},
            }},
        ),
    }
    spec = build_program_spec(packs, protocol_id="P001")

    # Demographics now uses DemographicsWriter — expands into variable families
    # Core families: AGE (AGE, AGEN, AGEGR, AGEGRN), SEX (SEX, SEXN),
    # RACE (RACE, RACEN), ETHNICITY (ETH, ETHN) = 10 rows
    assert len(spec.demographics) >= 4  # at minimum core vars
    var_names = [r.variable for r in spec.demographics]
    assert "AGE" in var_names
    assert "SEX" in var_names

    # Treatment variables should be in spec.treatment_variables
    assert len(spec.treatment_variables) == 1
    assert spec.treatment_variables[0].variable == "LOTN"


def test_build_spec_all_variable_tabs():
    """Test all 5 variable tab concepts map correctly."""
    concept_fields = {
        "demographics": "demographics",
        "clinical_characteristics": "clinical_characteristics",
        "biomarkers": "biomarkers",
        "lab_variables": "lab_variables",
        "treatment_variables": "treatment_variables",
    }
    packs = {}
    for concept in concept_fields:
        packs[concept] = EvidencePack(
            protocol_id="P001", concept=concept,
            candidates=[
                EvidenceCandidate(candidate_id=f"{concept}_c1", snippet=f"def for {concept}",
                    canonical_term=f"VAR_{concept}", sponsor_term=f"VAR_{concept}", llm_confidence=0.8),
            ],
            overall_confidence=0.8,
            concept_metadata={"per_candidate": {
                f"{concept}_c1": {"time_period": "STUDY_PD", "variable_name": f"VAR_{concept}",
                                   "label": f"Test {concept}", "values": "numeric",
                                   "code_lists_group": "", "additional_notes": ""},
            }},
        )

    spec = build_program_spec(packs, protocol_id="P001")
    for concept, field in concept_fields.items():
        rows = getattr(spec, field)
        if concept == "demographics":
            # Demographics uses DemographicsWriter with family expansion
            assert len(rows) >= 4, f"{field} should have at least 4 rows (core families)"
            var_names = [r.variable for r in rows]
            assert "AGE" in var_names
        else:
            assert len(rows) == 1, f"{field} should have 1 row, got {len(rows)}"
            assert rows[0].variable == f"VAR_{concept}"


# ── Static template tests ──────────────────────────────────────────────────

def test_demographics_static_template():
    from protocol_spec_assist.concepts.demographics import STATIC_TEMPLATE, _build_static_only_pack
    # Verify template is exhaustive
    var_names = [t["variable_name"] for t in STATIC_TEMPLATE]
    assert "AGE" in var_names
    assert "SEX" in var_names
    assert "RACE" in var_names
    assert "BMI" in var_names
    assert "PRACTIC" in var_names
    assert len(STATIC_TEMPLATE) >= 15

    # Static-only pack should produce candidates
    pack = _build_static_only_pack("P001")
    assert len(pack.candidates) == len(STATIC_TEMPLATE)
    assert pack.model_used == "static_template"
    assert pack.concept_metadata["per_candidate"]


def test_demographics_static_template_with_source():
    from protocol_spec_assist.concepts.demographics import STATIC_TEMPLATE, _build_static_only_pack
    pack_cota = _build_static_only_pack("P001", data_source="cota")
    assert len(pack_cota.candidates) == len(STATIC_TEMPLATE)
    # COTA-specific AGE definition should differ from generic
    age_cand = next(c for c in pack_cota.candidates if c.sponsor_term == "AGE")
    assert "COTA" in age_cand.snippet or "date_of_birth" in age_cand.snippet

    pack_generic = _build_static_only_pack("P001", data_source="generic")
    assert len(pack_generic.candidates) == len(STATIC_TEMPLATE)


def test_clinical_chars_static_template_with_source():
    from protocol_spec_assist.concepts.clinical_characteristics import _build_static_only_pack
    pack = _build_static_only_pack("P001", data_source="optum_cdm")
    # Should still produce candidates even though some vars unavailable in claims
    assert len(pack.candidates) > 0


def test_lab_variables_static_template_with_source():
    from protocol_spec_assist.concepts.lab_variables import _build_static_only_pack
    pack = _build_static_only_pack("P001", data_source="quest")
    assert len(pack.candidates) > 0


def test_clinical_chars_static_template():
    from protocol_spec_assist.concepts.clinical_characteristics import STATIC_TEMPLATE, _build_static_only_pack
    var_names = [t["variable_name"] for t in STATIC_TEMPLATE]
    assert "ECOG" in var_names
    assert "STAGE" in var_names
    assert "CCI" in var_names
    assert "BSYMPT" in var_names
    pack = _build_static_only_pack("P001")
    assert len(pack.candidates) == len(STATIC_TEMPLATE)


def test_biomarkers_static_template():
    from protocol_spec_assist.concepts.biomarkers import STATIC_TEMPLATE, _build_static_only_pack
    var_names = [t["variable_name"] for t in STATIC_TEMPLATE]
    assert "BCL2" in var_names
    assert "MYC" in var_names
    assert "CD20" in var_names
    assert "DBLHIT" in var_names
    # Each marker has 5 sub-variables + 2 composites
    assert len(STATIC_TEMPLATE) >= 70
    pack = _build_static_only_pack("P001")
    assert len(pack.candidates) == len(STATIC_TEMPLATE)


def test_lab_variables_static_template():
    from protocol_spec_assist.concepts.lab_variables import STATIC_TEMPLATE, _build_static_only_pack
    var_names = [t["variable_name"] for t in STATIC_TEMPLATE]
    assert "LABNEU" in var_names
    assert "LABPLA" in var_names
    assert "LABLDH" in var_names
    # Each lab has 4 sub-variables × 12 labs = 48
    assert len(STATIC_TEMPLATE) >= 40
    pack = _build_static_only_pack("P001")
    assert len(pack.candidates) == len(STATIC_TEMPLATE)


def test_treatment_variables_static_template():
    from protocol_spec_assist.concepts.treatment_variables import STATIC_TEMPLATE, _build_static_only_pack
    var_names = [t["variable_name"] for t in STATIC_TEMPLATE]
    assert "LOTN" in var_names
    assert "LOT1SD" in var_names
    assert "LOT1" in var_names
    assert "CSD" in var_names
    assert "TTNT" in var_names
    pack = _build_static_only_pack("P001")
    assert len(pack.candidates) == len(STATIC_TEMPLATE)


def test_demographics_merge_with_static():
    """Test that merge preserves LLM variables and fills gaps from template."""
    from protocol_spec_assist.concepts.demographics import (
        DemographicsExtraction, _merge_with_static_template, STATIC_TEMPLATE,
    )
    VarClass = DemographicsExtraction.VariableExtraction
    # LLM found only AGE with protocol-specific definition
    llm_vars = [VarClass(
        chunk_id="ch1", time_period="STUDY_PD", variable_name="AGE",
        label="Age at Index (custom)", values="integer",
        definition="Custom AGE definition from protocol",
        explicit="explicit", confidence=0.95,
        reasoning="Found in protocol",
    )]
    merged, unmapped = _merge_with_static_template(llm_vars, STATIC_TEMPLATE)
    # Should have all template vars (LLM's custom AGE replaces the static one)
    assert len(merged) == len(STATIC_TEMPLATE)
    assert len(unmapped) == 0  # AGE is in template, no extras
    # AGE should use LLM definition
    age_var = next(v for v in merged if v.variable_name == "AGE")
    assert age_var.definition == "Custom AGE definition from protocol"
    assert age_var.confidence == 0.95
    # SEX should be static default
    sex_var = next(v for v in merged if v.variable_name == "SEX")
    assert sex_var.explicit == "inferred"
    assert sex_var.confidence == 0.5


def test_demographics_merge_unmapped_goes_to_bucket():
    """Test that LLM variables not in template go to unmapped bucket, not into merged."""
    from protocol_spec_assist.concepts.demographics import (
        DemographicsExtraction, _merge_with_static_template, STATIC_TEMPLATE,
    )
    VarClass = DemographicsExtraction.VariableExtraction
    # LLM found ECOG (does not belong in demographics template)
    llm_vars = [VarClass(
        chunk_id="ch1", time_period="PRE_INT", variable_name="ECOG",
        label="ECOG Performance Status", values="0;1;2;3;4",
        definition="ECOG from clinical notes",
        explicit="explicit", confidence=0.9,
        reasoning="Found in protocol",
    )]
    merged, unmapped = _merge_with_static_template(llm_vars, STATIC_TEMPLATE)
    # ECOG should NOT be in merged (it's not a demographics variable)
    assert all(v.variable_name != "ECOG" for v in merged)
    # ECOG should be in the unmapped bucket
    assert len(unmapped) == 1
    assert unmapped[0].variable_name == "ECOG"


# ── Excel writer tests ──────────────────────────────────────────────────────

def test_excel_import():
    """Just verify the excel_writer module can be imported."""
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl not installed")
    from protocol_spec_assist.spec_output.excel_writer import save_excel
    assert callable(save_excel)


def test_excel_save_9_tabs(tmp_path):
    """Test that save_excel creates a valid .xlsx file with 9+ tabs."""
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl not installed")

    from protocol_spec_assist.spec_output.excel_writer import save_excel

    packs = {
        "index_date": _make_pack("index_date"),
        "primary_endpoint": _make_pack("primary_endpoint"),
        "eligibility_inclusion": _make_pack("eligibility_inclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "demographic", "lookback_window": None, "operational_detail": None},
            }}),
        "censoring_rules": _make_pack("censoring_rules",
            concept_metadata={"per_candidate": {
                "c0": {"rule_type": "event_based", "applies_to": "all"},
            }}),
    }
    spec = build_program_spec(packs, protocol_id="P001")

    out_path = str(tmp_path / "test_spec.xlsx")
    result = save_excel(spec, out_path)
    assert result == out_path

    wb = openpyxl.load_workbook(out_path)
    expected_tabs = [
        "1.Cover", "2.QC Review", "3.Data Prep", "4.StudyPop",
        "5A.Demos", "5B.ClinChars", "5C.BioVars", "5D.LabVars",
        "6.TreatVars", "7.Outcomes",
    ]
    for tab_name in expected_tabs:
        assert tab_name in wb.sheetnames, f"Missing tab: {tab_name}"
