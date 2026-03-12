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
