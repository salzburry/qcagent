"""Tests for program spec schema, HTML renderer, and Excel writer."""
import json
import tempfile
from pathlib import Path

from protocol_spec_assist.spec_output.spec_schema import ProgramSpec, build_program_spec
from protocol_spec_assist.spec_output.html_renderer import render_html, save_html


# ── ProgramSpec schema tests ──────────────────────────────────────────────────

def test_program_spec_defaults():
    spec = ProgramSpec(protocol_id="TEST001")
    assert spec.protocol_id == "TEST001"
    assert spec.inclusion_criteria == []
    assert spec.exclusion_criteria == []
    assert spec.endpoints == []
    assert spec.censoring_rules == []
    assert spec.generator_version == "0.3.0"


def test_program_spec_serializes():
    spec = ProgramSpec(
        protocol_id="TEST001",
        design_type="retrospective cohort",
        study_period_start="2016-01-01",
        study_period_end="2024-06-30",
        data_source="Flatiron LBCL",
    )
    data = spec.model_dump()
    assert data["protocol_id"] == "TEST001"
    assert data["design_type"] == "retrospective cohort"
    assert data["data_source"] == "Flatiron LBCL"


# ── build_program_spec tests ─────────────────────────────────────────────────

def _make_evidence_packs():
    """Create minimal evidence pack dicts for testing."""
    return {
        "index_date": {
            "protocol_id": "TEST001",
            "concept": "index_date",
            "candidates": [
                {
                    "candidate_id": "c1",
                    "snippet": "Index date is the first qualifying diagnosis date.",
                    "sponsor_term": "first qualifying diagnosis",
                    "llm_confidence": 0.85,
                    "page": 12,
                    "section_title": "Study Design",
                    "explicit": "explicit",
                }
            ],
            "overall_confidence": 0.85,
            "low_retrieval_signal": False,
            "contradictions_found": False,
        },
        "eligibility_inclusion": {
            "protocol_id": "TEST001",
            "concept": "eligibility_inclusion",
            "candidates": [
                {
                    "candidate_id": "inc1",
                    "snippet": "Patients must be aged 18 years or older.",
                    "sponsor_term": "Age >= 18",
                    "llm_confidence": 0.92,
                    "page": 8,
                    "section_title": "Eligibility",
                    "explicit": "explicit",
                },
                {
                    "candidate_id": "inc2",
                    "snippet": "Minimum 30 days of follow-up required.",
                    "sponsor_term": "Min follow-up 30d",
                    "llm_confidence": 0.78,
                    "page": 8,
                    "section_title": "Eligibility",
                    "explicit": "explicit",
                },
            ],
            "concept_metadata": {
                "per_candidate": {
                    "inc1": {"criterion_label": "Age >= 18", "domain": "demographic", "operational_detail": None, "lookback_window": None},
                    "inc2": {"criterion_label": "Min follow-up 30d", "domain": "enrollment", "operational_detail": "30 days from index", "lookback_window": None},
                }
            },
            "overall_confidence": 0.85,
            "low_retrieval_signal": False,
            "contradictions_found": False,
        },
        "study_period": {
            "protocol_id": "TEST001",
            "concept": "study_period",
            "candidates": [],
            "concept_metadata": {
                "study_period_start": "January 2016",
                "study_period_end": "June 2024",
                "data_source": "Flatiron LBCL",
                "data_source_version": "Q2 2024",
                "design_type": "retrospective cohort",
            },
            "overall_confidence": 0.90,
            "low_retrieval_signal": False,
            "contradictions_found": False,
        },
    }


def test_build_program_spec_from_packs():
    packs = _make_evidence_packs()
    spec = build_program_spec("TEST001", packs, protocol_title="Test Protocol")

    assert spec.protocol_id == "TEST001"
    assert spec.protocol_title == "Test Protocol"
    assert spec.index_date_definition == "Index date is the first qualifying diagnosis date."
    assert spec.index_date_confidence == 0.85
    assert spec.study_period_start == "January 2016"
    assert spec.study_period_end == "June 2024"
    assert spec.data_source == "Flatiron LBCL"
    assert spec.design_type == "retrospective cohort"
    assert len(spec.inclusion_criteria) == 2
    assert spec.inclusion_criteria[0].criterion_id == "INC-01"
    assert spec.inclusion_criteria[0].criterion_label == "Age >= 18"
    assert spec.inclusion_criteria[1].domain == "enrollment"
    assert "index_date" in spec.concepts_extracted
    assert "study_period" in spec.concepts_extracted


def test_build_program_spec_empty_packs():
    spec = build_program_spec("EMPTY", {})
    assert spec.protocol_id == "EMPTY"
    assert spec.inclusion_criteria == []
    assert spec.concepts_extracted == []


# ── HTML renderer tests ──────────────────────────────────────────────────────

def test_render_html_contains_key_elements():
    packs = _make_evidence_packs()
    spec = build_program_spec("TEST001", packs, protocol_title="Test Protocol")
    html = render_html(spec)

    assert "TEST001" in html
    assert "Test Protocol" in html
    assert "DRAFT" in html
    assert "Index Date" in html
    assert "Inclusion Criteria" in html
    assert "first qualifying diagnosis" in html
    assert "Age" in html and "18" in html
    assert "retrospective cohort" in html
    assert "Flatiron LBCL" in html


def test_save_html_creates_file():
    spec = ProgramSpec(protocol_id="TEST001")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_html(spec, f"{tmpdir}/test_spec.html")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "TEST001" in content


# ── Excel writer tests ───────────────────────────────────────────────────────

def test_save_excel_creates_file():
    try:
        import openpyxl  # noqa: F401
        from protocol_spec_assist.spec_output.excel_writer import save_excel
    except ImportError:
        import pytest
        pytest.skip("openpyxl not installed")

    packs = _make_evidence_packs()
    spec = build_program_spec("TEST001", packs)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_excel(spec, f"{tmpdir}/test_spec.xlsx")
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_excel_has_expected_tabs():
    try:
        from openpyxl import load_workbook
        from protocol_spec_assist.spec_output.excel_writer import save_excel
    except ImportError:
        import pytest
        pytest.skip("openpyxl not installed")

    packs = _make_evidence_packs()
    spec = build_program_spec("TEST001", packs)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_excel(spec, f"{tmpdir}/test_spec.xlsx")
        wb = load_workbook(path)
        tabs = wb.sheetnames
        assert "Overview" in tabs
        assert "Inclusion Criteria" in tabs
        assert "Exclusion Criteria" in tabs
        assert "Endpoints" in tabs
        assert "Censoring Rules" in tabs


def test_excel_inclusion_criteria_populated():
    try:
        from openpyxl import load_workbook
        from protocol_spec_assist.spec_output.excel_writer import save_excel
    except ImportError:
        import pytest
        pytest.skip("openpyxl not installed")

    packs = _make_evidence_packs()
    spec = build_program_spec("TEST001", packs)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_excel(spec, f"{tmpdir}/test_spec.xlsx")
        wb = load_workbook(path)
        ws = wb["Inclusion Criteria"]
        # Header row + 2 criteria rows
        assert ws.cell(row=1, column=1).value == "ID"
        assert ws.cell(row=2, column=1).value == "INC-01"
        assert ws.cell(row=3, column=1).value == "INC-02"
